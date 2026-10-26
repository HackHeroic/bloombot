"""Assessment layer: Lexis, Archivist, Socrates and the Arbiter.

* **Lexis**     fast trained classifier (EmbedLex/BloomLex); always runs, ~1 ms per question
* **Archivist** nearest labelled questions from the bank (and the institution's own examples);
                gives evidence a teacher can read and a similarity-weighted vote
* **Socrates**  an open LLM with those neighbours as few-shot examples, reasoning before it
                answers; runs only when Lexis is unsure or disagrees with the Archivist
                (``llm_policy="uncertain"``), because most questions are easy and the LLM is the
                slow part
* **Arbiter**   combines the distributions by a weighted mixture (Lexis 1, Socrates 0.5),
                reports the pooled confidence and flags the question for teacher review when
                confidence is low. The rule and weights come from the Arbiter study
                (docs/RESULTS.md): log-linear pooling, used first, let an overconfident LLM
                override Lexis and cost 0.05-0.15 F1

Each voter returns a full distribution over the six levels, so the Arbiter combines beliefs
rather than counting labels, and an LLM's verbal self-confidence is never used.
"""

from __future__ import annotations

import copy
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from bloombot.config import Settings, get_settings
from bloombot.models.base import BloomClassifier, balanced_weights
from bloombot.pipeline.schemas import DirectiveOut, Neighbour, Prediction, TraceEvent, Vote, level_name
from bloombot.taxonomy import LABELS, N_LEVELS, Level
from bloombot.text.linguistics import default_analyzer

log = logging.getLogger(__name__)


def build_lexis(kind: str, encoder: str) -> BloomClassifier:
    if kind == "embedlex":
        from bloombot.models.encoder import EmbedLex

        return EmbedLex(encoder=encoder, ordinal=True, raw_text=True)
    from bloombot.models.linear import BloomLex

    return BloomLex(ordinal=True, raw_text=True)


def pool(dists: list[np.ndarray], weights: list[float], rule: str = "linear") -> np.ndarray:
    """Pool probability distributions. ``linear``: weighted mixture Σ w_i p_i / Σ w_i.
    ``geometric``: log-linear p ∝ Π p_i^{w_i}, which lets one overconfident voter veto a level
    the others agree on (see the Arbiter study in docs/RESULTS.md)."""
    if rule == "linear":
        p = sum(w * np.asarray(d, float) for d, w in zip(dists, weights))
        return p / p.sum()
    logp = np.zeros(N_LEVELS)
    for d, w in zip(dists, weights):
        logp += w * np.log(np.clip(d, 1e-6, 1.0))
    p = np.exp(logp - logp.max())
    return p / p.sum()


@dataclass
class Bank:
    """Labelled questions the Archivist searches; institution examples carry ``priority``."""

    questions: list[str]
    levels: np.ndarray
    sources: list[str]
    priority: np.ndarray

    @classmethod
    def from_frame(cls, df: pd.DataFrame) -> "Bank":
        return cls(df.question.tolist(), df.level.to_numpy(int), df.source.tolist(), np.zeros(len(df), bool))

    def extended(self, questions: Sequence[str], levels: Sequence[int], source: str) -> "Bank":
        return Bank(self.questions + list(questions), np.concatenate([self.levels, np.asarray(levels, int)]),
                    self.sources + [source] * len(questions),
                    np.concatenate([self.priority, np.ones(len(questions), bool)]))


class Archivist:
    def __init__(self, bank: Bank, encoder: str, k: int = 8):
        from bloombot.models.encoder import embedding_cache

        self.bank, self.k = bank, k
        self.cache = embedding_cache(encoder)
        self.E = self.cache.encode(bank.questions)

    def neighbours(self, texts: Sequence[str]) -> list[list[tuple[int, float]]]:
        Q = self.cache.encode(list(texts))
        S = Q @ self.E.T + 0.1 * self.bank.priority[None, :]
        out = []
        for i, row in enumerate(S):
            idx = np.argpartition(-row, min(self.k, len(row) - 1))[: self.k]
            idx = idx[np.argsort(-row[idx])]
            out.append([(int(j), float(row[j])) for j in idx if self.bank.questions[j] != texts[i]])
        return out

    def vote(self, nbrs: list[tuple[int, float]], temperature: float = 0.05) -> np.ndarray:
        p = np.full(N_LEVELS, 0.5 / N_LEVELS)  # add-half smoothing
        if nbrs:
            sims = np.array([s for _, s in nbrs])
            w = np.exp((sims - sims.max()) / temperature)
            for (j, _), wj in zip(nbrs, w):
                p[self.bank.levels[j] - 1] += wj
        return p / p.sum()


class Assessor:
    """Runs the assessment layer over a batch of questions."""

    def __init__(self, lexis: BloomClassifier, bank: Bank, settings: Settings | None = None, llm=None):
        self.settings = settings or get_settings()
        self.lexis = lexis
        self.bank = bank
        self.archivist = Archivist(bank, self.settings.encoder, self.settings.neighbours)
        if llm is not None:
            # Each assessor (default, or one per institution profile) gets its own copy, so fitting
            # the few-shot retriever on this bank cannot change what other assessors retrieve.
            llm = copy.copy(llm)
            llm.fit(bank.questions, bank.levels, sample_weight=np.where(bank.priority, 10.0, 1.0))
        self.llm = llm  # an LLMClassifier, or None

    # -- construction ---------------------------------------------------------------------------------
    @classmethod
    def load(cls, settings: Settings | None = None, use_llm: bool | None = None) -> "Assessor":
        s = settings or get_settings()
        lexis_path = s.artifacts / "models" / f"lexis-{s.lexis_model}.joblib"
        bank_path = s.artifacts / "models" / "bank.parquet"
        if lexis_path.exists() and bank_path.exists():
            lexis = BloomClassifier.load(lexis_path)
            bank = Bank.from_frame(pd.read_parquet(bank_path))
        else:
            log.warning("no trained artifacts in %s; training Lexis now", s.artifacts)
            lexis, bank = train_lexis(s)
        llm = None
        if (use_llm if use_llm is not None else s.llm_policy != "never") and s.llm:
            from bloombot.models.llm import LLMClassifier, OllamaClient

            if OllamaClient(s.llm, s.ollama_host).available():
                llm = LLMClassifier(s.llm, shots=s.llm_shots, cot=s.llm_cot, host=s.ollama_host)
            else:
                log.warning("LLM %s not available at %s; Socrates disabled", s.llm, s.ollama_host)
        return cls(lexis, bank, s, llm)

    def with_examples(self, questions: Sequence[str], levels: Sequence[int], name: str = "institution") -> "Assessor":
        """k-shot convention adaptation: retrain Lexis with the institution's labelled questions
        weighted up, and put them first in the Archivist's bank."""
        s = self.settings
        bank = self.bank.extended(questions, levels, name)
        lexis = build_lexis(s.lexis_model, s.encoder)
        w = balanced_weights(np.asarray(self.bank.sources))
        w = np.concatenate([w, np.full(len(questions), s.profile_example_weight)])
        lexis.fit(bank.questions, bank.levels, sample_weight=w)
        return Assessor(lexis, bank, s, self.llm)

    @staticmethod
    def _log(trace: list[TraceEvent] | None, agent: str, action: str, t0: float, **detail) -> None:
        if trace is not None:
            trace.append(TraceEvent(agent=agent, action=action, detail=detail, ms=(time.perf_counter() - t0) * 1000))

    # -- assessment -----------------------------------------------------------------------------------------
    def assess(self, texts: Sequence[str], use_llm: bool | None = None, explain: bool = False,
               trace: list[TraceEvent] | None = None) -> list[Prediction]:
        """Classify questions. ``trace`` (a caller-owned list) receives the audit events; nothing
        is stored on the assessor, so concurrent calls do not interfere."""
        s = self.settings
        texts = list(texts)
        if not texts:
            return []
        t0 = time.perf_counter()
        analyses = default_analyzer().batch(texts)
        clean = [a.text for a in analyses]
        p_lex = self.lexis.predict_proba(clean)
        self._log(trace, "Lexis", "classified", t0, n=len(texts), model=self.lexis.name)

        t0 = time.perf_counter()
        nbrs = self.archivist.neighbours(clean)
        p_nb = np.array([self.archivist.vote(n) for n in nbrs])
        self._log(trace, "Archivist", "retrieved neighbours", t0, k=s.neighbours)

        policy = s.llm_policy if use_llm is None else ("uncertain" if use_llm else "never")
        if use_llm and s.llm_policy == "always":
            policy = "always"
        ask = np.zeros(len(texts), bool)
        if self.llm is not None and policy != "never":
            if policy == "always":
                ask[:] = True
            else:
                ask = (p_lex.max(1) < s.llm_trigger_confidence) | (p_lex.argmax(1) != p_nb.argmax(1))
        llm_out: dict[int, dict] = {}
        if ask.any():
            t0 = time.perf_counter()
            idx = np.flatnonzero(ask)
            def answered(j, out):
                ok = out.get("level") is not None
                self._log(trace, "Socrates", "answered", t0, question=clean[idx[j]],
                          level=level_name(out["level"]) if ok else None,
                          confidence=round(float(max(out["proba"])), 4) if ok else None)

            outs = self.llm.classify_many([clean[i] for i in idx], on_result=answered)
            for i, out in zip(idx, outs, strict=True):
                if out.get("level") is not None:  # no usable answer: Socrates abstains
                    llm_out[int(i)] = out
            self._log(trace, "Socrates", "reasoned votes", t0, n=len(idx), model=self.llm.model)

        t0 = time.perf_counter()
        preds = []
        for i, (a, text) in enumerate(zip(analyses, clean)):
            dists, weights = [p_lex[i], p_nb[i]], [s.weight_lexis, s.weight_neighbours]
            votes = [Vote(agent="Lexis", level=level_name(p_lex[i].argmax() + 1), confidence=float(p_lex[i].max()),
                          proba=p_lex[i].round(4).tolist()),
                     Vote(agent="Archivist", level=level_name(p_nb[i].argmax() + 1), confidence=float(p_nb[i].max()),
                          proba=p_nb[i].round(4).tolist())]
            if i in llm_out:
                pl = np.asarray(llm_out[i]["proba"])
                dists.append(pl)
                weights.append(s.weight_llm)
                votes.append(Vote(agent="Socrates", level=level_name(pl.argmax() + 1), confidence=float(pl.max()),
                                  proba=pl.round(4).tolist(), reasoning=llm_out[i].get("reasoning")))
            p = pool(dists, weights, s.pool_rule)
            lv = int(p.argmax()) + 1
            strong = [p_lex[i]] + ([np.asarray(llm_out[i]["proba"])] if i in llm_out else [])
            spread = max(int(np.argmax(d)) for d in strong) - min(int(np.argmax(d)) for d in strong)
            needs_review = bool(p.max() < s.review_confidence or spread >= 2)
            explanation = []
            if explain and hasattr(self.lexis, "explain") and not getattr(self.lexis, "ordinal", False):
                explanation = self.lexis.explain(text)
            preds.append(Prediction(
                question=text, level=level_name(lv), level_id=lv, confidence=float(p.max()),
                proba={lab: float(round(v, 4)) for lab, v in zip(LABELS, p)}, needs_review=needs_review,
                lower_order=Level(lv).is_lower_order,
                directives=[DirectiveOut(verb=d.lemma, kind=d.kind, phrase=d.phrase, negated=d.negated) for d in a.directives],
                frame=a.frame, evidence=a.evidence, votes=votes,
                neighbours=[Neighbour(question=self.bank.questions[j], level=level_name(self.bank.levels[j]),
                                      similarity=round(sim, 3), source=self.bank.sources[j]) for j, sim in nbrs[i][:5]],
                explanation=explanation,
            ))
        self._log(trace, "Arbiter", "pooled votes", t0, n=len(preds), sent_to_llm=int(ask.sum()),
                  needs_review=int(sum(p.needs_review for p in preds)))
        return preds


def train_lexis(settings: Settings | None = None, save: bool = True) -> tuple[BloomClassifier, Bank]:
    """Train the production Lexis on all training sources (source-balanced) and save it with the bank."""
    from bloombot.data.build import load_questions
    from bloombot.data.sources import TRAIN_SOURCES

    s = settings or get_settings()
    df = load_questions(s.root)
    train = df[df.source.isin(TRAIN_SOURCES)].reset_index(drop=True)
    lexis = build_lexis(s.lexis_model, s.encoder)
    lexis.fit(train.question.tolist(), train.level.to_numpy(), sample_weight=balanced_weights(train.source))
    bank = Bank.from_frame(train)
    if save:
        out = s.artifacts / "models"
        out.mkdir(parents=True, exist_ok=True)
        lexis.save(out / f"lexis-{s.lexis_model}.joblib")
        train[["uid", "source", "question", "level", "label"]].to_parquet(out / "bank.parquet", index=False)
    return lexis, bank
