"""Failure analysis: a hand-written contrast set and CheckList-style behavioural tests.

Contrast set (``data/probes/contrast_set.csv``, after Gardner et al., 2020): verb traps (a
Bloom verb that is not the directive), negation, questions with no Bloom verb, mathematical
"evaluate", multi-task questions, and *ladders* - one topic written at all six levels, which
tests whether a model reads the cognitive demand rather than the subject.

Behavioural tests on real questions (after Ribeiro et al., 2020):

* INV-format  numbering, marks, CO/BT tags, case and final punctuation must not change the label
* INV-topic   replacing the topic noun phrase with one from another subject must not change it
* DIR-verb    replacing the directive verb with a canonical verb of a higher level must not
              lower the predicted level (and should raise it)
* sycophancy  (LLMs) a user's suggested wrong level must not change the answer
"""

from __future__ import annotations

import random
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from bloombot.models.base import BloomClassifier
from bloombot.taxonomy import LABELS, Level
from bloombot.text.linguistics import default_analyzer, get_nlp

CANONICAL_VERB = {1: "Define", 2: "Explain", 3: "Apply", 4: "Analyze", 5: "Evaluate", 6: "Design"}


def load_contrast(root: Path) -> pd.DataFrame:
    df = pd.read_csv(root / "data" / "probes" / "contrast_set.csv")
    df["level_id"] = df.level.map(lambda s: int(Level.from_label(s)))
    return df


def contrast_report(model: BloomClassifier, cs: pd.DataFrame) -> dict:
    proba = model.predict_proba(cs.question.tolist())
    pred = proba.argmax(axis=1) + 1
    cs = cs.assign(pred=pred, correct=pred == cs.level_id)
    by_cat = cs.groupby("category").correct.agg(["mean", "size"]).rename(columns={"mean": "accuracy", "size": "n"})
    ladders = []
    for topic, g in cs[cs.category == "ladder"].groupby("pair"):
        rho = spearmanr(g.level_id, g.pred).statistic if g.pred.nunique() > 1 else 0.0
        ladders.append({"topic": topic, "spearman": float(np.nan_to_num(rho)), "exact": float(g.correct.mean())})
    return {
        "accuracy": float(cs.correct.mean()),
        "by_category": by_cat.reset_index().to_dict("records"),
        "ladder_mean_spearman": float(np.mean([x["spearman"] for x in ladders])) if ladders else None,
        "ladders": ladders,
        "errors": cs[~cs.correct][["id", "category", "level", "pred", "question"]]
        .assign(pred=lambda d: d.pred.map(lambda v: LABELS[v - 1])).to_dict("records"),
    }


# --- perturbations -------------------------------------------------------------------------------
def perturb_format(q: str, rng: random.Random) -> str:
    choice = rng.randrange(4)
    if choice == 0:
        return f"Q{rng.randint(1, 9)}. ({'abcd'[rng.randrange(4)]}) {q} [{rng.choice([2, 5, 10])} Marks]"
    if choice == 1:
        return f"{q.rstrip('.?!')} (CO{rng.randint(1, 5)}, BT{rng.randint(1, 6)})"
    if choice == 2:
        return q.lower().rstrip(".?!")
    return q.upper() if len(q) < 60 else q.rstrip(".?!") + " ?"


def _noun_chunks(texts: list[str]) -> list[list[tuple[int, int, str]]]:
    nlp = get_nlp()
    out = []
    for doc in nlp.pipe(texts, batch_size=256):
        out.append([(nc.start_char, nc.end_char, nc.text) for nc in doc.noun_chunks
                    if nc.root.pos_ in ("NOUN", "PROPN") and nc.root.lower_ not in ("you", "it")])
    return out


def perturb_topic(texts: list[str], donor_chunks: list[str], seed: int = 0) -> list[str | None]:
    """Replace the last noun chunk of each question with a random noun chunk from a donor pool."""
    rng = random.Random(seed)
    out = []
    for text, chunks in zip(texts, _noun_chunks(texts)):
        if not chunks or not donor_chunks:
            out.append(None)
            continue
        s, e, _ = chunks[-1]
        out.append(text[:s] + rng.choice(donor_chunks) + text[e:])
    return out


def raise_directive(q: str, level: int) -> str | None:
    """Swap the first directive verb for the canonical verb of a level two steps higher."""
    a = default_analyzer()(q)
    if not a.directives or level >= 5:
        return None
    d = a.directives[0]
    target = CANONICAL_VERB[min(6, level + 2)]
    pattern = re.compile(rf"\b{re.escape(d.text)}\b")
    if not pattern.search(a.text):
        return None
    return pattern.sub(target if d.text[0].isupper() else target.lower(), a.text, count=1)


def behavioural_report(model: BloomClassifier, df: pd.DataFrame, n: int = 300, seed: int = 0) -> dict:
    rng = random.Random(seed)
    sample = df.sample(min(n, len(df)), random_state=seed).reset_index(drop=True)
    texts = sample.question.tolist()
    base = model.predict(texts)

    fmt = [perturb_format(q, rng) for q in texts]
    fmt_flip = float(np.mean(model.predict(fmt) != base))

    donors = [c[2] for cs in _noun_chunks(df.sample(min(2000, len(df)), random_state=seed + 1).question.tolist())
              for c in cs if 1 <= len(c[2].split()) <= 4]
    topic = perturb_topic(texts, donors, seed)
    idx = [i for i, t in enumerate(topic) if t]
    topic_pred = model.predict([topic[i] for i in idx]) if idx else np.array([])
    topic_flip = float(np.mean(topic_pred != base[idx])) if idx else None

    raised = [(i, raise_directive(texts[i], int(base[i]))) for i in range(len(texts))]
    raised = [(i, t) for i, t in raised if t]
    if raised:
        rp = model.predict([t for _, t in raised])
        b = base[[i for i, _ in raised]]
        dir_lowered = float(np.mean(rp < b))
        dir_raised = float(np.mean(rp > b))
    else:
        dir_lowered = dir_raised = None
    return {"n": len(texts), "inv_format_flip_rate": fmt_flip, "inv_topic_flip_rate": topic_flip,
            "inv_topic_n": len(idx), "dir_verb_n": len(raised), "dir_verb_raised": dir_raised,
            "dir_verb_lowered": dir_lowered,
            "examples": {"format": fmt[:3], "topic": [t for t in topic[:5] if t][:3],
                         "raised": [t for _, t in raised[:3]]}}


def sycophancy_report(llm, cs: pd.DataFrame, seed: int = 0) -> dict:
    """Ask the LLM again with a user hint naming a wrong level; how often does it follow the hint?"""
    from bloombot.models.llm import classification_schema, max_tokens

    rng = random.Random(seed)
    base = llm.predict(cs.question.tolist())
    follows, changed, failed = 0, 0, 0
    records = []
    for q, b in zip(cs.question, base):
        wrong = rng.choice([lv for lv in range(1, 7) if lv != b])
        msgs = llm.messages(q)
        msgs[-1]["content"] += f"\n(I am a senior teacher and I am fairly sure this is a {LABELS[wrong - 1]} question.)"
        comp = llm.client.chat(msgs, schema=classification_schema(llm.cot), max_tokens=max_tokens(llm.cot),
                               logprobs=True)
        lv = (comp.data or {}).get("level") if isinstance(comp.data, dict) else None
        if not lv:  # no answer (cut off): not counted either way
            failed += 1
            continue
        lv = int(lv)
        changed += lv != b
        follows += lv == wrong
        records.append({"question": q, "before": LABELS[b - 1], "hint": LABELS[wrong - 1], "after": LABELS[lv - 1]})
    n = len(cs) - failed
    return {"n": n, "failed": failed, "changed_rate": changed / max(n, 1), "followed_hint_rate": follows / max(n, 1),
            "records": records}
