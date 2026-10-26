"""Design and verification layers: Composer, Level check, Critic, Student.

* **Composer**     rewrites a question to a target level on the same topic. With an LLM it is
                   prompted with the level's definition, typical directive verbs, two retrieved
                   example questions already at the target level and feedback from failed
                   attempts. Without an LLM a transparent template is used.
* **Level check**  the rewrite is classified by the assessment layer - not by the model that
                   wrote it, since self-verification is biased toward accepting its own output.
* **Critic**       rule checks: one well-formed question of sensible length, no answer or tags,
                   same topic (content-word overlap or embedding similarity with the original),
                   not a near-duplicate of the original or of another question in the paper.
* **Student**      for Analyze and above: an LLM asked whether recalling memorised material
                   alone answers the question; such "fake higher-order" questions are rejected.

A rewrite is accepted only if every check passes; otherwise the Composer retries with the
reasons, up to ``max_attempts``, and then the job goes to the teacher.
"""

from __future__ import annotations

import re
import time
from typing import Sequence

import numpy as np

from bloombot.config import Settings, get_settings
from bloombot.pipeline.schemas import Attempt, CheckResult, RewriteJob, RewriteResult, TraceEvent, level_name
from bloombot.taxonomy import DESCRIPTIONS, Level
from bloombot.text.linguistics import CUE_WORDS, content_words, get_nlp, topic_overlap

TARGET_VERBS = {
    Level.REMEMBER: "define, list, state, name, recall, identify",
    Level.UNDERSTAND: "explain (in your own words), summarise, classify, give an example of, interpret",
    Level.APPLY: "calculate, solve, use ... to ..., demonstrate, apply ... to the given case",
    Level.ANALYZE: "compare, differentiate, examine, analyse the given ..., identify the cause of",
    Level.EVALUATE: "justify, critique, assess, judge, recommend with reasons, which is better and why",
    Level.CREATE: "design, propose, develop, formulate, construct a new ...",
}

_TEMPLATES = {
    Level.REMEMBER: "Define {topic}.",
    Level.UNDERSTAND: "Explain {topic} in your own words, with an example.",
    Level.APPLY: "Apply {topic} to a given practical case and show each step of your working.",
    Level.ANALYZE: "Compare {topic} with a closely related alternative and analyse how they differ.",
    Level.EVALUATE: "Evaluate the strengths and limitations of {topic}, and justify when it is the better choice.",
    Level.CREATE: "Design a new solution that uses {topic} for a realistic scenario, and explain your design choices.",
}

COMPOSER_SYSTEM = """You write university exam questions. Rewrite the given question so that it \
targets a specific level of the revised Bloom's taxonomy, keeping the same topic and course content.

Requirements:
- exactly one question, at most 60 words, answerable in a written exam
- same subject matter as the original; you may add a short concrete scenario or data if the level needs it
- do not include the answer, marks, or any level/CO tags
- the main instruction must demand the target cognitive process, not merely mention its verb"""

STUDENT_SYSTEM = """You are a strong student who has memorised the textbook but has not practised \
applying it. Decide whether you could fully answer the exam question by recalling memorised \
facts, definitions or textbook explanations alone, without analysing, judging or producing \
anything new."""


def topic_phrase(text: str) -> str:
    doc = get_nlp()(text)
    chunks = [nc for nc in doc.noun_chunks
              if nc.root.pos_ in ("NOUN", "PROPN") and nc.root.lemma_.lower() not in CUE_WORDS]
    if not chunks:
        return "the given topic"
    best = max(chunks, key=lambda nc: len(nc.text))
    t = re.sub(r"^(?:the|a|an|its|their|your)\s+", "", best.text, flags=re.I)
    return f"the {t}" if not t.lower().startswith("the ") else t


class Rewriter:
    def __init__(self, assessor, llm_client=None, settings: Settings | None = None,
                 trace: list[TraceEvent] | None = None):
        self.assessor = assessor
        self.client = llm_client  # an OllamaClient or None
        self.settings = settings or get_settings()
        self.trace: list[TraceEvent] = trace if trace is not None else []

    def _log(self, agent, action, t0, **detail):
        self.trace.append(TraceEvent(agent=agent, action=action, detail=detail, ms=(time.perf_counter() - t0) * 1000))

    # -- Composer -----------------------------------------------------------------------------------------
    def _examples(self, text: str, target: int, n: int = 2) -> list[str]:
        arch = self.assessor.archivist
        Q = arch.cache.encode([text])[0]
        sims = arch.E @ Q
        mask = arch.bank.levels == target
        idx = np.flatnonzero(mask)
        if not len(idx):
            return []
        top = idx[np.argsort(-sims[idx])[:n]]
        return [arch.bank.questions[i] for i in top]

    def compose(self, job: RewriteJob, feedback: list[str], context: str | None = None) -> tuple[str, str, str | None]:
        target = Level.from_label(job.to_level)
        if self.client is None:
            return _TEMPLATES[target].format(topic=topic_phrase(job.original)), "template", None
        ex = self._examples(job.original, int(target))
        user = (f"Original question ({job.from_level}): {job.original}\n"
                + (f"Context given with the question: {context}\n" if context else "")
                + f"\nTarget level: {int(target)}. {target.label} - {DESCRIPTIONS[target]}\n"
                f"Typical instructions at this level: {TARGET_VERBS[target]}\n"
                + ("Example questions at this level (other topics):\n" + "\n".join(f"- {e}" for e in ex) + "\n" if ex else "")
                + (f"Marks available: {job.marks:g}\n" if job.marks else "")
                + ("\nYour previous attempt was rejected because:\n" + "\n".join(f"- {f}" for f in feedback) if feedback else ""))
        schema = {"type": "object", "properties": {"question": {"type": "string"}, "rationale": {"type": "string"}},
                  "required": ["question", "rationale"]}
        comp = self.client.chat([{"role": "system", "content": COMPOSER_SYSTEM}, {"role": "user", "content": user}],
                                schema=schema, temperature=0.4 if feedback else 0.2, max_tokens=300,
                                seed=len(feedback))
        data = comp.data if isinstance(comp.data, dict) else {}
        q = str(data.get("question", "")).strip()
        return q, "llm", data.get("rationale")

    # -- verification -------------------------------------------------------------------------------------
    def level_check(self, text: str, target: int):
        # Fast models only: the LLM that wrote the rewrite must not also grade it.
        pred = self.assessor.assess([text], use_llm=False, trace=self.trace)[0]
        p_target = pred.proba[level_name(target)]
        ok = pred.level_id == target or p_target >= self.settings.level_check_min_proba
        return CheckResult(name="level", passed=ok, score=round(p_target, 3),
                           detail=f"classified as {pred.level} (P({level_name(target)})={p_target:.2f})"), pred

    def critic(self, text: str, original: str, others: Sequence[str]) -> list[CheckResult]:
        s = self.settings
        out = []
        words = len(text.split())
        form_ok = 5 <= words <= 80 and text.count("?") <= 2 and not re.search(r"\banswer\s*:|\b(?:CO|BT)\s*\d", text, re.I)
        out.append(CheckResult(name="form", passed=form_ok, detail=f"{words} words"))
        a, b = content_words(original), content_words(text)
        overlap = topic_overlap(a, b)
        cache = self.assessor.archivist.cache
        E = cache.encode([original, text, *others])
        cos_orig = float(E[0] @ E[1])
        # Content words are the reliable signal; similarity alone is fooled by shared phrasing.
        topic_ok = overlap >= s.topic_min_overlap if a else cos_orig >= 0.6
        out.append(CheckResult(name="topic", passed=topic_ok, score=round(overlap, 3),
                               detail=f"content-word overlap {overlap:.2f}, similarity {cos_orig:.2f}"))
        max_other = float((E[2:] @ E[1]).max()) if len(others) else 0.0
        dup_ok = cos_orig < 0.98 and max_other < s.duplicate_max_similarity
        out.append(CheckResult(name="not_duplicate", passed=dup_ok, score=round(max(max_other, cos_orig), 3),
                               detail=f"similarity to original {cos_orig:.2f}, to other questions {max_other:.2f}"))
        return out

    def student(self, text: str) -> CheckResult | None:
        if self.client is None:
            return None
        schema = {"type": "object", "properties": {"recall_only": {"type": "boolean"}, "why": {"type": "string"}},
                  "required": ["recall_only", "why"]}
        comp = self.client.chat([{"role": "system", "content": STUDENT_SYSTEM},
                                 {"role": "user", "content": f"Exam question: {text}"}], schema=schema, max_tokens=150)
        data = comp.data if isinstance(comp.data, dict) else {}
        recall = bool(data.get("recall_only", False))
        return CheckResult(name="student", passed=not recall, detail=str(data.get("why", ""))[:300])

    # -- loop -----------------------------------------------------------------------------------------------
    def run(self, job: RewriteJob, others: Sequence[str] = (), context: str | None = None) -> RewriteResult:
        target = int(Level.from_label(job.to_level))
        attempts: list[Attempt] = []
        feedback: list[str] = []
        max_attempts = self.settings.max_attempts if self.client is not None else 1
        for n in range(1, max_attempts + 1):
            t0 = time.perf_counter()
            self._log("Composer", "drafting", t0, question_id=job.question_id, attempt=n, from_level=job.from_level,
                      to_level=job.to_level, feedback=feedback)
            text, composer, rationale = self.compose(job, feedback, context)
            self._log("Composer", "drafted", t0, question_id=job.question_id, attempt=n, text=text)
            if not text:
                feedback = ["the answer was empty"]
                attempts.append(Attempt(number=n, question="", composer=composer, checks=[], passed=False))
                continue
            t0 = time.perf_counter()
            checks = self.critic(text, job.original, others)
            lvl, pred = self.level_check(text, target)
            checks.insert(0, lvl)
            if target >= int(Level.ANALYZE) and all(c.passed for c in checks):
                st = self.student(text)
                if st is not None:
                    checks.append(st)
            passed = all(c.passed for c in checks)
            self._log("Verifier", "checked", t0, question_id=job.question_id, attempt=n, passed=passed,
                      failed=[c.name for c in checks if not c.passed], predicted_level=pred.level,
                      checks=[c.model_dump() for c in checks])
            attempts.append(Attempt(number=n, question=text, composer=composer, rationale=rationale, checks=checks,
                                    passed=passed, predicted_level=pred.level))
            if passed:
                return RewriteResult(job=job, accepted=True, final_question=text, attempts=attempts, needs_teacher=False)
            feedback = [f"{c.name}: {c.detail}" for c in checks if not c.passed]
        return RewriteResult(job=job, accepted=False, final_question=None, attempts=attempts, needs_teacher=True)
