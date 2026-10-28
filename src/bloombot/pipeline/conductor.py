"""Conductor: runs the stages in order and keeps the audit trail.

    Scribe -> Lexis + Archivist (+ Socrates if unsure) -> Arbiter -> Architect
           -> for each job: Composer -> Level check -> Critic -> Student (retry with reasons)
           -> report on the rebalanced paper

Stages never call each other; the Conductor passes typed data between them, so each stage can
be tested, replaced or run on its own (the API exposes them separately as well).
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field

from bloombot.config import Settings, get_settings
from bloombot.pipeline import blueprint as bp_mod
from bloombot.pipeline import feasibility
from bloombot.pipeline.assess import Assessor
from bloombot.pipeline.rewrite import Rewriter
from bloombot.pipeline.schemas import (
    Blueprint, Pace, Paper, PaperReport, Prediction, RebalanceResult, TraceEvent, Vote,
)
from bloombot.taxonomy import LABELS, Level


@dataclass
class Conductor:
    assessor: Assessor
    settings: Settings = field(default_factory=get_settings)
    trace: list[TraceEvent] = field(default_factory=list)
    cancel: threading.Event | None = None  # set by the client to stop a running job

    @classmethod
    def load(cls, settings: Settings | None = None, use_llm: bool | None = None) -> "Conductor":
        s = settings or get_settings()
        return cls(Assessor.load(s, use_llm=use_llm), s)

    def _client(self):
        return self.assessor.llm.client if self.assessor.llm is not None else None

    def assess_paper(self, paper: Paper, blueprint: Blueprint | None = None, use_llm: bool | None = None,
                     pace: Pace = "typical") -> PaperReport:
        t0 = time.perf_counter()
        preds = self.assessor.assess([q.text for q in paper.questions], use_llm=use_llm, trace=self.trace,
                                     cancel=self.cancel)
        rep = self.report(paper, preds, blueprint, pace)
        self.trace.append(TraceEvent(agent="Conductor", action="assessed paper", ms=(time.perf_counter() - t0) * 1000,
                                     detail={"questions": len(paper.questions), "compliant": rep.compliant,
                                             "needs_review": len(rep.needs_review), "time": rep.timing.verdict,
                                             "minutes": [rep.timing.total_low, rep.timing.total_high]}))
        return rep

    @staticmethod
    def teacher_levels(paper: Paper, preds: list[Prediction]) -> list[Prediction]:
        """A level the teacher chose replaces the prediction (the votes stay, plus a Teacher vote)."""
        out = []
        for q, p in zip(paper.questions, preds):
            if q.teacher_level:
                lv = LABELS.index(q.teacher_level) + 1
                onehot = {lab: float(lab == q.teacher_level) for lab in LABELS}
                teacher = Vote(agent="Teacher", level=q.teacher_level, confidence=1.0, proba=list(onehot.values()))
                p = p.model_copy(update={"level": q.teacher_level, "level_id": lv, "confidence": 1.0, "proba": onehot,
                                         "needs_review": False, "lower_order": Level(lv).is_lower_order,
                                         "votes": [v for v in p.votes if v.agent != "Teacher"] + [teacher]})
            out.append(p)
        return out

    def report(self, paper: Paper, preds: list[Prediction], blueprint: Blueprint | None = None,
               pace: Pace = "typical") -> PaperReport:
        """Bloom mix, time and marking for already-classified questions (no model is run)."""
        preds = self.teacher_levels(paper, preds)
        rep = bp_mod.report(paper, preds, blueprint)
        rep.timing = feasibility.estimate(paper, preds, pace)
        rep.marking = feasibility.marking(paper, preds, rep.timing)
        verdicts = ([rep.compliant] if rep.compliant is not None else []) + [c.ok for c in rep.marking if c.ok is not None]
        if rep.timing.verdict != "unknown":
            verdicts.append(rep.timing.verdict != "too_long")
        rep.ready = all(verdicts) if verdicts else None
        return rep

    def rebalance(self, paper: Paper, blueprint: Blueprint, use_llm: bool | None = None,
                  pace: Pace = "typical") -> RebalanceResult:
        before = self.assess_paper(paper, blueprint, use_llm, pace)
        t0 = time.perf_counter()
        jobs = bp_mod.plan(before, blueprint)
        self.trace.append(TraceEvent(agent="Architect", action="planned rewrites", ms=(time.perf_counter() - t0) * 1000,
                                     detail={"jobs": len(jobs), "blueprint": blueprint.id}))
        rewriter = Rewriter(self.assessor, self._client(), self.settings, trace=self.trace)
        by_id = {q.id: q for q in paper.questions}
        results = []
        for job in jobs:
            others = [q.text for q in paper.questions if q.id != job.question_id]
            results.append(rewriter.run(job, others, context=by_id[job.question_id].context))

        final = paper.model_copy(deep=True)
        accepted = {r.job.question_id: r.final_question for r in results if r.accepted and r.final_question}
        for q in final.questions:
            if q.id in accepted:
                # a new question: the teacher's level and the old multiple-choice options no longer apply
                q.text, q.teacher_level, q.choices = accepted[q.id], None, []
        after = self.assess_paper(final, blueprint, use_llm, pace)
        return RebalanceResult(before=before, plan=jobs, rewrites=results, after=after, final_paper=final)
