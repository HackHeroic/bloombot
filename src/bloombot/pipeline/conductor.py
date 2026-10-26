"""Conductor: runs the stages in order and keeps the audit trail.

    Scribe -> Lexis + Archivist (+ Socrates if unsure) -> Arbiter -> Architect
           -> for each job: Composer -> Level check -> Critic -> Student (retry with reasons)
           -> report on the rebalanced paper

Stages never call each other; the Conductor passes typed data between them, so each stage can
be tested, replaced or run on its own (the API exposes them separately as well).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from bloombot.config import Settings, get_settings
from bloombot.pipeline import blueprint as bp_mod
from bloombot.pipeline.assess import Assessor
from bloombot.pipeline.rewrite import Rewriter
from bloombot.pipeline.schemas import Blueprint, Paper, PaperReport, RebalanceResult, TraceEvent


@dataclass
class Conductor:
    assessor: Assessor
    settings: Settings = field(default_factory=get_settings)
    trace: list[TraceEvent] = field(default_factory=list)

    @classmethod
    def load(cls, settings: Settings | None = None, use_llm: bool | None = None) -> "Conductor":
        s = settings or get_settings()
        return cls(Assessor.load(s, use_llm=use_llm), s)

    def _client(self):
        return self.assessor.llm.client if self.assessor.llm is not None else None

    def assess_paper(self, paper: Paper, blueprint: Blueprint | None = None, use_llm: bool | None = None) -> PaperReport:
        t0 = time.perf_counter()
        preds = self.assessor.assess([q.text for q in paper.questions], use_llm=use_llm, trace=self.trace)
        rep = bp_mod.report(paper, preds, blueprint)
        self.trace.append(TraceEvent(agent="Conductor", action="assessed paper", ms=(time.perf_counter() - t0) * 1000,
                                     detail={"questions": len(paper.questions), "compliant": rep.compliant,
                                             "needs_review": len(rep.needs_review)}))
        return rep

    def rebalance(self, paper: Paper, blueprint: Blueprint, use_llm: bool | None = None) -> RebalanceResult:
        before = self.assess_paper(paper, blueprint, use_llm)
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
                q.text = accepted[q.id]
        after = self.assess_paper(final, blueprint, use_llm)
        return RebalanceResult(before=before, plan=jobs, rewrites=results, after=after, final_paper=final)
