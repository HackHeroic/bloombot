"""Architect: measure a paper against a blueprint and plan the fewest rewrites that fix it.

Weights: each question weighs its marks (or 1 if the paper has no marks). Alternatives joined
by OR share one slot: an item in a group with n alternatives (a single part, or a whole
multi-part question) counts with weight / n - the expected weight a student meets.

Planning is greedy. While a bound is violated, pick the question whose move closes the
largest part of the gap with the smallest jump in level, preferring confidently-labelled
questions (an uncertain label should go to the teacher, not to a rewrite) and never touching
locked ones. The target level is the nearest level that satisfies the bound without breaking
another one. Greedy is not optimal in general, but papers have tens of questions and the plan
is a proposal the teacher edits.
"""

from __future__ import annotations

from pathlib import Path

import yaml

from bloombot.pipeline.schemas import (
    Blueprint, BoundCheck, LevelBound, Paper, PaperReport, Prediction, QuestionAssessment, RewriteJob, level_name,
)
from bloombot.taxonomy import LABELS, Level

BLUEPRINT_DIR = Path(__file__).resolve().parents[3] / "configs" / "blueprints"


def load_blueprints(directory: Path = BLUEPRINT_DIR) -> dict[str, Blueprint]:
    out = {}
    for path in sorted(directory.glob("*.yaml")):
        bp = Blueprint.model_validate(yaml.safe_load(path.read_text()))
        out[bp.id] = bp
    return out


def question_weights(paper: Paper, weight_by: str = "marks") -> list[float]:
    has_marks = all(q.marks for q in paper.questions)
    use_marks = weight_by == "marks" and has_marks
    # number of alternatives (options) per OR-group; each alternative is met with probability 1/n
    options: dict[str, set] = {}
    for q in paper.questions:
        if q.group:
            options.setdefault(q.group, set()).add(q.option or q.id)
    out = []
    for q in paper.questions:
        w = float(q.marks) if use_marks else 1.0
        if q.group:
            w /= len(options[q.group])
        out.append(w)
    return out


def distribution(levels: list[int], weights: list[float]) -> dict[str, float]:
    total = sum(weights) or 1.0
    d = {lab: 0.0 for lab in LABELS}
    for lv, w in zip(levels, weights):
        d[LABELS[lv - 1]] += w / total
    return d


def check(dist: dict[str, float], blueprint: Blueprint | None) -> list[BoundCheck]:
    if blueprint is None:
        return []
    out = []
    for b in blueprint.bounds:
        share = sum(dist[lv] for lv in b.levels)
        gap = 0.0
        if b.max_share is not None and share > b.max_share + 1e-9:
            gap = b.max_share - share
        elif b.min_share is not None and share < b.min_share - 1e-9:
            gap = b.min_share - share
        out.append(BoundCheck(bound=b, description=b.describe(), share=round(share, 4), ok=gap == 0.0, gap=round(gap, 4)))
    return out


def report(paper: Paper, predictions: list[Prediction], blueprint: Blueprint | None = None) -> PaperReport:
    weight_by = blueprint.weight_by if blueprint else "marks"
    weights = question_weights(paper, weight_by)
    levels = [p.level_id for p in predictions]
    dist = distribution(levels, weights)
    checks = check(dist, blueprint)
    used = "marks" if weight_by == "marks" and all(q.marks for q in paper.questions) else "count"
    return PaperReport(
        paper=paper,
        items=[QuestionAssessment(question=q, prediction=p) for q, p in zip(paper.questions, predictions)],
        distribution={k: round(v, 4) for k, v in dist.items()},
        weight_by=used,
        checks=checks,
        compliant=all(c.ok for c in checks) if checks else None,
        needs_review=[q.id for q, p in zip(paper.questions, predictions) if p.needs_review],
    )


def _violations(levels: list[int], weights: list[float], bp: Blueprint) -> list[BoundCheck]:
    return [c for c in check(distribution(levels, weights), bp) if not c.ok]


def plan(rep: PaperReport, blueprint: Blueprint, max_jobs: int = 20) -> list[RewriteJob]:
    paper = rep.paper
    weights = question_weights(paper, blueprint.weight_by)
    levels = [it.prediction.level_id for it in rep.items]
    conf = [it.prediction.confidence for it in rep.items]
    planned: dict[int, int] = {}
    jobs: list[RewriteJob] = []

    def total_violation(lvls):
        return sum(abs(c.gap) for c in _violations(lvls, weights, blueprint))

    for _ in range(max_jobs):
        viol = _violations(levels, weights, blueprint)
        if not viol:
            break
        worst = max(viol, key=lambda c: abs(c.gap))
        inside = {int(Level.from_label(lv)) for lv in worst.bound.levels}
        moving_out = worst.gap < 0
        best = None
        for i, q in enumerate(paper.questions):
            if q.locked or i in planned:
                continue
            if moving_out != (levels[i] in inside):
                continue
            targets = [t for t in range(1, 7) if (t not in inside) == moving_out and t != levels[i]]
            for t in sorted(targets, key=lambda t: (abs(t - levels[i]), -t)):
                trial = list(levels)
                trial[i] = t
                gain = total_violation(levels) - total_violation(trial)
                if gain <= 1e-9:
                    continue
                # Higher gain first; then smaller level jumps; then more confident current labels.
                score = (round(gain, 6), -abs(t - levels[i]), conf[i])
                if best is None or score > best[0]:
                    best = (score, i, t)
                break  # nearest useful target for this question
        if best is None:
            break
        _, i, t = best
        planned[i] = t
        q = paper.questions[i]
        jobs.append(RewriteJob(question_id=q.id, original=q.text, from_level=level_name(levels[i]),
                               to_level=level_name(t), marks=q.marks,
                               reason=f"{worst.description}: currently {worst.share:.0%}"))
        levels[i] = t
    return jobs
