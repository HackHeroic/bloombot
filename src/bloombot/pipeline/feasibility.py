"""Can the paper be done in the time, and do its marks make sense?

The Bloom mix says nothing about length: a paper can meet every bound and still need four hours
of work for a one-hour exam. This module estimates, per question, how long an average student
needs, and checks the marking.

Time model (a transparent rule of thumb, not a measurement):

* reading: the question at 150 words per minute, plus the shared stem of a multi-part question
  read once
* each action the question asks for ("prove", "verify", "compute", "explain" ...) adds a range
  for its kind of answer, below; a question with no recognisable action falls back to its
  predicted Bloom level
* a written answer (explain, analyse, judge, create) also needs at least the time to plan and write
  as many words as its marks call for: about 25 words per mark, at 12-20 words a minute, plus
  17-23% for planning what to say
* the teacher's pace setting scales the result

The ranges follow the per-item times in classroom-testing guides (Nitko 2001; Oosterhof 2001:
short answers 1-4 min, problems with calculations a few minutes, short essays 15-20 min, data
analysis 15-25 min), stretched for written university answers. Words per mark follow common
answer-writing guidance (2 marks ~40-50 words, 3 marks ~60-80, 5 marks ~120-150); there is no
official rule. Undergraduates copy at 26-36 words a minute (Summers and Catarro 2003, as reported in
the South African Journal of Occupational Therapy); composing is slower, hence 12-20. Planning
follows the University of Waikato's exam advice (5-7 minutes of planning per 30 of writing). These are
defaults for a teacher to calibrate, which is why every estimate lists its reasons.

The verbs show what kind of work a question asks for, not how big it is ("construct the LL(1)
table" may take two minutes or twenty). So the checks only rely on the safe direction: a paper is
"too long" when even the low estimate exceeds the time, and a question is under-marked when even its
low estimate is far above the paper's minutes per mark. Over-marking is flagged only for pure recall
("Define X" for 10 marks), whose time has a firm upper bound.
"""

from __future__ import annotations

import re
from statistics import median

from bloombot.pipeline.schemas import MarkingCheck, Pace, Paper, Prediction, QuestionTime, TimeEstimate

READING_WPM = 150
WORDS_PER_MARK = 25
WRITING_WPM = (20, 12)  # fast, slow
PLANNING = (0.17, 0.23)  # share of writing time spent deciding what to say
WRITTEN = {"explain", "analyse", "judge", "create"}
PACE = {"faster": 0.75, "typical": 1.0, "slower": 4 / 3}

# kind of answer -> (minutes low, minutes high)
KINDS = {
    "recall": (1, 3),
    "explain": (3, 8),
    "compute": (3, 8),
    "prove": (6, 15),
    "analyse": (5, 12),
    "judge": (6, 15),
    "create": (15, 30),
}
VERBS = {
    "recall": "define state list name recall identify label mention",
    "explain": "explain describe discuss outline summarize summarise interpret illustrate comment why",
    "compute": "compute calculate recompute find determine solve apply use convert estimate obtain simplify "
               "verify check evaluate draw sketch plot divide multiply trace eliminate reduce minimise minimize "
               "normalise normalize translate generate encode decode perform fill complete",
    "prove": "prove derive show",
    "analyse": "analyse analyze compare contrast differentiate distinguish examine classify",
    "judge": "justify critique assess argue recommend defend",
    "create": "design develop propose construct formulate implement create plan",
}
VERB_KIND = {v: k for k, vs in VERBS.items() for v in vs.split()}
LEVEL_KIND = {"Remember": "recall", "Understand": "explain", "Apply": "compute", "Analyze": "analyse",
              "Evaluate": "judge", "Create": "create"}
MAX_ACTIONS = 6

# A verb counts as an action where a clause starts: at the start, after punctuation, a sub-part
# label such as "(ii)", or a joining word ("... and verify it").
_CLAUSE = r"(?:^|[.;:,?!]\s*|\(\s*(?:[a-h]|i{1,3}|iv|vi{0,3}|ix|x)\s*\)\s*|\b(?:and|then|also|hence|now)\s+)"
_ACTION = re.compile(_CLAUSE + r"(" + "|".join(sorted(VERB_KIND, key=len, reverse=True)) + r")\b", re.I)
_CODE = re.compile(r"\bwrite\s+(?:an?\s+)?(?:\w+\s+)?(?:program|code|function|algorithm|query|script)\b", re.I)
_WRITE = re.compile(_CLAUSE + r"(write)\b", re.I)


def _words(text: str | None) -> int:
    return len((text or "").split())


def actions(text: str, level: str) -> list[str]:
    """Kinds of answer the question asks for, in order."""
    found = []
    for m in _ACTION.finditer(text):
        verb = m.group(1).lower()
        kind = VERB_KIND[verb]
        if verb == "evaluate" and level == "Evaluate":  # "evaluate whether ..." is a judgement
            kind = "judge"
        if verb == "show" and not re.match(r"show\s+that\b", text[m.start(1):], re.I):  # "show the table"
            kind = "compute"
        if verb in ("construct", "plan", "implement") and level != "Create":  # "construct the DFA" is applying
            kind = "compute"
        found.append((m.start(1), verb, kind))
    for m in _WRITE.finditer(text):
        found.append((m.start(1), "write", "create" if _CODE.match(text, m.start(1)) else "explain"))
    return [f"{verb}:{kind}" for _, verb, kind in sorted(found)][:MAX_ACTIONS]


def fmt_minutes(m: float) -> str:
    if m < 60:
        return f"{max(1, round(m))} min"
    m = 5 * round(m / 5)
    h, rest = divmod(int(m), 60)
    return f"{h} h" + (f" {rest} min" if rest else "")


def question_time(qid: str, text: str, level: str, stem_words: int, factor: float,
                  marks: float | None = None) -> QuestionTime:
    acts = actions(text, level)
    basis = "verbs" if acts else "level"
    if not acts:
        acts = [f"{level.lower()}:{LEVEL_KIND[level]}"]
    reading = (_words(text) + stem_words) / READING_WPM
    low = reading + sum(KINDS[a.split(":")[1]][0] for a in acts)
    high = reading + sum(KINDS[a.split(":")[1]][1] for a in acts)
    labels = []
    for a in acts:
        verb, kind = a.split(":")
        lo, hi = KINDS[kind]
        labels.append(f"{verb} ({kind}) {lo * factor:.0f}-{hi * factor:.0f} min")
    if marks and any(a.split(":")[1] in WRITTEN for a in acts):
        words = WORDS_PER_MARK * marks
        w_low = reading + words / WRITING_WPM[0] * (1 + PLANNING[0])
        w_high = reading + words / WRITING_WPM[1] * (1 + PLANNING[1])
        if w_high > high:
            low, high = max(low, w_low), w_high
            labels.append(f"plan and write ~{words:.0f} words for {marks:g} marks "
                          f"{w_low * factor:.0f}-{w_high * factor:.0f} min")
    return QuestionTime(id=qid, low=round(low * factor, 1), high=round(high * factor, 1),
                        reading=round(reading * factor, 1), actions=labels, basis=basis)


def estimate(paper: Paper, predictions: list[Prediction], pace: Pace = "typical") -> TimeEstimate:
    factor = PACE[pace]
    seen_stems: set[str] = set()
    times = []
    for q, p in zip(paper.questions, predictions):
        stem = q.context if q.context and q.context not in seen_stems else None
        if stem:
            seen_stems.add(stem)
        times.append(question_time(q.id, q.text, p.level, _words(stem), factor, q.marks))

    # A student answers one alternative of each OR-group: count the longest.
    low = high = 0.0
    options: dict[str, dict[str, list[float]]] = {}
    for q, t in zip(paper.questions, times):
        if q.group:
            o = options.setdefault(q.group, {}).setdefault(q.option or q.id, [0.0, 0.0])
            o[0] += t.low
            o[1] += t.high
        else:
            low, high = low + t.low, high + t.high
    for opts in options.values():
        lo, hi = max(opts.values(), key=lambda x: x[1])
        low, high = low + lo, high + hi

    d = paper.duration_minutes
    work = f"About {fmt_minutes(low)} to {fmt_minutes(high)} of work"
    if not d:
        verdict, detail = "unknown", f"{work}. Enter the exam duration to check it fits."
    elif low > d:
        verdict, detail = "too_long", f"{work} for a {fmt_minutes(d)} exam: too long even for a fast student."
    elif high > d:
        verdict, detail = "tight", f"{work} for a {fmt_minutes(d)} exam: slower students will not finish."
    else:
        verdict, detail = "fits", f"{work} for a {fmt_minutes(d)} exam."
    return TimeEstimate(questions=times, total_low=round(low, 1), total_high=round(high, 1), duration_minutes=d,
                        pace=pace, verdict=verdict, detail=detail)


def _q(ids: list[str]) -> str:
    shown = ", ".join("Q" + i for i in ids[:6])
    return shown + (f" and {len(ids) - 6} more" if len(ids) > 6 else "")


def marking(paper: Paper, predictions: list[Prediction], timing: TimeEstimate) -> list[MarkingCheck]:
    qs = paper.questions
    missing = [q.id for q in qs if q.marks is None]
    checks = []
    if len(missing) == len(qs):
        checks.append(MarkingCheck(name="Marks printed", ok=None, detail="No marks are printed, so the marking cannot be "
                                   "checked and the Bloom mix counts every question once."))
        return checks
    if missing:
        checks.append(MarkingCheck(name="Marks printed", ok=False, questions=missing,
                                   detail=f"{len(missing)} of {len(qs)} questions have no marks: {_q(missing)}."))
    else:
        checks.append(MarkingCheck(name="Marks printed", ok=True, detail=f"All {len(qs)} questions have marks."))

    from bloombot.pipeline.intake import effective_total

    if paper.max_marks and not missing:
        total = effective_total(qs)
        ok = abs(total - paper.max_marks) < 0.01
        checks.append(MarkingCheck(name="Total marks", ok=ok, detail=(
            f"The questions add up to {total:g}, as printed." if ok else
            f"The questions add up to {total:g}, but the paper says {paper.max_marks:g} marks.")))

    # Marks should follow the work. Compare minutes per mark with the paper's median, in the safe
    # direction only (see the module docstring).
    by_id = {t.id: t for t in timing.questions}
    marked = [q for q in qs if q.marks]
    if len(marked) >= 3:
        mid = median((by_id[q.id].low + by_id[q.id].high) / 2 / q.marks for q in marked)
        heavy = [q for q in marked if by_id[q.id].low / q.marks > 2.5 * mid]
        light = [q for q in marked if by_id[q.id].high / q.marks < mid / 2.5
                 and all("(recall)" in a for a in by_id[q.id].actions)]
        notes = [f"Q{q.id}: {q.marks:g} mark{'s' if q.marks != 1 else ''} for "
                 f"{fmt_minutes(by_id[q.id].low)}-{fmt_minutes(by_id[q.id].high)} of work" for q in (heavy + light)[:5]]
        problems = [f"{len(heavy)} need{'s' if len(heavy) == 1 else ''} far more work than {'its' if len(heavy) == 1 else 'their'} marks"] if heavy else []
        problems += [f"{len(light)} recall question{'s' if len(light) != 1 else ''} carr{'ies' if len(light) == 1 else 'y'} many marks"] if light else []
        checks.append(MarkingCheck(
            name="Marks match the work", ok=not problems, questions=[q.id for q in heavy + light],
            detail=("No question is clearly under- or over-marked for the work it asks for." if not problems else
                    f"{' and '.join(problems)} (the paper averages {mid:.1f} min of work per mark): "
                    + "; ".join(notes) + ".")))

    # OR-alternatives should be interchangeable: same marks and similar level.
    level = {q.id: p.level_id for q, p in zip(qs, predictions)}
    unfair = []
    groups: dict[str, dict[str, list]] = {}
    for q in qs:
        if q.group:
            groups.setdefault(q.group, {}).setdefault(q.option or q.id, []).append(q)
    for opts in groups.values():
        if len(opts) < 2:
            continue
        marks = {o: sum(x.marks or 0 for x in items) for o, items in opts.items()}
        levels = {o: max(level[x.id] for x in items) for o, items in opts.items()}
        names = " or ".join("Q" + o for o in opts)
        if len(set(marks.values())) > 1:
            unfair.append(f"{names} carry {' vs '.join(f'{m:g}' for m in marks.values())} marks")
        elif max(levels.values()) - min(levels.values()) >= 2:
            unfair.append(f"{names} differ by {max(levels.values()) - min(levels.values())} Bloom levels")
    if groups:
        checks.append(MarkingCheck(name="OR alternatives are equivalent", ok=not unfair,
                                   detail="; ".join(unfair) + "." if unfair else
                                   "Each choice carries the same marks at a similar level."))
    return checks


__all__ = ["estimate", "marking", "actions", "fmt_minutes", "KINDS", "PACE", "READING_WPM"]
