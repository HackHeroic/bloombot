"""Human-readable question-paper report (Markdown)."""

from __future__ import annotations

from bloombot.pipeline.feasibility import fmt_minutes
from bloombot.pipeline.schemas import Blueprint, PaperReport, RebalanceResult
from bloombot.taxonomy import LABELS


def _bar(share: float, width: int = 20) -> str:
    n = round(share * width)
    return "█" * n + "·" * (width - n)


def render_markdown(rep: PaperReport, blueprint: Blueprint | None = None, rebalance: RebalanceResult | None = None) -> str:
    lines = [f"# Bloom report: {rep.paper.title or 'question paper'}", ""]
    if rep.paper.warnings:
        lines += [f"> ⚠ {w}" for w in rep.paper.warnings] + [""]
    lines += [f"Weighted by **{rep.weight_by}**; OR-alternatives share one slot.", "", "## Distribution", "",
              "| Level | Share | |", "|---|---:|---|"]
    for lab in LABELS:
        s = rep.distribution[lab]
        lines.append(f"| {lab} | {s:.0%} | `{_bar(s)}` |")
    if blueprint is not None:
        lines += ["", f"## Blueprint: {blueprint.name}", ""]
        if blueprint.description:
            lines += [f"_{blueprint.description.strip()}_", ""]
        for c in rep.checks:
            lines.append(f"- {'✅' if c.ok else '❌'} {c.description} — currently **{c.share:.0%}**")
    if rep.timing is not None or rep.marking:
        lines += ["", "## Time and marks", ""]
        if rep.timing is not None:
            icon = {"fits": "✅", "tight": "⚠️", "too_long": "❌", "unknown": "ℹ️"}[rep.timing.verdict]
            lines.append(f"- {icon} {rep.timing.detail} (pace: {rep.timing.pace})")
        for c in rep.marking:
            lines.append(f"- {'ℹ️' if c.ok is None else '✅' if c.ok else '❌'} **{c.name}.** {c.detail}")
    times = {t.id: t for t in rep.timing.questions} if rep.timing is not None else {}
    lines += ["", "## Questions", "", "| # | Marks | Time | Level | Conf. | Printed tag | Evidence | Question |",
              "|---|---:|---:|---|---:|---|---|---|"]
    for it in rep.items:
        q, p = it.question, it.prediction
        ev = ", ".join(d.verb + (" (negated)" if d.negated else "") for d in p.directives) or (p.frame or "—")
        flag = " ⚑" if p.needs_review else ""
        tag = q.tagged_level or ""
        if tag and tag != p.level:
            tag = f"**{tag}**"
        text = q.text.replace("|", "\\|")
        t = times.get(q.id)
        tm = f"{fmt_minutes(t.low)}–{fmt_minutes(t.high)}" if t else ""
        lines.append(f"| {q.number or q.id} | {q.marks or ''} | {tm} | {p.level}{flag} | {p.confidence:.2f} | {tag} | {ev} | {text} |")
    if rep.needs_review:
        lines += ["", f"⚑ = low confidence, please confirm: {', '.join(rep.needs_review)}"]
    if rebalance is not None:
        lines += ["", "## Proposed rewrites", ""]
        if not rebalance.plan:
            lines.append("No rewrites needed.")
        for r in rebalance.rewrites:
            j = r.job
            lines += [f"### {j.question_id}: {j.from_level} → {j.to_level}", f"_Reason: {j.reason}_", "",
                      f"- Original: {j.original}"]
            if r.accepted:
                lines.append(f"- **Rewrite:** {r.final_question}")
            else:
                lines.append("- ❗ No rewrite passed all checks; please write this one yourself.")
            last = r.attempts[-1] if r.attempts else None
            if last:
                lines.append("- Checks: " + ", ".join(f"{c.name} {'✓' if c.passed else '✗'}" for c in last.checks)
                             + f" ({len(r.attempts)} attempt{'s' if len(r.attempts) > 1 else ''})")
            lines.append("")
        lines += ["## After rewriting", "", "| Level | Before | After |", "|---|---:|---:|"]
        for lab in LABELS:
            lines.append(f"| {lab} | {rebalance.before.distribution[lab]:.0%} | {rebalance.after.distribution[lab]:.0%} |")
        if rebalance.after.checks:
            lines += [""] + [f"- {'✅' if c.ok else '❌'} {c.description} — now **{c.share:.0%}**" for c in rebalance.after.checks]
    return "\n".join(lines) + "\n"
