"""Sample question papers (data/papers): one list for the API, the CLI and the web UI.

Every file in data/papers is a sample; its id is the file name without "sample_" and "_paper"
(sample_os_paper.txt -> "os"). The description is read from the paper itself: the course, the
marks, the duration and the layout.
"""

from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path

from pydantic import BaseModel

from bloombot.config import ROOT
from bloombot.pipeline.intake import effective_total, parse_file
from bloombot.pipeline.schemas import Paper

SAMPLES_DIR = ROOT / "data" / "papers"
_COURSE = re.compile(r"(?:course\s*:\s*|,\s*)([A-Z][A-Za-z&,' -]+?\s*\(\w*\d\w*\))", re.I)
_TABLE = re.compile(r"^\s*Q\.?\s*No\b", re.I | re.M)


class Sample(BaseModel):
    id: str
    file: str
    subject: str
    description: str
    questions: int
    marks: float | None = None
    duration_minutes: float | None = None
    layout: str


def _id(path: Path) -> str:
    return re.sub(r"^sample_|_paper$", "", path.stem)


def _fmt_duration(m: float | None) -> str:
    if not m:
        return "no time printed"
    h, rest = divmod(int(m), 60)
    return (f"{h} h" if h else "") + (f" {rest} min" if rest else "") if m >= 60 else f"{int(m)} min"


@lru_cache(maxsize=1)
def list_samples() -> list[Sample]:
    out = []
    for path in sorted(SAMPLES_DIR.glob("*")):
        if path.suffix.lower() not in (".txt", ".pdf", ".docx", ".csv") or path.name.startswith("."):
            continue
        paper = parse_file(path.read_bytes(), path.name)
        head = path.read_text(errors="ignore")[:2000] if path.suffix.lower() == ".txt" else ""
        m = _COURSE.search(head)
        subject = m.group(1).strip() if m else (paper.title or path.stem)
        marks = paper.max_marks or (effective_total(paper.questions) if all(q.marks for q in paper.questions) else None)
        layout = "table layout" if _TABLE.search(head) else "numbered questions"
        if any(q.group for q in paper.questions):
            layout += ", OR choices"
        parts = [subject, f"{marks:g} marks" if marks else "no marks", _fmt_duration(paper.duration_minutes),
                 f"{len(paper.questions)} questions", layout]
        out.append(Sample(id=_id(path), file=str(path.relative_to(ROOT)), subject=subject, description=" · ".join(parts),
                          questions=len(paper.questions), marks=marks, duration_minutes=paper.duration_minutes,
                          layout=layout))
    return out


def sample_path(sample_id: str) -> Path | None:
    for path in SAMPLES_DIR.glob("*"):
        if _id(path) == sample_id:
            return path
    return None


def load_sample(sample_id: str) -> Paper | None:
    path = sample_path(sample_id)
    if path is None:
        return None
    paper = parse_file(path.read_bytes(), path.name)
    paper.title = next(s.subject for s in list_samples() if s.id == sample_id)
    return paper


__all__ = ["Sample", "list_samples", "load_sample", "sample_path", "SAMPLES_DIR"]
