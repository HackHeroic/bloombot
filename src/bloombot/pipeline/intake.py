"""Scribe: turn a question paper (PDF, DOCX, TXT or CSV) into structured questions.

University papers number questions in many ways ("Q.1 a)", "1(a)", "(ii)", "Q2."), put
alternatives between "OR" lines, split into "PART A/B" sections, and print marks, course
outcomes and Bloom tags in brackets or table columns. The parser is a line-based state machine:

* a section header starts a new section
* a main-question line starts a new question; an "OR" line makes the next main question an
  alternative of the previous one (same ``group``)
* a sub-part line starts a sub-question of the current main question; a main question with
  sub-parts becomes their stem and is not assessed itself
* any other line continues the current item
* marks, CO and BT/L/K tags are read from the item and removed from its text
* a paper that numbers its questions "Q1", "Q2" ... is trusted: a bare number at the start of a
  line (a matrix row, a wrapped formula) is then a continuation, not a new question

Scanned PDFs (no text layer) are reported, not guessed; OCR is out of scope.
"""

from __future__ import annotations

import io
import re
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from bloombot.pipeline.schemas import Paper, Question, level_name
from bloombot.taxonomy import parse_label
from bloombot.text.normalize import clean_question, normalize_unicode

_SECTION = re.compile(r"^\s*(?:part|section|unit|module|group)\s*[-–:.]?\s*([a-z0-9]{1,4}|[ivx]{1,4})\b\s*[:.\-–]?\s*(.*)$", re.I)
_OR = re.compile(r"^\s*[\(\[]?\s*or\s*[\)\]]?\s*$", re.I)
# A bare number starts a question only if punctuation or a table-column gap follows it, so a wrapped
# line such as "2 moles of an ideal gas ..." stays a continuation.
_MAIN = re.compile(r"^\s*(?:q(?:uestion)?\s*\.?\s*(?:no\.?)?\s*)?(\d{1,2})(?:\s*[.):\]]\s*|\s{2,}|\t)(?:\(?([a-h])\s*[.)\]])?\s*(.*)$", re.I)
_MAIN_Q = re.compile(r"^\s*q(?:uestion)?\s*\.?\s*(?:no\.?)?\s*(\d{1,2})\s*[.):\]-]?\s*(.*)$", re.I)
_SUB = re.compile(r"^\s*(?:\(?\s*([a-h]|i{1,3}|iv|vi{0,3}|ix|x)\s*[.)\]]\s+|([a-h])(?:\s{2,}|\t))(.*)$", re.I)
_Q_HEAD = re.compile(r"^\s*q(?:uestion)?\s*\.?\s*(?:no\.?)?\s*\d{1,2}\b", re.I)
_INLINE_SUB = re.compile(r"^\s*(\d{1,2})\s*\(\s*([a-h]|i{1,3}|iv|vi{0,3}|ix|x)\s*\)\s*(.*)$", re.I)

_MARKS = [
    re.compile(r"[\(\[]\s*(\d{1,3}(?:\.\d)?)\s*(?:marks?|m)?\s*[\)\]]\s*$", re.I),
    re.compile(r"\b(\d{1,3}(?:\.\d)?)\s*(?:marks?|mks?|m)\s*[\)\]]?\s*$", re.I),
]
_CO = re.compile(r"\bCO\s*-?\s*(\d{1,2})\b", re.I)
_BT = re.compile(r"\b(?:BT|BL|RBT|K|L)\s*-?\s*([1-6])\b")
_NOISE = re.compile(
    r"^\s*(?:time\b|max(?:imum)?\.?\s*marks|duration|roll\s*no|reg(?:ister|\.)?\s*no|instructions?|note\s*:|"
    r"answer\s+(?:all|any)|attempt\s+(?:all|any)|all\s+questions\s+carry|page\s+\d+|\*+$|-+$|_+$|figures?\s+to\s+the\s+right)",
    re.I,
)
_INSTRUCTION_STEM = re.compile(r"^\s*(?:answer|attempt|solve|write)\s+(?:any|all|the following)\b|^\s*(?:any\s+\w+\s+of\s+the\s+following)", re.I)


def _sub_parts(m: re.Match) -> tuple[str, str]:
    bracketed, bare, rest = m.groups()
    return (bracketed or bare), rest


@dataclass
class _Item:
    qid: str
    number: str
    lines: list[str] = field(default_factory=list)
    section: str | None = None
    group: str | None = None
    children: list["_Item"] = field(default_factory=list)

    @property
    def raw(self) -> str:
        return " ".join(s.strip() for s in self.lines if s.strip())


def _extract_meta(raw: str, scheme: str | None) -> tuple[str, float | None, str | None, str | None, bool]:
    """Pull CO, Bloom tag and marks out of an item; return (clean text, marks, co, tagged level,
    whether the marks were only guessed from a bare trailing number)."""
    text = raw
    co = tagged = None
    m = _CO.search(text)
    if m:
        co = f"CO{m.group(1)}"
    m = _BT.search(text)
    if m and scheme:
        lv = parse_label(m.group(1), scheme)  # type: ignore[arg-type]
        tagged = level_name(lv) if lv else None
    # Tags go first so that marks printed before them ("[2] CO1 BT1") end up at the end.
    stripped = _BT.sub(" ", _CO.sub(" ", text)).rstrip(" ,;|/")
    marks, guessed = None, False
    for rx in _MARKS:
        m = rx.search(stripped)
        if m:
            marks = float(m.group(1))
            stripped = stripped[: m.start()]
            break
    if marks is None:
        # a bare trailing number after the sentence is the marks column of a table row
        m = re.search(r"(?<=[.?!)])\s*(\d{1,3})\s*$", stripped)
        if m:
            marks, guessed = float(m.group(1)), True
            stripped = stripped[: m.start()]
    # The printed tag is the label itself: it must not reach the classifier.
    return clean_question(stripped), marks, co, tagged, guessed


def parse_text(text: str, title: str = "", bt_scheme: str | None = "revised") -> Paper:
    text = normalize_unicode(text)
    lines = [ln.rstrip() for ln in text.splitlines()]
    mains: list[_Item] = []
    section: str | None = None
    current_main: _Item | None = None
    current: _Item | None = None
    pending_or = False
    warnings: list[str] = []
    explicit_q = sum(bool(_Q_HEAD.match(ln)) for ln in lines) >= 2

    for ln in lines:
        if not ln.strip() or _NOISE.match(ln):
            continue
        if _OR.match(ln):
            pending_or = True
            continue
        sec = _SECTION.match(ln)
        if sec and len(ln.split()) <= 8 and not re.search(r"[?]", ln):
            section = f"{sec.group(1).upper()}"
            continue
        inline = _INLINE_SUB.match(ln)
        mq = _MAIN_Q.match(ln) or (None if inline or explicit_q else _MAIN.match(ln))
        sub = _SUB.match(ln)
        if inline:  # "11(a) ..." numbering
            num, part, rest = inline.groups()
            if current_main is None or current_main.qid != num:
                item = _Item(qid=num, number=f"Q{num}", section=section)
                if pending_or and mains:  # "11(a) 11(b) OR 12(a) 12(b)": OR between whole questions
                    prev = mains[-1]
                    prev.group = prev.group or f"g{prev.qid}"
                    item.group = prev.group
                    for c in prev.children:
                        c.group = c.group or prev.group
                    pending_or = False
                mains.append(item)
                current_main = item
                grp = item.group
            elif pending_or and current_main.children:  # "11(a) OR 11(b)": OR between parts
                prev = current_main.children[-1]
                prev.group = prev.group or f"g{prev.qid}"
                grp = prev.group
                pending_or = False
            else:
                grp = current_main.group
            child = _Item(qid=f"{num}{part.lower()}", number=f"Q{num} ({part.lower()})", lines=[rest], section=section,
                          group=grp)
            current_main.children.append(child)
            current = child
            continue
        if mq and (not sub or _MAIN_Q.match(ln)):
            groups = mq.groups()
            num, rest = groups[0], groups[-1]
            part = groups[1] if len(groups) == 3 else None
            lead = _SUB.match(rest) if part is None else None
            if lead:  # "Q.1 a) Define ..." or "1   a  Define ..." - the first sub-part shares the line
                part, rest = _sub_parts(lead)
            item = _Item(qid=num, number=f"Q{num}", lines=[], section=section)
            if pending_or and mains:
                prev = mains[-1]
                prev.group = prev.group or f"g{prev.qid}"
                item.group = prev.group
                item.qid = f"{num}-or" if any(m.qid == num for m in mains) else num
                for c in prev.children:
                    c.group = prev.group
            pending_or = False
            mains.append(item)
            current_main = item
            if part:
                child = _Item(qid=f"{item.qid}{part.lower()}", number=f"Q{num} ({part.lower()})", lines=[rest],
                              section=section, group=item.group)
                item.children.append(child)
                current = child
            else:
                item.lines.append(rest)
                current = item
            continue
        if sub and current_main is not None:
            part, rest = _sub_parts(sub)
            if pending_or and current_main.children:
                prev = current_main.children[-1]
                prev.group = prev.group or f"g{prev.qid}"
                grp = prev.group
            else:
                grp = current_main.group
            pending_or = False
            child = _Item(qid=f"{current_main.qid}{part.lower()}", number=f"{current_main.number} ({part.lower()})",
                          lines=[rest], section=section, group=grp)
            current_main.children.append(child)
            current = child
            continue
        if current is None:
            # Preamble line before the first question (title, course code ...).
            if not title and len(ln.split()) <= 12:
                title = ln.strip()
            continue
        current.lines.append(ln)

    questions: list[Question] = []
    guessed: list[Question] = []
    for main in mains:
        stem, stem_marks, stem_co, stem_tag, stem_guess = _extract_meta(main.raw, bt_scheme)
        if main.children:
            for c in main.children:
                t, marks, co, tag, guess = _extract_meta(c.raw, bt_scheme)
                if not t:
                    continue
                context = stem if stem and not _INSTRUCTION_STEM.match(stem) and len(stem.split()) > 3 else None
                # OR between whole questions: the option is the main question; OR between parts: the part
                option = main.qid if main.group and c.group == main.group else (c.qid if c.group else None)
                questions.append(Question(id=c.qid, number=c.number, text=t, context=context, marks=marks,
                                          section=c.section, group=c.group, option=option,
                                          course_outcome=co or stem_co, tagged_level=tag or stem_tag))
                if guess:
                    guessed.append(questions[-1])
        elif stem:
            questions.append(Question(id=main.qid, number=main.number, text=stem, marks=stem_marks, section=main.section,
                                      group=main.group, option=main.qid if main.group else None,
                                      course_outcome=stem_co, tagged_level=stem_tag))
            if stem_guess:
                guessed.append(questions[-1])
    if not questions:
        warnings.append("no questions found; is the file a scanned image or an unusual layout?")
    # A marks column is printed for most questions; a few bare trailing numbers are subscripts or
    # numbers from the question itself ("... F1", a matrix entry), not marks.
    if guessed and sum(q.marks is not None for q in questions) < len(questions) / 2:
        for q in guessed:
            q.marks = None
    seen: dict[str, int] = {}
    for q in questions:  # keep ids unique
        if q.id in seen:
            seen[q.id] += 1
            q.id = f"{q.id}.{seen[q.id]}"
        else:
            seen[q.id] = 0
    missing = sum(q.marks is None for q in questions)
    if questions and 0 < missing < len(questions):
        warnings.append(f"{missing} of {len(questions)} questions have no marks; the report will weight by count")
    total = effective_total(questions) if missing == 0 and questions else None
    return Paper(title=title, questions=questions, total_marks=total, warnings=warnings)


def effective_total(questions: list[Question]) -> float:
    """Marks a student actually answers: for each OR-group, only the largest alternative counts."""
    total, groups = 0.0, {}
    for q in questions:
        if q.group:
            opts = groups.setdefault(q.group, {})
            key = q.option or q.id
            opts[key] = opts.get(key, 0.0) + (q.marks or 0.0)
        else:
            total += q.marks or 0.0
    return total + sum(max(opts.values()) for opts in groups.values())


def _pdf_text(data: bytes) -> str:
    import pdfplumber

    with pdfplumber.open(io.BytesIO(data)) as pdf:
        # x_tolerance below the default 3 keeps words apart in tightly set (e.g. LaTeX) PDFs
        return "\n".join((page.extract_text(x_tolerance=1.5) or "") for page in pdf.pages)


def _docx_text(data: bytes) -> str:
    xml = zipfile.ZipFile(io.BytesIO(data)).read("word/document.xml").decode("utf8")
    paras = []
    for p in re.findall(r"<w:p[ >].*?</w:p>", xml, flags=re.S):
        # table cells become tab-separated pieces of one line
        p = re.sub(r"</w:tc>", "\t", p)
        paras.append(re.sub(r"<[^>]+>", "", p))
    return "\n".join(paras)


def _csv_paper(data: bytes, title: str) -> Paper:
    df = pd.read_csv(io.BytesIO(data))
    cols = {c.lower().strip(): c for c in df.columns}
    qcol = next((cols[c] for c in cols if c in ("question", "questions", "text")), df.columns[0])
    mcol = next((cols[c] for c in cols if c.startswith("mark")), None)
    idcol = next((cols[c] for c in cols if c in ("id", "number", "q.no", "qno", "q no")), None)
    qs = []
    for i, row in df.iterrows():
        text = clean_question(row[qcol])
        if not text:
            continue
        qs.append(Question(id=str(row[idcol]) if idcol else str(i + 1), number=str(row[idcol]) if idcol else f"Q{i + 1}",
                           text=text, marks=float(row[mcol]) if mcol and pd.notna(row[mcol]) else None))
    return Paper(title=title, questions=qs, total_marks=sum(q.marks or 0 for q in qs) or None)


def parse_file(data: bytes, filename: str, bt_scheme: str | None = "revised") -> Paper:
    name = filename.lower()
    title = Path(filename).stem
    if name.endswith(".pdf"):
        text = _pdf_text(data)
        if len(text.strip()) < 20:
            return Paper(title=title, questions=[], warnings=["the PDF has no text layer (scanned?); OCR is not supported"])
        return parse_text(text, title, bt_scheme)
    if name.endswith(".docx"):
        return parse_text(_docx_text(data), title, bt_scheme)
    if name.endswith(".csv"):
        return _csv_paper(data, title)
    return parse_text(data.decode("utf-8", errors="replace"), title, bt_scheme)
