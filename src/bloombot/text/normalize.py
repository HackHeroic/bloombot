"""Surface clean-up of question text before any modelling.

Exam papers wrap the actual question in numbering ("Q3 (b)"), marks ("[10 M]") and
outcome tags ("CO2", "BT4", "L3"). Outcome tags are worse than noise: a "BT4" left in
the text is the label itself, so a model that sees it is cheating. :func:`clean_question`
removes all of these and nothing else; it never changes the wording of the question.
"""

from __future__ import annotations

import re
import unicodedata

_QUOTES = str.maketrans({"‘": "'", "’": "'", "“": '"', "”": '"', "–": "-", "—": "-"})

_TAG = r"(?:CO|PO|PSO|LO|CLO)\s*-?\s*\d{1,2}|(?:BT|BL|RBT|K)\s*-?\s*[1-6]|L\s*-?\s*[1-6]"
_MARKS = r"\d{1,3}(?:\.\d)?\s*(?:marks?|mks?|m)\b"

# A leading enumerator: "Q1.", "Q.1)", "Question 3:", "1.", "1)", "(a)", "a)", "(iv)", "ii.", "1(a)".
_LEADING_ENUM = re.compile(
    r"^\s*(?:(?:q(?:uestion)?\.?\s*no\.?|q(?:uestion)?\.?)\s*\d{1,2}\s*[a-z]?\s*[.):\-]?\s*"
    r"|\d{1,2}\s*\(\s*[a-z]\s*\)\s*"
    r"|\(?\d{1,2}\s*[.)]\s*(?=[^\d\s])"
    r"|\(?\s*(?:[a-h]|i{1,3}|iv|v|vi{1,3}|ix|x)\s*[.)]\s+"
    r"|\(\s*(?:[a-h]|i{1,3}|iv|v|vi{1,3}|ix|x)\s*\)\s*)",
    re.IGNORECASE,
)
# A bracketed group made of tags/marks with at least one tag or mark: "(CO2, BT3)", "[10 Marks]",
# "{K2}". A bare "(2)" mid-sentence is usually an equation reference, so it is left alone.
_BRACKETED_META = re.compile(
    rf"(?<!\w)[\(\[\{{]\s*(?:\d{{1,3}}\s*[,;/|&]\s*)*(?:{_TAG}|{_MARKS})"
    rf"(?:\s*[,;/|&]?\s*(?:{_TAG}|{_MARKS}|\d{{1,3}}))*\s*[\)\]\}}]",
    re.IGNORECASE,
)
# Tags or marks dangling at the end: "... algorithm. CO3 BT2". Bare numbers count as marks only
# straight after sentence-final punctuation ("... algorithm? 10" or "... algorithm. (10)"), so
# "if x^2 = 16" keeps its 16.
_TRAILING_META = re.compile(rf"(?:\s+(?:{_TAG}|{_MARKS}))+\s*$", re.IGNORECASE)
_TRAILING_MARKS = re.compile(r"(?<=[.?!:])\s*[\(\[]?\s*\d{1,3}\s*[\)\]]?\s*$")
_SPACE = re.compile(r"\s+")


_MOJIBAKE = re.compile("[ÂÃâÎÏï][\u0080-¿]")


def fix_mojibake(text: str) -> str:
    """Undo UTF-8 text that was decoded as Latin-1/cp1252 and re-encoded ("Newtonâ\x80\x99s")."""
    if not _MOJIBAKE.search(text):
        return text
    for enc in ("latin-1", "cp1252"):
        try:
            fixed = text.encode(enc).decode("utf-8")
        except (UnicodeEncodeError, UnicodeDecodeError):
            continue
        if not _MOJIBAKE.search(fixed):
            return fixed
    return text


def normalize_unicode(text: str) -> str:
    text = unicodedata.normalize("NFKC", fix_mojibake(text)).translate(_QUOTES)
    return "".join(ch for ch in text if unicodedata.category(ch)[0] != "C" or ch in "\n\t")


def strip_enumerator(text: str) -> str:
    """Remove up to two leading enumerators ("Q2 (a)" -> "")."""
    for _ in range(2):
        new = _LEADING_ENUM.sub("", text, count=1)
        if new == text:
            break
        text = new
    return text


def strip_meta(text: str) -> str:
    """Remove marks and course-outcome / Bloom tags (label leakage)."""
    text = _BRACKETED_META.sub(" ", text)
    prev = None
    while prev != text:
        prev = text
        text = _TRAILING_MARKS.sub("", _TRAILING_META.sub("", text.rstrip()))
    return text


def clean_question(text: object) -> str:
    """Return the bare question: no numbering, marks, CO/BT tags or stray whitespace."""
    if text is None:
        return ""
    s = normalize_unicode(str(text))
    s = _SPACE.sub(" ", s).strip()
    s = strip_enumerator(s)
    s = strip_meta(s)
    s = _SPACE.sub(" ", s).strip(" \t-:;,")
    return s


def dedup_key(text: str) -> str:
    """Key for exact-duplicate detection: case, punctuation and spacing do not matter."""
    s = normalize_unicode(text).lower()
    s = re.sub(r"[^a-z0-9 ]+", " ", s)
    return _SPACE.sub(" ", s).strip()
