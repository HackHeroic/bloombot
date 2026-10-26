"""Registry of public Bloom-labelled question sources.

A *source* is one population of questions with one provenance - not one download. This
matters because cross-dataset evaluation is only meaningful if a held-out source really is
unseen: several public files re-package each other (Yahya's 600 questions appear in four of
them), so each loader keeps only the rows that originate from its own population and the
build step removes anything shared between sources.

Every source records where it comes from, its licence and the label scheme of its raw file.
Files are downloaded from the original publishers; nothing is redistributed here.
"""

from __future__ import annotations

import io
import re
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Literal

import pandas as pd

from bloombot.taxonomy import Level, Scheme, parse_label

Role = Literal["train", "test"]


@dataclass(frozen=True)
class Download:
    name: str  # local file name inside data/external/<source>/
    url: str


@dataclass(frozen=True)
class Source:
    id: str
    title: str
    citation: str
    origin: str  # what the questions are
    kind: Literal["exam", "item_bank", "web", "llm_generated"]
    region: str
    licence: str
    downloads: tuple[Download, ...]
    loader: Callable[[Path], pd.DataFrame]  # -> columns: question, raw_label, level
    role: Role = "train"
    notes: tuple[str, ...] = field(default_factory=tuple)
    # may its questions be shown on a public page (the dataset explorer)? Counts are always shown.
    public: bool = True


# --- helpers ---------------------------------------------------------------------------------
def _read_csv(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path)
    except UnicodeDecodeError:
        return pd.read_csv(path, encoding="cp1252")


def _docx_paragraphs(path: Path) -> list[str]:
    """Non-empty paragraphs of a .docx as plain text (no python-docx dependency)."""
    xml = zipfile.ZipFile(path).read("word/document.xml").decode("utf8")
    paras = (re.sub(r"<[^>]+>", "", p).strip() for p in re.findall(r"<w:p[ >].*?</w:p>", xml, flags=re.S))
    return [p for p in paras if p]


def _labelled(df: pd.DataFrame, qcol: str, lcol: str, scheme: Scheme | None) -> pd.DataFrame:
    out = pd.DataFrame({"question": df[qcol].astype(str), "raw_label": df[lcol].astype(str)})
    out["level"] = [parse_label(v, scheme) for v in df[lcol]]
    return out


def _plos_supplement(path: Path) -> pd.DataFrame:
    """The PLOS supplements list questions as paragraphs under level headings ("Knowledge", ...)."""
    recs, current = [], None
    for para in _docx_paragraphs(path):
        heading = parse_label(para.removesuffix(":").strip(), "original") if len(para.split()) <= 2 else None
        if heading is not None:
            current = (para.removesuffix(":").strip(), heading)
        elif current is not None:
            recs.append((para, current[0], current[1]))
    return pd.DataFrame(recs, columns=["question", "raw_label", "level"])


# --- loaders ---------------------------------------------------------------------------------
def load_yahya(d: Path) -> pd.DataFrame:
    return _plos_supplement(d / "S2_yahya.docx")


def load_mohammed(d: Path) -> pd.DataFrame:
    return _plos_supplement(d / "S1_mohammed.docx")


def load_sangodiah(d: Path) -> pd.DataFrame:
    return _labelled(_read_csv(d / "sangodiah2017.csv"), "QUESTION", "BT LEVEL", "revised")


def load_gani(d: Path) -> pd.DataFrame:
    return _labelled(_read_csv(d / "gani2023.csv"), "QUESTION", "BT LEVEL", "revised")


SCARIA_QUALITY = ("Understandable", "Answerable", "DomainRelated")


def load_scaria(d: Path) -> pd.DataFrame:
    """Two experts rated every generated question. Keep a row only when both gave the same Bloom
    level and neither marked it not understandable, not answerable or off-topic."""
    frames = []
    for i in range(1, 6):
        x = pd.read_excel(d / f"PS{i}_ExpertEvaluation.xlsx")
        x.columns = [str(c).strip() for c in x.columns]
        x["prompt_set"] = i
        frames.append(x)
    x = pd.concat(frames, ignore_index=True)
    s1, s2 = pd.to_numeric(x.SkillLevel_1, errors="coerce"), pd.to_numeric(x.SkillLevel_2, errors="coerce")
    keep = s1.between(1, 6) & (s1 == s2)
    for crit in SCARIA_QUALITY:
        for k in (1, 2):
            col = f"{crit}_{k}"
            if col in x:
                keep &= pd.to_numeric(x[col], errors="coerce").fillna(1) != 0
    x = x[keep]
    out = pd.DataFrame({"question": x["Questions"].astype(str), "raw_label": s1[keep].astype(int).astype(str)})
    out["level"] = [Level(int(v)) for v in out.raw_label]
    return out


def _key(q: object) -> str:
    return " ".join(re.sub(r"[^a-z0-9 ]+", " ", str(q).lower()).split())


def load_indian_engineering(d: Path) -> pd.DataFrame:
    """Devane's Kaggle v5 file mixes five populations. Keep only the Indian engineering exam
    questions: drop rows that are Yahya or Gani questions, and rows found verbatim in two
    LLM-generated Kaggle sets (dineshsheelam, abhaygotmare). v5 uses the REVISED BT order."""
    with zipfile.ZipFile(d / "devane_v5.zip") as z:
        dev = _read_csv_bytes(z.read(_only_csv(z)))
    exclude: set[str] = set()
    with zipfile.ZipFile(d / "dineshsheelam.zip") as z:
        exclude |= set(_read_csv_bytes(z.read(_only_csv(z))).iloc[:, 0].map(_key))
    with zipfile.ZipFile(d / "abhaygotmare.zip") as z:
        ab = _read_csv_bytes(z.read(_only_csv(z)))
        exclude |= set(ab[[c for c in ab.columns if "question" in c.lower()][0]].map(_key))
    # Yahya and Gani rows are removed by the cross-source de-duplication in build.py; the
    # synthetic sets are not sources of their own, so they are excluded here.
    dev = dev[~dev.Questions.map(_key).isin(exclude)]
    return _labelled(dev, "Questions", "Category", "revised")


def _only_csv(z: zipfile.ZipFile) -> str:
    names = [n for n in z.namelist() if n.lower().endswith(".csv")]
    if len(names) != 1:
        raise ValueError(f"expected one CSV in archive, found {names}")
    return names[0]


def _read_csv_bytes(b: bytes) -> pd.DataFrame:
    try:
        return pd.read_csv(io.BytesIO(b))
    except UnicodeDecodeError:
        return pd.read_csv(io.BytesIO(b), encoding="cp1252")


_DAS_NAMES = {"memorization": "Remember", "understanding": "Understand"}
_OPTION_LINE = re.compile(r"^\s*(?:\(?[a-dA-D]\)|[a-dA-D][.)]|\(?[ivx]+\))\s+")


def _das_stem(q: str) -> str:
    """Drop multiple-choice options, keep the stem."""
    lines = [ln for ln in str(q).splitlines() if ln.strip()]
    stem = [ln for i, ln in enumerate(lines) if i == 0 or not _OPTION_LINE.match(ln)]
    return " ".join(stem)


def load_indian_boards(d: Path) -> pd.DataFrame:
    """Das et al.: Indian grade-10 board exams (CBSE, ICSE, NIOS), one sheet per paper. Labels mix
    original and revised names; rows with more than one level are dropped as ambiguous."""
    with zipfile.ZipFile(d / "das_boards.zip") as z:
        name = next(n for n in z.namelist() if n.lower().endswith(".xlsx"))
        book = pd.read_excel(io.BytesIO(z.read(name)), sheet_name=None)
    recs = []
    for sheet, df in book.items():
        lcol = next((c for c in df.columns if "ognitive" in str(c)), None)
        qcol = next((c for c in df.columns if "uestion" in str(c)), None)
        if lcol is None or qcol is None:
            continue
        for q, lab in zip(df[qcol], df[lcol]):
            if pd.isna(q) or pd.isna(lab):
                continue
            lab = str(lab).strip()
            if re.search(r",|\bor\b", lab, flags=re.I):
                continue
            norm = _DAS_NAMES.get(lab.lower(), lab)
            recs.append((_das_stem(q), lab, parse_label(norm), sheet))
    return pd.DataFrame(recs, columns=["question", "raw_label", "level", "paper"])


# --- registry --------------------------------------------------------------------------------
_PLOS = "https://journals.plos.org/plosone/article/file?type=supplementary&id=10.1371/journal.pone.0230442.{}"
_GITLAB = "https://gitlab.com/api/v4/projects/80280833/repository/files/data%2F{}/raw?ref=main"
_SCARIA = "https://raw.githubusercontent.com/nicyscaria/AEQG_Blooms_Evaluation_LLMs/main/generated_questions/PS{}_ExpertEvaluation.xlsx"
_KAGGLE = "https://www.kaggle.com/api/v1/datasets/download/{}"

SOURCES: tuple[Source, ...] = (
    Source(
        id="yahya2012", title="Yahya et al. item bank",
        citation="Yahya, Toukal & Osman (2012), Studies in Computational Intelligence 431, doi:10.1007/978-3-642-30732-4_17; "
                 "file: Mohammed & Omar (2020) PLOS ONE S2",
        origin="600 item-bank questions, 100 per level, many templated", kind="item_bank", region="—",
        licence="CC BY 4.0 (PLOS supplement)",
        downloads=(Download("S2_yahya.docx", _PLOS.format("s002")),), loader=load_yahya,
        notes=("labels use the original taxonomy (Synthesis -> Create, Evaluation -> Evaluate)",),
    ),
    Source(
        id="mohammed2020", title="Mohammed & Omar open-ended set",
        citation="Mohammed & Omar (2020), PLOS ONE 15(3): e0230442, S1",
        origin="141 questions collected from websites, books and earlier work", kind="web", region="—",
        licence="CC BY 4.0 (PLOS supplement)",
        downloads=(Download("S1_mohammed.docx", _PLOS.format("s001")),), loader=load_mohammed,
        notes=("labels use the original taxonomy",),
    ),
    Source(
        id="sangodiah2017", title="Sangodiah et al. university exams",
        citation="Sangodiah et al. (2017), JATIT 95(12); revised labels released by Faraji et al. (2026), arXiv:2606.13684",
        origin="Malaysian university exam questions (computing, business)", kind="exam", region="Malaysia",
        licence="MIT (release repository)",
        downloads=(Download("sangodiah2017.csv", _GITLAB.format("Sangodiah_2017_415_revised.csv")),), loader=load_sangodiah,
    ),
    Source(
        id="gani2023", title="Gani et al. university exams",
        citation="Gani et al. (2023), Education and Information Technologies 28(12); revised labels released by Faraji et al. (2026)",
        origin="Malaysian university exam questions (economics and others)", kind="exam", region="Malaysia",
        licence="MIT (release repository)",
        downloads=(Download("gani2023.csv", _GITLAB.format("Gani_2023_collected-dataset-1200_revised.csv")),), loader=load_gani,
    ),
    Source(
        id="scaria2024", title="Scaria et al. LLM-generated questions",
        citation="Scaria, Chenna & Subramani (2024), AIED 2024, LNAI 14830, doi:10.1007/978-3-031-64299-9_12",
        origin="LLM-generated graduate data-science questions; level agreed by two experts", kind="llm_generated",
        region="India (course context)", licence="MIT",
        downloads=tuple(Download(f"PS{i}_ExpertEvaluation.xlsx", _SCARIA.format(i)) for i in range(1, 6)),
        loader=load_scaria,
        notes=("kept: both experts gave the same level and neither flagged it not understandable/answerable/on-topic",),
    ),
    Source(
        id="indian_engineering", title="Indian engineering exam questions",
        citation="V. Devane, Kaggle dataset vijaydevane/blooms-taxonomy-dataset (v5)",
        origin="Indian university engineering exam questions, BT-coded (mostly BT1-BT4)", kind="exam", region="India",
        licence="Apache 2.0 (Kaggle metadata)",
        downloads=(
            Download("devane_v5.zip", _KAGGLE.format("vijaydevane/blooms-taxonomy-dataset")),
            Download("dineshsheelam.zip", _KAGGLE.format("dineshsheelam/blooms-taxonomy-dataset")),
            Download("abhaygotmare.zip", _KAGGLE.format("abhaygotmare/blooms-taxonomy-questions-level")),
        ),
        loader=load_indian_engineering,
        notes=("v5 uses the revised BT order (earlier versions did not)",
               "rows also present in Yahya, Gani or two synthetic Kaggle sets are removed"),
    ),
    Source(
        id="indian_boards", title="Indian school board exams (test only)",
        citation="Das et al., Mendeley Data doi:10.17632/48nfn95jc8.2",
        origin="Grade-10 CBSE/ICSE/NIOS maths and science papers; options stripped from MCQs", kind="exam",
        region="India", licence="CC BY-NC 3.0 (non-commercial; used for evaluation only)",
        downloads=(Download("das_boards.zip", "https://data.mendeley.com/public-files/datasets/48nfn95jc8/files/"
                                               "f5505123-0ad8-43e2-aff6-1ee1b2493565/file_downloaded"),),
        loader=load_indian_boards, role="test", public=False,  # non-commercial, and the held-out test set
        notes=("very few Evaluate/Create rows; report per-level results",),
    ),
)

SOURCE_BY_ID = {s.id: s for s in SOURCES}
TRAIN_SOURCES = tuple(s.id for s in SOURCES if s.role == "train")
