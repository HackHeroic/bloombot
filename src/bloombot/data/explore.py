"""Browse the labelled dataset: counts for every source, questions only where the licence allows.

A source's questions are shown only if ``Source.public`` is true. The Indian school-board set is
CC BY-NC and is the held-out test set, so it appears as counts only; showing its questions would
also leak the test set.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import pandas as pd
from pydantic import BaseModel, Field

from bloombot.data.sources import SOURCE_BY_ID
from bloombot.taxonomy import LABELS

LICENCE_URL = {
    "CC BY 4.0": "https://creativecommons.org/licenses/by/4.0/",
    "CC BY-NC 3.0": "https://creativecommons.org/licenses/by-nc/3.0/",
    "MIT": "https://opensource.org/license/mit",
    "Apache 2.0": "https://www.apache.org/licenses/LICENSE-2.0",
}


class SourceInfo(BaseModel):
    id: str
    title: str
    citation: str
    origin: str
    kind: str
    region: str
    licence: str
    licence_url: str | None = None
    role: str
    public: bool = Field(description="questions may be shown; otherwise counts only")
    rows: int
    per_level: dict[str, int]


class DatasetItem(BaseModel):
    uid: str
    source: str
    level: str
    question: str


class DatasetPage(BaseModel):
    total: int = Field(description="matching questions in public sources")
    offset: int
    limit: int
    items: list[DatasetItem]
    by_level: dict[str, int] = Field(description="matching public questions per level (ignores the level filter)")
    by_source: dict[str, int] = Field(description="matching public questions per source (ignores the source filter)")
    sources: list[SourceInfo] = Field(description="every source with its full counts, public or not")


def licence_url(licence: str) -> str | None:
    return next((url for key, url in LICENCE_URL.items() if licence.startswith(key)), None)


@lru_cache(maxsize=1)
def _load(processed: Path) -> tuple[pd.DataFrame, tuple[SourceInfo, ...]]:
    path = processed / "questions.parquet"
    if not path.exists():
        raise FileNotFoundError(f"{path} is missing: run `bloombot data build` first")
    df = pd.read_parquet(path, columns=["uid", "source", "question", "label"])
    # Counts come from the data card when present: a public deployment ships only the public sources'
    # questions, but still reports every source's size.
    card = processed / "datacard.json"
    counts = ({sid: (c["rows"], c["per_level"]) for sid, c in json.loads(card.read_text())["sources"].items()}
              if card.exists() else {})
    for sid, g in df.groupby("source", sort=False):
        counts.setdefault(sid, (len(g), {lab: int((g["label"] == lab).sum()) for lab in LABELS}))
    infos = []
    for sid, (rows, per_level) in counts.items():
        s = SOURCE_BY_ID[sid]
        infos.append(SourceInfo(id=sid, title=s.title, citation=s.citation, origin=s.origin, kind=s.kind, region=s.region,
                                licence=s.licence, licence_url=licence_url(s.licence), role=s.role, public=s.public,
                                rows=rows, per_level={lab: int(per_level.get(lab, 0)) for lab in LABELS}))
    return df, tuple(infos)


def query(processed: Path, source: str | None = None, level: str | None = None, q: str | None = None,
          offset: int = 0, limit: int = 25) -> DatasetPage:
    df, infos = _load(processed)
    public = {i.id for i in infos if i.public}
    pub = df[df["source"].isin(public)]
    if q:
        pub = pub[pub["question"].str.contains(q, case=False, regex=False)]
    by_source = pub[pub["label"] == level] if level else pub
    by_level = pub[pub["source"] == source] if source else pub
    hits = by_level[by_level["label"] == level] if level else by_level
    page = hits.iloc[offset:offset + limit]
    return DatasetPage(
        total=len(hits), offset=offset, limit=limit,
        items=[DatasetItem(uid=r.uid, source=r.source, level=r.label, question=r.question) for r in page.itertuples()],
        by_level={lab: int((by_level["label"] == lab).sum()) for lab in LABELS},
        by_source={s: int((by_source["source"] == s).sum()) for s in sorted(public)},
        sources=list(infos),
    )


def bundle(processed: Path, per_cell: int = 20, seed: int = 0) -> dict:
    """A small, licence-safe sample for the hosted page: up to ``per_cell`` questions per public
    source and level, plus every source's counts and licence."""
    df, infos = _load(processed)
    public = [i.id for i in infos if i.public]
    pub = df[df["source"].isin(public)]
    sample = pub.sample(frac=1, random_state=seed).groupby(["source", "label"]).head(per_cell)
    return {
        "notice": "Questions from public sources only, each under its own licence (see sources[].licence and "
                  "citation); the Indian school-board set (CC BY-NC, test only) is shown as counts only.",
        "sources": [i.model_dump() for i in infos],
        "items": [{"uid": r.uid, "source": r.source, "level": r.label, "question": r.question}
                  for r in sample.sort_values(["source", "label", "uid"]).itertuples()],
    }


__all__ = ["query", "bundle", "DatasetPage", "SourceInfo", "licence_url"]
