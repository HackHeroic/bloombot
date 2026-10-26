"""Download, clean and de-duplicate all sources into one table.

Steps, in order (counts for every step go into the data card):

1. download each file into ``data/external/<source>/`` (skipped if present)
2. load with the source's own loader and label scheme; drop rows whose label cannot be read
3. clean the text (:func:`bloombot.text.normalize.clean_question`) and drop garbled rows
   (encoding damage such as "??"), one-word rows and very long rows
4. within a source: collapse exact duplicates; if copies disagree on the label, drop them all
5. across sources: drop exact and near duplicates (character n-gram cosine >= 0.9) from the
   later source in registry order, so each question belongs to exactly one source
6. assign near-duplicate groups (cosine >= 0.8) inside each source, used to keep
   paraphrases in the same fold when splitting
"""

from __future__ import annotations

import hashlib
import json
import logging
import shutil
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse.csgraph import connected_components
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors

from bloombot.data.sources import SOURCES, Source
from bloombot.taxonomy import LABELS, Level
from bloombot.text.normalize import clean_question, dedup_key

log = logging.getLogger(__name__)

NEAR_DUP_CROSS = 0.90
NEAR_DUP_GROUP = 0.80
MIN_WORDS, MAX_WORDS = 2, 200


@dataclass
class Paths:
    root: Path

    @property
    def external(self) -> Path:
        return self.root / "data" / "external"

    @property
    def processed(self) -> Path:
        return self.root / "data" / "processed"

    @property
    def table(self) -> Path:
        return self.processed / "questions.parquet"

    @property
    def card(self) -> Path:
        return self.processed / "datacard.json"


def download(source: Source, dest: Path, seed_dir: Path | None = None, timeout: float = 120.0) -> None:
    import httpx

    dest.mkdir(parents=True, exist_ok=True)
    for dl in source.downloads:
        target = dest / dl.name
        if target.exists() and target.stat().st_size > 0:
            continue
        if seed_dir is not None and (seed_dir / source.id / dl.name).exists():
            shutil.copy(seed_dir / source.id / dl.name, target)
            continue
        log.info("downloading %s <- %s", target, dl.url)
        with httpx.stream("GET", dl.url, follow_redirects=True, timeout=timeout,
                          headers={"User-Agent": "bloombot-data/1.0"}) as r:
            r.raise_for_status()
            tmp = target.with_suffix(target.suffix + ".part")
            with tmp.open("wb") as fh:
                for chunk in r.iter_bytes():
                    fh.write(chunk)
            tmp.rename(target)


def _garbled(text: str) -> bool:
    return "??" in text or "�" in text or sum(ch.isalpha() for ch in text) < 0.5 * len(text)


def _near_dup_pairs(texts: list[str], threshold: float) -> list[tuple[int, int]]:
    if len(texts) < 2:
        return []
    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), lowercase=True, sublinear_tf=True, min_df=1)
    X = vec.fit_transform(texts)
    nn = NearestNeighbors(radius=1 - threshold, metric="cosine", algorithm="brute").fit(X)
    dist, idx = nn.radius_neighbors(X, return_distance=True)
    pairs = []
    for i, (ds, js) in enumerate(zip(dist, idx)):
        for d, j in zip(ds, js):
            if j > i:
                pairs.append((i, int(j)))
    return pairs


def load_source(source: Source, external: Path) -> tuple[pd.DataFrame, dict]:
    raw = source.loader(external / source.id)
    stats = {"raw_rows": int(len(raw))}
    df = raw.copy()
    df = df[df["level"].notna()]
    stats["dropped_unreadable_label"] = stats["raw_rows"] - len(df)
    df["question"] = df.question.map(clean_question)
    df["n_words"] = df.question.str.split().str.len().fillna(0).astype(int)
    bad = df.question.map(_garbled) | (df.n_words < MIN_WORDS) | (df.n_words > MAX_WORDS)
    stats["dropped_garbled_or_length"] = int(bad.sum())
    df = df[~bad].copy()
    df["key"] = df.question.map(dedup_key)
    df["level"] = df["level"].map(int)

    n_levels = df.groupby("key")["level"].nunique()
    conflicted = set(n_levels[n_levels > 1].index)
    stats["dropped_conflicting_duplicates"] = int(df.key.isin(conflicted).sum())
    df = df[~df.key.isin(conflicted)]
    before = len(df)
    df = df.drop_duplicates("key")
    stats["dropped_exact_duplicates"] = before - len(df)
    df.insert(0, "source", source.id)
    return df.reset_index(drop=True), stats


def build(root: Path, seed_dir: Path | None = None, sources: tuple[Source, ...] = SOURCES) -> pd.DataFrame:
    paths = Paths(root)
    frames, card = [], {"sources": {}, "near_dup_cross_threshold": NEAR_DUP_CROSS}
    for src in sources:
        download(src, paths.external / src.id, seed_dir)
        df, stats = load_source(src, paths.external)
        frames.append(df)
        files = {dl.name: hashlib.sha256((paths.external / src.id / dl.name).read_bytes()).hexdigest()
                 for dl in src.downloads}
        card["sources"][src.id] = {"stats": stats, "files_sha256": files, "title": src.title, "citation": src.citation,
                                   "origin": src.origin, "kind": src.kind, "region": src.region,
                                   "licence": src.licence, "role": src.role, "notes": list(src.notes)}
        log.info("%s: %d rows after cleaning", src.id, len(df))

    data = pd.concat(frames, ignore_index=True)
    order = {s.id: i for i, s in enumerate(sources)}
    data["_prio"] = data.source.map(order)
    data = data.sort_values(["_prio"], kind="stable").reset_index(drop=True)

    # Cross-source exact duplicates: keep the copy in the earliest source.
    exact = data.duplicated("key", keep="first")
    for sid, n in data[exact].source.value_counts().items():
        card["sources"][sid]["stats"]["dropped_cross_source_exact"] = int(n)
    data = data[~exact].reset_index(drop=True)

    # Cross-source near duplicates.
    # Pairs are visited in priority order and a row that is already dropped no longer removes
    # others, so in a chain A~B~C (A, C dissimilar) C survives when B is the one dropped.
    drop: set[int] = set()
    for i, j in sorted(_near_dup_pairs(data.question.tolist(), NEAR_DUP_CROSS)):
        if data.source.iat[i] == data.source.iat[j] or i in drop or j in drop:
            continue
        drop.add(j if data._prio.iat[j] >= data._prio.iat[i] else i)
    for sid, n in data.loc[sorted(drop)].source.value_counts().items():
        card["sources"][sid]["stats"]["dropped_cross_source_near"] = int(n)
    data = data.drop(index=sorted(drop)).reset_index(drop=True)

    # Paraphrase groups within each source (for grouped splits).
    data["group"] = -1
    offset = 0
    for sid, idx in data.groupby("source").groups.items():
        idx = list(idx)
        pairs = _near_dup_pairs(data.loc[idx, "question"].tolist(), NEAR_DUP_GROUP)
        n = len(idx)
        if pairs:
            from scipy.sparse import coo_matrix

            a, b = zip(*pairs)
            g = coo_matrix((np.ones(len(a)), (a, b)), shape=(n, n))
            _, labels = connected_components(g, directed=False)
        else:
            labels = np.arange(n)
        data.loc[idx, "group"] = labels + offset
        offset += int(labels.max()) + 1

    data["label"] = data["level"].map(lambda v: Level(v).label)
    data["uid"] = [f"{s}:{i:05d}" for s, i in zip(data.source, data.groupby("source").cumcount())]
    data = data[["uid", "source", "question", "level", "label", "raw_label", "n_words", "group"]]

    for sid in card["sources"]:
        sub = data[data.source == sid]
        card["sources"][sid]["rows"] = int(len(sub))
        card["sources"][sid]["per_level"] = {lab: int((sub.label == lab).sum()) for lab in LABELS}
        card["sources"][sid]["paraphrase_groups"] = int(sub.group.nunique())
    card["total_rows"] = int(len(data))

    paths.processed.mkdir(parents=True, exist_ok=True)
    data.to_parquet(paths.table, index=False)
    paths.card.write_text(json.dumps(card, indent=2))
    return data


def load_questions(root: Path) -> pd.DataFrame:
    path = Paths(root).table
    if not path.exists():
        raise FileNotFoundError(f"{path} not found; run `bloombot data build` first")
    return pd.read_parquet(path)
