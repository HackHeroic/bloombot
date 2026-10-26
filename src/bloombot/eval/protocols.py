"""Evaluation protocols.

* ``indomain``  grouped, stratified 5-fold CV inside one source - the optimistic number most
                papers report. Paraphrase groups never straddle folds.
* ``transfer``  train on one whole source, test on every other source (Faraji et al., 2026).
                The *drop* of a source is in-domain score minus its mean transfer score.
* ``loso``      leave-one-source-out: train on all other training sources, test on the held-out
                one. This is the realistic setting for a tool used on a new college's papers.
* ``heldout``   train on all training sources, test on the test-only sources (Indian boards).

Multi-source training gives each source equal total weight by default, so the largest
source (Indian engineering, ~3.5k) does not dominate.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from bloombot.eval.metrics import per_level_f1, scores
from bloombot.models.base import BloomClassifier, balanced_weights

Factory = Callable[[], BloomClassifier]


@dataclass
class Result:
    protocol: str
    model: str
    train: str
    test: str
    metrics: dict
    per_level: dict

    def row(self) -> dict:
        return {"protocol": self.protocol, "model": self.model, "train": self.train, "test": self.test,
                **self.metrics, **{f"f1_{k}": v for k, v in self.per_level.items()}}


def _pred_frame(test: pd.DataFrame, proba: np.ndarray, protocol: str, model: str, train: str) -> pd.DataFrame:
    out = test[["uid", "source", "question", "level"]].copy()
    out["pred"] = proba.argmax(axis=1) + 1
    out["confidence"] = proba.max(axis=1)
    for k in range(proba.shape[1]):
        out[f"p{k + 1}"] = proba[:, k]
    out["protocol"], out["model"], out["train"] = protocol, model, train
    return out


def _evaluate(model: BloomClassifier, train: pd.DataFrame | None, test: pd.DataFrame, weights=None) -> np.ndarray:
    if train is not None and model.trainable:
        model.fit(train.question.tolist(), train.level.to_numpy(), sample_weight=weights)
    return model.predict_proba(test.question.tolist())


def _result(protocol, name, train_name, test, proba) -> Result:
    y, pred = test.level.to_numpy(), proba.argmax(axis=1) + 1
    return Result(protocol, name, train_name, str(test.source.iat[0]) if test.source.nunique() == 1 else "mixed",
                  scores(y, pred, proba), per_level_f1(y, pred))


def indomain(df: pd.DataFrame, factory: Factory, sources: Iterable[str], folds: int = 5, seed: int = 0):
    results, preds = [], []
    for src in sources:
        d = df[df.source == src].reset_index(drop=True)
        proba = np.zeros((len(d), 6))
        cv = StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=seed)
        name, fallback = None, False
        for tr, te in cv.split(d, d.level, groups=d.group):
            m = factory()
            name = m.name
            proba[te] = _evaluate(m, d.iloc[tr], d.iloc[te])
            fallback |= getattr(m, "fallback_", False)
        results.append(_result("indomain", name, src, d, proba))
        results[-1].metrics["calibration_fallback"] = bool(fallback)
        preds.append(_pred_frame(d, proba, "indomain", name, src))
    return results, pd.concat(preds, ignore_index=True)


def transfer(df: pd.DataFrame, factory: Factory, sources: list[str]):
    results, preds = [], []
    for a in sources:
        m = factory()
        train = df[df.source == a]
        if m.trainable:
            m.fit(train.question.tolist(), train.level.to_numpy())
        for b in sources:
            if a == b:
                continue
            test = df[df.source == b]
            proba = m.predict_proba(test.question.tolist())
            results.append(_result("transfer", m.name, a, test, proba))
            results[-1].metrics["calibration_fallback"] = bool(getattr(m, "fallback_", False))
            preds.append(_pred_frame(test, proba, "transfer", m.name, a))
        if not m.trainable:  # one pass is enough for a model that does not learn
            pass
    return results, pd.concat(preds, ignore_index=True)


def loso(df: pd.DataFrame, factory: Factory, train_sources: list[str], balance: bool = True):
    results, preds = [], []
    for held in train_sources:
        train = df[df.source.isin([s for s in train_sources if s != held])]
        test = df[df.source == held]
        w = balanced_weights(train.source) if balance else None
        m = factory()
        proba = _evaluate(m, train, test, w)
        results.append(_result("loso", m.name, f"all-but-{held}", test, proba))
        preds.append(_pred_frame(test, proba, "loso", m.name, f"all-but-{held}"))
    return results, pd.concat(preds, ignore_index=True)


def heldout(df: pd.DataFrame, factory: Factory, train_sources: list[str], test_sources: list[str], balance: bool = True):
    train = df[df.source.isin(train_sources)]
    m = factory()
    if m.trainable:
        m.fit(train.question.tolist(), train.level.to_numpy(), sample_weight=balanced_weights(train.source) if balance else None)
    results, preds = [], []
    for t in test_sources:
        test = df[df.source == t]
        proba = m.predict_proba(test.question.tolist())
        results.append(_result("heldout", m.name, "all-train", test, proba))
        preds.append(_pred_frame(test, proba, "heldout", m.name, "all-train"))
    return results, pd.concat(preds, ignore_index=True), m


def adapt(df: pd.DataFrame, factory: Factory, train_sources: list[str], targets: list[str],
          ks: tuple[int, ...] = (0, 10, 25, 50, 100), seeds: tuple[int, ...] = (0, 1, 2),
          target_weight: float = 10.0, test_frac: float = 0.5):
    """k-shot convention adaptation. Institutions label the same verb differently (concept
    shift), so a global model cannot be right everywhere. Here a teacher labels k questions
    from the new source; they are added to the training data with a high weight and the model
    is retrained. The test half of the target source is fixed per seed so k values compare."""
    rows, preds = [], []
    for tgt in targets:
        others = df[df.source.isin([s for s in train_sources if s != tgt])]
        base_w = balanced_weights(others.source)
        target = df[df.source == tgt]
        for seed in seeds:
            rng = np.random.default_rng(seed)
            groups = target.group.unique()
            rng.shuffle(groups)
            test_groups = set(groups[: int(len(groups) * test_frac)])
            test = target[target.group.isin(test_groups)]
            pool = target[~target.group.isin(test_groups)].sample(frac=1.0, random_state=seed)
            for k in ks:
                if k > len(pool):
                    continue
                shots = pool.iloc[:k]
                train = pd.concat([others, shots])
                w = np.concatenate([base_w, np.full(len(shots), target_weight)])
                m = factory()
                proba = _evaluate(m, train, test, w)
                r = _result("adapt", m.name, f"all-but-{tgt}+{k}", test, proba)
                rows.append({**r.row(), "k": k, "seed": seed, "target": tgt})
                pf = _pred_frame(test, proba, "adapt", m.name, f"all-but-{tgt}+{k}")
                pf["k"], pf["seed"] = k, seed
                preds.append(pf)
    return pd.DataFrame(rows), pd.concat(preds, ignore_index=True)


def summarize(results: list[Result]) -> pd.DataFrame:
    return pd.DataFrame([r.row() for r in results])


def drop_table(indomain_rows: pd.DataFrame, transfer_rows: pd.DataFrame, metric: str = "weighted_f1") -> pd.DataFrame:
    """Per model and source: in-domain score, mean score when that source's model is applied
    to the other sources, and the drop between them."""
    ind = indomain_rows.set_index(["model", "train"])[metric]
    tr = transfer_rows.groupby(["model", "train"])[metric].mean()
    out = pd.DataFrame({"indomain": ind, "transfer_mean": tr}).dropna()
    out["drop"] = out.indomain - out.transfer_mean
    return out.reset_index().rename(columns={"train": "source"})
