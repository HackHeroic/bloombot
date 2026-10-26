"""Metrics. Bloom levels are ordered, so besides the usual classification scores we report
ordinal ones: quadratic-weighted kappa (QWK), mean absolute error in levels and the share of
predictions within one level. Weighted F1 is kept for comparison with prior work; macro F1
shows whether rare levels (Evaluate, Create) are handled at all.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
from sklearn.metrics import accuracy_score, cohen_kappa_score, confusion_matrix, f1_score

from bloombot.taxonomy import LABELS, N_LEVELS


def expected_calibration_error(proba: np.ndarray, y: np.ndarray, bins: int = 10) -> float:
    conf = proba.max(axis=1)
    correct = (proba.argmax(axis=1) + 1) == y
    edges = np.linspace(0, 1, bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            ece += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return float(ece)


def scores(y_true: Sequence[int], y_pred: Sequence[int], proba: np.ndarray | None = None) -> dict[str, float]:
    y_true, y_pred = np.asarray(y_true, int), np.asarray(y_pred, int)
    present = sorted(set(y_true.tolist()))
    out = {
        "n": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", labels=present, zero_division=0)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", labels=present, zero_division=0)),
        "within_one": float(np.mean(np.abs(y_true - y_pred) <= 1)),
        "mae": float(np.mean(np.abs(y_true - y_pred))),
        "qwk": float(cohen_kappa_score(y_true, y_pred, weights="quadratic", labels=list(range(1, N_LEVELS + 1)))),
        "lots_hots_accuracy": float(np.mean((y_true <= 3) == (y_pred <= 3))),
    }
    if proba is not None:
        out["ece"] = expected_calibration_error(proba, y_true)
    return out


def per_level_f1(y_true, y_pred) -> dict[str, float]:
    f = f1_score(y_true, y_pred, average=None, labels=list(range(1, N_LEVELS + 1)), zero_division=0)
    return {lab: float(v) for lab, v in zip(LABELS, f)}


def confusion(y_true, y_pred) -> list[list[int]]:
    return confusion_matrix(y_true, y_pred, labels=list(range(1, N_LEVELS + 1))).tolist()


def bootstrap_ci(y_true, y_pred, metric: str = "weighted_f1", n_boot: int = 1000, seed: int = 0,
                 alpha: float = 0.05) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    n = len(y_true)
    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        vals.append(scores(y_true[idx], y_pred[idx])[metric])
    lo, hi = np.quantile(vals, [alpha / 2, 1 - alpha / 2])
    return float(lo), float(hi)


def paired_bootstrap(y_true, pred_a, pred_b, metric: str = "weighted_f1", n_boot: int = 1000, seed: int = 0) -> dict:
    """Is model A better than model B on the same items? Returns mean difference, 95% CI and
    the share of resamples in which A does not beat B (a one-sided p-value)."""
    rng = np.random.default_rng(seed)
    y_true, pred_a, pred_b = map(np.asarray, (y_true, pred_a, pred_b))
    n = len(y_true)
    diffs = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        diffs.append(scores(y_true[idx], pred_a[idx])[metric] - scores(y_true[idx], pred_b[idx])[metric])
    diffs = np.asarray(diffs)
    return {"diff": float(diffs.mean()), "ci": [float(np.quantile(diffs, 0.025)), float(np.quantile(diffs, 0.975))],
            "p_a_not_better": float(np.mean(diffs <= 0))}
