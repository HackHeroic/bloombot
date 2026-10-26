"""Common interface for every classifier.

Every model maps a list of question strings to an (n, 6) probability matrix whose columns are
the levels Remember..Create in order. Labels are the ints 1-6 of :class:`bloombot.taxonomy.Level`.
Keeping one interface lets the evaluation protocols treat a verb rule, a linear model, a
neural encoder and a prompted LLM identically.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Sequence

import joblib
import numpy as np

from bloombot.taxonomy import N_LEVELS, Level


class BloomClassifier(ABC):
    name: str = "base"
    #: False for models that need no training data (rules, zero-shot LLMs).
    trainable: bool = True

    def fit(self, texts: Sequence[str], y: Sequence[int], sample_weight: np.ndarray | None = None) -> "BloomClassifier":
        return self

    @abstractmethod
    def predict_proba(self, texts: Sequence[str]) -> np.ndarray: ...

    def predict(self, texts: Sequence[str]) -> np.ndarray:
        return self.predict_proba(texts).argmax(axis=1) + 1

    def predict_levels(self, texts: Sequence[str]) -> list[Level]:
        return [Level(int(v)) for v in self.predict(texts)]

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)

    @staticmethod
    def load(path: Path) -> "BloomClassifier":
        return joblib.load(path)

    def __repr__(self) -> str:
        return f"<{type(self).__name__} {self.name}>"


def full_proba(proba: np.ndarray, classes: Sequence[int]) -> np.ndarray:
    """Expand a model's probability columns (only the classes it saw) to all six levels."""
    out = np.zeros((proba.shape[0], N_LEVELS))
    for j, c in enumerate(classes):
        out[:, int(c) - 1] = proba[:, j]
    s = out.sum(axis=1, keepdims=True)
    s[s == 0] = 1.0
    return out / s


def to_index(y: Sequence[int]) -> np.ndarray:
    return np.asarray(y, dtype=int) - 1


def balanced_weights(groups: Sequence, y: Sequence[int] | None = None) -> np.ndarray:
    """Weights that give every group (e.g. source) the same total weight; mean weight 1."""
    groups = np.asarray(groups)
    _, inv, counts = np.unique(groups, return_inverse=True, return_counts=True)
    w = 1.0 / counts[inv]
    return w * len(w) / w.sum()
