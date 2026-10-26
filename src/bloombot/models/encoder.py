"""Sentence-embedding models (CPU or Apple MPS; no CUDA needed).

* :class:`EmbedLR` - frozen sentence encoder + logistic regression. With ~10k short
  questions, encoding takes about a minute on a laptop CPU and training is instant.
* :class:`EmbedLex` - the embedding concatenated with BloomLex's linguistic features, so the
  dense semantic view and the explicit directive/frame view complement each other.

Embeddings are cached per text, so protocols that retrain many times encode each question once.
"""

from __future__ import annotations

import hashlib
from functools import lru_cache
from typing import Sequence

import numpy as np
from scipy import sparse
from sklearn.linear_model import LogisticRegression

from bloombot.models.base import BloomClassifier, full_proba
from bloombot.models.linear import BloomLex
from bloombot.text.linguistics import default_analyzer

DEFAULT_ENCODER = "BAAI/bge-small-en-v1.5"


class _HashingEncoder:
    """Dependency-free stand-in: L2-normalised hashed character n-grams. Used when
    ``encoder="hashing"`` (tests, or machines that cannot download a model)."""

    def __init__(self, dim: int = 512):
        from sklearn.feature_extraction.text import HashingVectorizer

        self.vec = HashingVectorizer(analyzer="char_wb", ngram_range=(3, 4), n_features=dim, alternate_sign=False,
                                     norm="l2", lowercase=True)

    def encode(self, texts, **_):
        return self.vec.transform(texts).toarray().astype(np.float32)


@lru_cache(maxsize=4)
def _load(name: str):
    if name == "hashing":
        return _HashingEncoder()
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(name, device="cpu")


class EmbeddingCache:
    def __init__(self, model_name: str = DEFAULT_ENCODER, batch_size: int = 64):
        self.model_name, self.batch_size = model_name, batch_size
        self._store: dict[str, np.ndarray] = {}

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        keys = [hashlib.sha1(t.encode()).hexdigest() for t in texts]
        todo = [t for t, k in dict(zip(texts, keys)).items() if k not in self._store]
        todo = list(dict.fromkeys(todo))
        if todo:
            vecs = _load(self.model_name).encode(todo, batch_size=self.batch_size, normalize_embeddings=True,
                                                 show_progress_bar=False, convert_to_numpy=True)
            for t, v in zip(todo, vecs):
                self._store[hashlib.sha1(t.encode()).hexdigest()] = v.astype(np.float32)
        return np.stack([self._store[k] for k in keys])


_CACHES: dict[str, EmbeddingCache] = {}


def embedding_cache(model_name: str = DEFAULT_ENCODER) -> EmbeddingCache:
    if model_name not in _CACHES:
        _CACHES[model_name] = EmbeddingCache(model_name)
    return _CACHES[model_name]


class EmbedLR(BloomClassifier):
    name = "embed_lr"

    def __init__(self, encoder: str = DEFAULT_ENCODER, C: float = 2.0, masked: bool = False):
        self.encoder, self.C, self.masked = encoder, C, masked

    def _X(self, texts):
        if self.masked:
            texts = [a.masked for a in default_analyzer().batch(texts)]
        else:
            texts = [a.text for a in default_analyzer().batch(texts)]  # same cleaning as other models
        return embedding_cache(self.encoder).encode(texts)

    def fit(self, texts, y, sample_weight=None):
        self.clf_ = LogisticRegression(C=self.C, max_iter=3000).fit(self._X(texts), np.asarray(y), sample_weight=sample_weight)
        return self

    def predict_proba(self, texts):
        return full_proba(self.clf_.predict_proba(self._X(texts)), self.clf_.classes_)

    def __getstate__(self):
        return self.__dict__  # the encoder itself is loaded lazily, never pickled


class EmbedLex(BloomLex):
    """BloomLex features + sentence embedding (scaled) in one logistic regression."""

    name = "embedlex"

    def __init__(self, encoder: str = DEFAULT_ENCODER, embed_weight: float = 1.0, **kw):
        super().__init__(**kw)
        self.encoder, self.embed_weight = encoder, embed_weight

    def _matrix(self, analyses, fit):
        X = super()._matrix(analyses, fit)
        E = embedding_cache(self.encoder).encode([a.text for a in analyses]) * self.embed_weight
        return sparse.hstack([X, sparse.csr_matrix(E)]).tocsr()
