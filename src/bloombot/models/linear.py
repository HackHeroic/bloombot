"""CPU models: lexicon rule, TF-IDF + SVM, POS-weighted TF-IDF + SVM, and BloomLex.

* :class:`LexiconRule` - argmax of the directive/frame prior from the parser. No training.
* :class:`TfidfSVM` - word 1-2 grams + character 2-5 grams, linear SVM with Platt scaling.
  The standard strong baseline in the Bloom literature.
* :class:`PosTfidfSVM` - TF-IDF with term frequency weighted by part of speech (verbs count
  most), after the TFPOS-IDF idea of Mohammed & Omar (2020), the baseline Faraji et al. (2026)
  use for cross-dataset tests.
* :class:`BloomLex` - our model. Instead of raw words it sees what the parser extracted:
  directive verbs and their objects, question frame, embedded (trap) verbs, negation, the
  lexicon prior, and the *delexicalised* question with topic words masked. The hypothesis is
  that these features describe the cognitive task and therefore transfer across sources,
  while topic words do not. Optionally ordinal (Frank & Hall, 2001).
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
from scipy import sparse
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion
from sklearn.svm import LinearSVC

from bloombot.models.base import BloomClassifier, full_proba
from bloombot.taxonomy import N_LEVELS
from bloombot.text.linguistics import Analysis, default_analyzer


class LexiconRule(BloomClassifier):
    name = "lexicon_rule"
    trainable = False

    def predict_proba(self, texts: Sequence[str]) -> np.ndarray:
        return np.array([a.prior for a in default_analyzer().batch(texts)])


def _fit_calibrated_svm(X, y, w, C: float, seed: int):
    """Linear SVM with Platt scaling. Calibration uses up to 3 folds; with a class of only 2
    examples it uses 2 folds. A class with a single example cannot be calibrated at all: then a
    logistic regression is used and ``fallback`` is True (reported by the caller, never silent)."""
    _, counts = np.unique(y, return_counts=True)
    folds = int(min(3, counts.min()))
    if folds < 2:
        return LogisticRegression(max_iter=2000, C=1.0).fit(X, y, sample_weight=w), True
    clf = CalibratedClassifierCV(LinearSVC(C=C, random_state=seed), method="sigmoid", cv=folds)
    return clf.fit(X, y, sample_weight=w), False


class TfidfSVM(BloomClassifier):
    name = "tfidf_svm"

    def __init__(self, C: float = 0.5, seed: int = 0):
        self.C, self.seed = C, seed

    def fit(self, texts, y, sample_weight=None):
        self.vec_ = FeatureUnion([
            ("word", TfidfVectorizer(ngram_range=(1, 2), min_df=1, sublinear_tf=True, lowercase=True)),
            ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=2, sublinear_tf=True, lowercase=True)),
        ])
        X = self.vec_.fit_transform(list(texts))
        self.clf_, self.fallback_ = _fit_calibrated_svm(X, np.asarray(y), sample_weight, self.C, self.seed)
        return self

    def predict_proba(self, texts):
        X = self.vec_.transform(list(texts))
        return full_proba(self.clf_.predict_proba(X), self.clf_.classes_)


POS_WEIGHTS = {"VERB": 3, "AUX": 1, "NOUN": 2, "PROPN": 2, "ADJ": 2, "ADV": 1}


def _pos_weighted_tokens(a: Analysis) -> list[str]:
    out = []
    for word, pos in a.tokens:
        if not any(ch.isalnum() for ch in word):
            continue
        out.extend([word] * POS_WEIGHTS.get(pos, 1))
    return out


class PosTfidfSVM(BloomClassifier):
    name = "pos_tfidf_svm"

    def __init__(self, C: float = 0.5, seed: int = 0):
        self.C, self.seed = C, seed

    def _docs(self, texts):
        return [_pos_weighted_tokens(a) for a in default_analyzer().batch(texts)]

    def fit(self, texts, y, sample_weight=None):
        self.vec_ = TfidfVectorizer(analyzer=lambda toks: toks, sublinear_tf=False, min_df=1)
        X = self.vec_.fit_transform(self._docs(texts))
        self.clf_, self.fallback_ = _fit_calibrated_svm(X, np.asarray(y), sample_weight, self.C, self.seed)
        return self

    def predict_proba(self, texts):
        X = self.vec_.transform(self._docs(texts))
        return full_proba(self.clf_.predict_proba(X), self.clf_.classes_)


# --- BloomLex ----------------------------------------------------------------------------------
def _len_bin(n: int) -> str:
    return "1-5" if n <= 5 else "6-10" if n <= 10 else "11-20" if n <= 20 else "21-40" if n <= 40 else "40+"


def structural_features(a: Analysis) -> dict[str, float]:
    f: dict[str, float] = {}
    active = [d for d in a.directives if not d.negated]
    for i, d in enumerate(active):
        f[f"verb={d.lemma}"] = 1.0
        if i == 0:
            f[f"verb0={d.lemma}"] = 1.0
        if d.phrase:
            f[f"phrase={d.phrase}"] = 1.0
        if d.obj:
            f[f"vobj={d.lemma}:{d.obj}"] = 1.0
        f[f"kind={d.kind}"] = 1.0
    for d in a.directives:
        if d.negated:
            f[f"neg_verb={d.lemma}"] = 1.0
    for v in a.embedded_bloom_verbs:
        f[f"emb={v}"] = 1.0
    f[f"n_directives={min(len(active), 3)}"] = 1.0
    f[f"evidence={a.evidence}"] = 1.0
    if a.frame:
        f[f"frame={a.frame}"] = 1.0
    if a.wh_word:
        f[f"wh={a.wh_word}"] = 1.0
    if a.is_question:
        f["is_question"] = 1.0
    f[f"len={_len_bin(a.n_tokens)}"] = 1.0
    f[f"sents={min(a.n_sentences, 3)}"] = 1.0
    return f


class _Ordinal:
    """Frank & Hall (2001): K-1 binary models for P(y > k), turned into class probabilities."""

    def __init__(self, C: float, class_weight):
        self.C, self.class_weight = C, class_weight

    def fit(self, X, y, sample_weight=None):
        self.models_ = []
        for k in range(1, N_LEVELS):
            t = (y > k).astype(int)
            if t.min() == t.max():
                self.models_.append(float(t[0]))
                continue
            m = LogisticRegression(C=self.C, max_iter=3000, class_weight=self.class_weight)
            m.fit(X, t, sample_weight=sample_weight)
            self.models_.append(m)
        self.classes_ = np.arange(1, N_LEVELS + 1)
        return self

    def predict_proba(self, X):
        n = X.shape[0]
        gt = np.column_stack([np.full(n, m) if isinstance(m, float) else m.predict_proba(X)[:, 1] for m in self.models_])
        gt = np.minimum.accumulate(gt, axis=1)  # P(y>k) must not increase with k
        cdf_gt = np.column_stack([np.ones(n), gt, np.zeros(n)])
        p = cdf_gt[:, :-1] - cdf_gt[:, 1:]
        p = np.clip(p, 1e-9, None)
        return p / p.sum(axis=1, keepdims=True)


class BloomLex(BloomClassifier):
    """Linguistic features + delexicalised n-grams + lexicon prior -> logistic regression."""

    name = "bloomlex"

    def __init__(self, C: float = 1.0, delex: bool = True, raw_text: bool = False, prior_weight: float = 2.0,
                 ordinal: bool = False, class_weight: str | None = None):
        self.C, self.delex, self.raw_text = C, delex, raw_text
        self.prior_weight, self.ordinal, self.class_weight = prior_weight, ordinal, class_weight

    def _matrix(self, analyses: list[Analysis], fit: bool):
        feats = [structural_features(a) for a in analyses]
        masked = [a.masked for a in analyses]
        prior = sparse.csr_matrix(np.array([a.prior for a in analyses]) * self.prior_weight)
        blocks = []
        if fit:
            self.dv_ = DictVectorizer()
            blocks.append(self.dv_.fit_transform(feats))
        else:
            blocks.append(self.dv_.transform(feats))
        if self.delex:
            if fit:
                self.mvec_ = TfidfVectorizer(ngram_range=(1, 3), min_df=2, sublinear_tf=True, token_pattern=r"[^\s]+")
                blocks.append(self.mvec_.fit_transform(masked))
            else:
                blocks.append(self.mvec_.transform(masked))
        if self.raw_text:
            texts = [a.text for a in analyses]
            if fit:
                self.rvec_ = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)
                blocks.append(self.rvec_.fit_transform(texts))
            else:
                blocks.append(self.rvec_.transform(texts))
        blocks.append(prior)
        return sparse.hstack(blocks).tocsr()

    def fit(self, texts, y, sample_weight=None):
        analyses = default_analyzer().batch(texts)
        X = self._matrix(analyses, fit=True)
        y = np.asarray(y)
        if self.ordinal:
            self.clf_ = _Ordinal(self.C, self.class_weight).fit(X, y, sample_weight)
        else:
            self.clf_ = LogisticRegression(C=self.C, max_iter=3000, class_weight=self.class_weight)
            self.clf_.fit(X, y, sample_weight=sample_weight)
        return self

    def predict_proba(self, texts):
        X = self._matrix(default_analyzer().batch(texts), fit=False)
        return full_proba(self.clf_.predict_proba(X), self.clf_.classes_)

    def explain(self, text: str, top: int = 8) -> list[tuple[str, float]]:
        """Top features pushing toward the predicted level (non-ordinal model only)."""
        a = default_analyzer()(text)
        X = self._matrix([a], fit=False)
        if self.ordinal:
            return []
        k = int(np.argmax(self.clf_.predict_proba(X)[0]))
        names = list(self.dv_.get_feature_names_out())
        if self.delex:
            names += [f"masked:{t}" for t in self.mvec_.get_feature_names_out()]
        if self.raw_text:
            names += [f"word:{t}" for t in self.rvec_.get_feature_names_out()]
        names += [f"prior:{i + 1}" for i in range(N_LEVELS)]
        contrib = X.multiply(self.clf_.coef_[k]).toarray()[0]
        order = np.argsort(-contrib)[:top]
        return [(names[i], float(contrib[i])) for i in order if contrib[i] > 0]
