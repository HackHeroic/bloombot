"""Linguistic analysis of a question: what is the student actually told to do?

The Bloom level is set by the *directive* - the predicate that tells the student what to do -
not by whichever Bloom verb happens to appear first. In "Explain how you would design a
schema", the directive is *explain*; *design* sits inside its complement and is a trap for a
bag-of-words model. We find directives from the dependency parse:

* the root of an imperative sentence (no subject, or subject "you"): "Define X", "Can you explain X"
* verbs coordinated with it: "Define X and *explain* Y"
* the complement of a request frame: "You are required to *design* ..."

Each directive carries a prior over levels from :mod:`bloombot.text.lexicon`, refined by its
object or particle ("write *notes*" vs "write a *program*"). A question with several directives
is labelled at the highest level it demands (OBE convention), so the priors are combined as the
distribution of the maximum: P(max <= k) = prod_i F_i(k). Questions with no directive fall back
to question frames ("What would happen if ...").

:func:`delexicalize` masks topic words (nouns, names, numbers) so that a model sees the
cognitive form of a question ("compare the <n> of <n> and <n>") rather than its subject. Topic
words are the main thing that does not transfer between datasets.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import asdict, dataclass, field
from functools import lru_cache
from typing import Iterable, Literal

import numpy as np

from bloombot.taxonomy import N_LEVELS, Level
from bloombot.text import lexicon
from bloombot.text.normalize import clean_question

DirectiveKind = Literal["imperative", "you_modal", "request", "coordinated"]

SPACY_MODEL = "en_core_web_sm"

# Nouns and adjectives that describe the *task* rather than the topic; kept when delexicalising.
CUE_WORDS = frozenset(
    """advantage advantages disadvantage disadvantages merit merits demerit demerits pro pros con cons
    difference differences similarity similarities example examples reason reasons importance significance
    role relationship relation comparison opinion view views note notes definition definitions diagram
    program code algorithm design plan proposal solution criteria criterion method methods step steps
    procedure essay report argument evidence conclusion alternative alternatives strategy model hypothesis
    experiment case scenario situation problem impact effect effects cause causes consequence implication
    implications limitation limitations feature features characteristic characteristics type types kind
    kinds function purpose application applications use value output result error flaw critique
    justification recommendation improvement approach way ways idea ideas term terms meaning concept
    principle principles process mechanism working structure part parts component components factor factors
    best better worst suitable appropriate effective efficient valid new novel original own given following
    different various main key brief short neat detail detailed suitable possible appropriate important
    major minor real specific general appropriate correct incorrect true false necessary sufficient""".split()
)
REQUEST_VERBS = frozenset({"require", "ask", "expect", "need", "want", "instruct", "invite", "encourage"})
_LEADING_SKIP = frozenset({"briefly", "clearly", "critically", "carefully", "now", "then", "also", "please", "hence", "thus", "next", "finally", "first", "further", "so"})
_WH = frozenset({"what", "which", "who", "whom", "whose", "when", "where", "why", "how"})
_EMBEDDED_DEPS = frozenset({"xcomp", "ccomp", "advcl", "relcl", "acl", "pcomp", "csubj", "pobj", "dobj", "nsubj"})


@dataclass
class Directive:
    lemma: str
    text: str
    kind: DirectiveKind
    obj: str | None = None
    phrase: str | None = None
    negated: bool = False
    prior: list[float] = field(default_factory=list)

    @property
    def top_level(self) -> Level:
        return Level(int(np.argmax(self.prior)) + 1)


@dataclass
class Analysis:
    text: str
    directives: list[Directive]
    frame: str | None
    is_question: bool
    wh_word: str | None
    n_tokens: int
    n_sentences: int
    embedded_bloom_verbs: list[str]
    masked: str
    prior: list[float]
    evidence: Literal["directive", "frame", "none"]
    tokens: list[tuple[str, str]] = field(default_factory=list)  # (lower-cased word, coarse POS)

    @property
    def verbs(self) -> list[str]:
        return [d.lemma for d in self.directives if not d.negated]

    def to_dict(self) -> dict:
        return asdict(self)


# --------------------------------------------------------------------------------------------
@lru_cache(maxsize=1)
def get_nlp():
    import spacy

    try:
        return spacy.load(SPACY_MODEL, disable=["ner"])
    except OSError as exc:  # pragma: no cover - environment problem, not logic
        raise RuntimeError(
            f"spaCy model {SPACY_MODEL!r} is not installed: python -m spacy download {SPACY_MODEL}"
        ) from exc


def combine_max(priors: Iterable[np.ndarray]) -> np.ndarray:
    """Distribution of max(L_1..L_n) for independent levels L_i ~ prior_i."""
    priors = list(priors)
    if not priors:
        return lexicon.UNIFORM.copy()
    cdf = np.ones(N_LEVELS)
    for p in priors:
        cdf *= np.cumsum(p)
    pmf = np.diff(np.concatenate([[0.0], cdf]))
    pmf = np.clip(pmf, 0, None)
    return pmf / pmf.sum()


def _lemma(tok) -> str:
    lem = tok.lemma_.lower()
    # spaCy sometimes lemmatises a sentence-initial imperative as a noun ("Lists" -> "list" is
    # fine, but "Analyses" -> "analysis"); prefer the surface form when it is a known verb.
    low = tok.text.lower()
    if low in lexicon.VERBS and lem not in lexicon.VERBS:
        return low
    return lem


def _is_negated(tok) -> bool:
    if any(c.dep_ == "neg" for c in tok.children):
        return True
    # "Do not define X" parses as root "do" + neg + xcomp/conj "define" in small models.
    head = tok.head
    return head is not tok and head.lemma_ == "do" and any(c.dep_ == "neg" for c in head.children)


def _object_or_particle(tok) -> tuple[str | None, str | None]:
    """Return (object lemma, phrase key) for a directive token."""
    verb = _lemma(tok)
    prt = next((c for c in tok.children if c.dep_ == "prt"), None)
    if prt is not None:
        key = f"{verb} {prt.lemma_.lower()}"
        if key in lexicon.PHRASES:
            return None, key
    objs = [c for c in tok.children if c.dep_ in ("dobj", "obj", "attr", "oprd")]
    if not objs:
        # "write short notes on X" is sometimes parsed with "notes" as npadvmod / nsubj.
        objs = [c for c in tok.children if c.pos_ in ("NOUN", "PROPN") and c.dep_ in ("npadvmod", "nsubj", "advmod", "compound")]
    for c in tok.children:
        if c.dep_ == "prep" and f"{verb} {c.lemma_.lower()}" in lexicon.PHRASES:
            return None, f"{verb} {c.lemma_.lower()}"
    if objs:
        obj = objs[0].lemma_.lower()
        key = f"{verb} {obj}"
        return obj, key if key in lexicon.PHRASES else None
    return None, None


def _directive(tok, kind: DirectiveKind) -> Directive | None:
    lemma = _lemma(tok)
    obj, phrase = _object_or_particle(tok)
    if phrase is not None:
        prior = lexicon.phrase_prior(*phrase.split(" ", 1))
    else:
        prior = lexicon.verb_prior(lemma)
    if prior is None:
        return None
    return Directive(lemma=lemma, text=tok.text, kind=kind, obj=obj, phrase=phrase,
                     negated=_is_negated(tok), prior=prior.tolist())


def _is_candidate(tok) -> bool:
    """A token that could head a directive: a Bloom verb, or a phrasal verb such as "come up"."""
    if _lemma(tok) in lexicon.VERBS:
        return True
    return any(c.dep_ == "prt" and f"{_lemma(tok)} {c.lemma_.lower()}" in lexicon.PHRASES for c in tok.children)


def _first_content(sent):
    for tok in sent:
        if tok.is_punct or tok.lower_ in _LEADING_SKIP or tok.pos_ in ("ADV", "INTJ") and tok.lower_ not in lexicon.VERBS:
            continue
        return tok
    return None


def _sentence_directives(sent) -> list[Directive]:
    root = sent.root
    found: list[tuple] = []  # (token, kind)
    first = _first_content(sent)

    subj = [c for c in root.children if c.dep_ in ("nsubj", "nsubjpass")]
    subj_is_you = bool(subj) and all(s.lower_ in ("you", "students", "student", "candidates") for s in subj)
    starts_wh = first is not None and first.lower_ in _WH

    # 1. Imperative root (also catches "List the ..." mis-tagged as a noun by a small model).
    if (
        first is not None
        and not starts_wh
        and _is_candidate(first)
        # "Using Ohm's law, calculate ...": a participle opening a clause is not the directive.
        and (first is root or (
            first.tag_ not in ("VBG", "VBN", "VBD", "VBZ")
            and first.dep_ not in ("nsubj", "nsubjpass", "aux", "auxpass", "compound", "amod")
            and not subj  # a declarative sentence ("Design patterns are ...") has a subject
        ))
        and not any(c.dep_ in ("nsubj", "nsubjpass") for c in first.children)
    ):
        found.append((first, "imperative"))
    if root.pos_ in ("VERB", "AUX") and not subj and not starts_wh and root not in [f[0] for f in found]:
        if root.lemma_.lower() in lexicon.VERBS or root.lower_ in lexicon.VERBS:
            found.append((root, "imperative"))
    # 2. "Can you explain ...", "Students should design ..." (but not "How would you ...": a frame).
    if subj_is_you and not starts_wh and (root.lemma_.lower() in lexicon.VERBS):
        found.append((root, "you_modal"))
    # 3. Request frames: "You are required to design ...", "Students are asked to evaluate ...".
    if root.lemma_.lower() in REQUEST_VERBS:
        for c in root.children:
            if c.dep_ in ("xcomp", "ccomp") and c.pos_ == "VERB":
                found.append((c, "request"))

    # 4. Verbs coordinated with any directive: "Define X and explain Y", "Compare and contrast".
    seen = {t.i for t, _ in found}
    queue = [t for t, _ in found]
    while queue:
        tok = queue.pop()
        for c in tok.children:
            if c.dep_ == "conj" and c.i not in seen and (c.pos_ == "VERB" or c.lower_ in lexicon.VERBS):
                seen.add(c.i)
                found.append((c, "coordinated"))
                queue.append(c)

    # 5. A sequenced second task the parser attached elsewhere: "..., and then design X", "...; evaluate Y".
    for tok in sent:
        if tok.i in seen or tok.tag_ != "VB" or not _is_candidate(tok):
            continue
        prev = tok.nbor(-1) if tok.i > sent.start else None
        if prev is not None and (prev.lower_ in ("then", "also", "hence", "thus") or prev.text == ";") and not any(
            c.dep_ in ("aux", "nsubj", "mark") for c in tok.children
        ):
            seen.add(tok.i)
            found.append((tok, "coordinated"))
    out: list[Directive] = []
    for tok, kind in sorted(found, key=lambda x: x[0].i):
        d = _directive(tok, kind)
        if d is not None and all(d.text != o.text or d.lemma != o.lemma for o in out):
            out.append(d)
    return out


def delexicalize(doc_or_tokens, keep: set[int] | None = None) -> str:
    """Replace topic words with placeholders; keep verbs, function words and task cues."""
    keep = keep or set()
    parts: list[str] = []
    for tok in doc_or_tokens:
        low = tok.lower_
        is_verb_use = tok.pos_ in ("VERB", "AUX") and (low in lexicon.VERBS or tok.lemma_.lower() in lexicon.VERBS)
        if tok.i in keep or is_verb_use or low in CUE_WORDS or tok.lemma_.lower() in CUE_WORDS:
            piece = low
        elif tok.like_num or tok.pos_ == "NUM":
            piece = "<num>"
        elif tok.pos_ == "PROPN":
            piece = "<name>"
        elif tok.pos_ in ("NOUN", "X", "SYM"):
            piece = "<n>"
        elif tok.pos_ == "ADJ":
            piece = "<adj>"
        else:
            piece = low
        if parts and piece == parts[-1] and piece.startswith("<"):
            continue
        parts.append(piece)
    return " ".join(parts)


def analyze_doc(doc, text: str | None = None) -> Analysis:
    text = text if text is not None else doc.text
    directives: list[Directive] = []
    keep: set[int] = set()
    for sent in doc.sents:
        ds = _sentence_directives(sent)
        directives.extend(ds)
    directive_texts = {d.text for d in directives}
    keep = {t.i for t in doc if t.text in directive_texts}

    lower = text.lower()
    frame_hit = lexicon.match_frame(lower)
    frame = frame_hit[0] if frame_hit else None

    active = [np.asarray(d.prior) for d in directives if not d.negated]
    if active:
        prior, evidence = combine_max(active), "directive"
    elif frame_hit is not None:
        prior, evidence = frame_hit[1], "frame"
    else:
        prior, evidence = lexicon.UNIFORM.copy(), "none"

    directive_idx = keep
    embedded = sorted({
        _lemma(t) for t in doc
        if t.i not in directive_idx and t.pos_ == "VERB" and _lemma(t) in lexicon.VERBS and t.dep_ in _EMBEDDED_DEPS | {"conj"}
    })
    first = next((t for t in doc if not t.is_punct), None)
    wh = first.lower_ if first is not None and first.lower_ in _WH else next(
        (t.lower_ for t in doc if t.lower_ in _WH and t.tag_ in ("WDT", "WP", "WP$", "WRB")), None)
    return Analysis(
        text=text,
        directives=directives,
        frame=frame,
        is_question=text.rstrip().endswith("?") or (first is not None and (first.lower_ in _WH or first.tag_ == "MD")),
        wh_word=wh,
        n_tokens=sum(1 for t in doc if not t.is_punct),
        n_sentences=sum(1 for _ in doc.sents),
        embedded_bloom_verbs=embedded,
        masked=delexicalize(doc, keep),
        prior=prior.tolist(),
        evidence=evidence,
        tokens=[(t.lower_, t.pos_) for t in doc if not t.is_space],
    )


# --------------------------------------------------------------------------------------------
class Analyzer:
    """Batch analyser with an in-memory cache keyed by the cleaned text."""

    def __init__(self, batch_size: int = 256, n_process: int = 1):
        self.batch_size = batch_size
        self.n_process = n_process
        self._cache: dict[str, Analysis] = {}

    @staticmethod
    def key(text: str) -> str:
        return hashlib.sha1(text.encode()).hexdigest()

    def __call__(self, text: str) -> Analysis:
        return self.batch([text])[0]

    def batch(self, texts: Iterable[str]) -> list[Analysis]:
        texts = [clean_question(t) for t in texts]
        todo = list(dict.fromkeys(t for t in texts if self.key(t) not in self._cache))
        if todo:
            nlp = get_nlp()
            for text, doc in zip(todo, nlp.pipe(todo, batch_size=self.batch_size, n_process=self.n_process)):
                self._cache[self.key(text)] = analyze_doc(doc, text)
        return [self._cache[self.key(t)] for t in texts]

    def __len__(self) -> int:
        return len(self._cache)


_DEFAULT: Analyzer | None = None


def default_analyzer() -> Analyzer:
    global _DEFAULT
    if _DEFAULT is None:
        _DEFAULT = Analyzer()
    return _DEFAULT


def analyze(text: str) -> Analysis:
    return default_analyzer()(text)


_WORD = re.compile(r"[A-Za-z][A-Za-z\-']+")


def content_words(text: str) -> set[str]:
    """Lemmas of the topic-bearing words (nouns, proper nouns, adjectives) for topic-overlap
    checks, so "turbine" matches "turbines". Bloom verbs and cue words are not topic."""
    from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

    words = {t.lemma_.lower() for t in get_nlp()(text)
             if t.pos_ in {"NOUN", "PROPN", "ADJ", "NUM"} and t.text.isalnum() and any(c.isalpha() for c in t.text)}
    return words - ENGLISH_STOP_WORDS - set(lexicon.VERBS) - CUE_WORDS


def topic_overlap(original: set[str], new: set[str]) -> float:
    """Share of the original's content words that the new text keeps. Words of 5+ letters
    also match a derivation sharing their stem ("marketing" ~ "market")."""
    def kept(w: str) -> bool:
        return w in new or (len(w) >= 5 and any(len(u) >= 5 and (w.startswith(u) or u.startswith(w)) for u in new))

    return sum(kept(w) for w in original) / max(1, len(original))
