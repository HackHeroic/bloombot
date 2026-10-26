"""Cognitive-process lexicon: directive verbs and question frames -> prior over Bloom levels.

Built from the 19 cognitive processes of Anderson & Krathwohl (2001, Table 5.1) and their
listed synonyms, extended with verbs that are common in exam papers. A verb is *evidence*,
not a label: "explain" can ask for recall of a textbook explanation (Understand) or for an
explanation of an unseen situation (Analyze), and "write" means Apply in "write a program"
but Create in "write a proposal". Ambiguous verbs therefore carry a distribution, and the
models learn from data how much to trust it (see ``bloombot.models``).

Weights are on the order Remember, Understand, Apply, Analyze, Evaluate, Create.
"""

from __future__ import annotations

import re
from functools import lru_cache

import numpy as np

from bloombot.taxonomy import N_LEVELS, Level

R, U, AP, AN, E, C = (Level.REMEMBER, Level.UNDERSTAND, Level.APPLY, Level.ANALYZE, Level.EVALUATE, Level.CREATE)

# --- Anderson & Krathwohl (2001) cognitive processes and their alternative names --------------
COGNITIVE_PROCESSES: dict[Level, dict[str, tuple[str, ...]]] = {
    R: {"recognize": ("recognise", "identify"), "recall": ("retrieve",)},
    U: {
        "interpret": ("clarify", "paraphrase", "represent", "translate"),
        "exemplify": ("illustrate", "instantiate"),
        "classify": ("categorize", "categorise", "subsume"),
        "summarize": ("summarise", "abstract", "generalize", "generalise"),
        "infer": ("conclude", "extrapolate", "interpolate", "predict"),
        "compare": ("contrast", "map", "match"),
        "explain": ("construct a model",),
    },
    AP: {"execute": ("carry out",), "implement": ("use",)},
    AN: {
        "differentiate": ("discriminate", "distinguish", "focus", "select"),
        "organize": ("organise", "integrate", "outline", "parse", "structure"),
        "attribute": ("deconstruct",),
    },
    E: {"check": ("coordinate", "detect", "monitor", "test"), "critique": ("judge",)},
    C: {"generate": ("hypothesize", "hypothesise"), "plan": ("design",), "produce": ("construct",)},
}


def _one_hot(level: Level) -> dict[Level, float]:
    return {level: 1.0}


# --- Exam directive verbs ---------------------------------------------------------------------
# Unambiguous-enough verbs, by level.
_SINGLE: dict[Level, str] = {
    R: "define list name state recall recite memorize memorise label tabulate enumerate "
       "reproduce repeat quote mention locate find_out spell tell",
    U: "explain describe discuss summarize summarise paraphrase interpret classify categorize "
       "categorise exemplify illustrate infer predict restate rephrase review report express "
       "elaborate clarify extend generalize generalise outline_brief indicate associate translate "
       "convert estimate understand",
    AP: "apply solve compute calculate use execute implement demonstrate operate perform carry_out "
        "determine evaluate_expression find derive simulate trace sketch draw construct_table "
        "prepare practise practice show employ manipulate modify convert_to program code run "
        "measure verify_by plot work_out",
    AN: "analyze analyse differentiate distinguish discriminate examine investigate organize organise "
        "deconstruct attribute dissect diagnose categorise_by correlate contrast inspect "
        "break_down infer_from relate separate subdivide test_for characterize characterise",
    E: "evaluate assess judge critique criticize criticise justify defend argue appraise rate rank "
       "prioritize prioritise recommend conclude decide select_best validate verify check "
       "support debate weigh grade measure_against interpret_critically comment",
    C: "create design develop formulate compose construct invent devise generate plan produce "
       "propose hypothesize hypothesise build author originate synthesize synthesise combine "
       "compile integrate assemble modify_to rearrange reorganize reorganise rewrite adapt "
       "improve improvise imagine",
}

# Verbs that genuinely span levels; weights reflect common exam usage.
_AMBIGUOUS: dict[str, dict[Level, float]] = {
    "explain":    {U: 0.70, AN: 0.15, R: 0.10, E: 0.05},
    "describe":   {U: 0.55, R: 0.40, AN: 0.05},
    "discuss":    {U: 0.55, AN: 0.25, E: 0.20},
    "illustrate": {U: 0.60, AP: 0.30, R: 0.10},
    "identify":   {R: 0.55, AN: 0.30, U: 0.15},
    "outline":    {R: 0.45, U: 0.45, AN: 0.10},
    "show":       {AP: 0.45, U: 0.35, AN: 0.20},
    "find":       {AP: 0.60, R: 0.25, AN: 0.15},
    "write":      {AP: 0.40, R: 0.25, U: 0.15, C: 0.20},
    "draw":       {AP: 0.45, U: 0.30, R: 0.25},
    "give":       {R: 0.45, U: 0.35, E: 0.20},
    "compare":    {U: 0.45, AN: 0.50, E: 0.05},
    "contrast":   {AN: 0.55, U: 0.45},
    "classify":   {U: 0.55, AN: 0.40, R: 0.05},
    "select":     {AN: 0.40, E: 0.40, AP: 0.20},
    "choose":     {E: 0.50, AN: 0.30, AP: 0.20},
    "construct":  {C: 0.50, AP: 0.50},
    "develop":    {C: 0.80, AP: 0.20},
    "modify":     {C: 0.50, AP: 0.50},
    "prepare":    {AP: 0.50, C: 0.50},
    "examine":    {AN: 0.70, E: 0.30},
    "interpret":  {U: 0.60, AN: 0.40},
    "predict":    {U: 0.50, AN: 0.35, C: 0.15},
    "relate":     {AN: 0.60, U: 0.40},
    "conclude":   {E: 0.50, AN: 0.30, U: 0.20},
    "verify":     {E: 0.50, AP: 0.50},
    "check":      {E: 0.60, AP: 0.40},
    "test":       {E: 0.50, AN: 0.30, AP: 0.20},
    "prove":      {AP: 0.40, E: 0.40, AN: 0.20},
    "derive":     {AP: 0.70, AN: 0.30},
    "estimate":   {AP: 0.50, U: 0.30, E: 0.20},
    "determine":  {AP: 0.70, AN: 0.20, E: 0.10},
    "comment":    {E: 0.60, U: 0.40},
    "suggest":    {C: 0.50, E: 0.30, AP: 0.20},
    "recommend":  {E: 0.70, C: 0.30},
    "argue":      {E: 0.85, C: 0.15},
    "list":       {R: 0.95, U: 0.05},
    "state":      {R: 0.90, U: 0.10},
    "mention":    {R: 0.90, U: 0.10},
    "tell":       {R: 0.70, U: 0.30},
    "plan":       {C: 0.80, AP: 0.20},
    "integrate":  {C: 0.50, AN: 0.50},
    "organize":   {AN: 0.60, C: 0.40},
    "organise":   {AN: 0.60, C: 0.40},
    "sketch":     {AP: 0.60, U: 0.40},
    "trace":      {AP: 0.60, AN: 0.40},
    "justify":    {E: 1.0},
    "summarize":  {U: 1.0},
    "summarise":  {U: 1.0},
}

# Two-word directives; matched on lemma bigrams before single verbs.
PHRASES: dict[str, dict[Level, float]] = {
    "write note":       {U: 0.60, R: 0.40},   # "write short notes on"
    "write program":    {AP: 0.70, C: 0.30},
    "write code":       {AP: 0.75, C: 0.25},
    "write algorithm":  {AP: 0.60, C: 0.40},
    "write query":      {AP: 0.85, C: 0.15},
    "write essay":      {C: 0.60, E: 0.40},
    "write proposal":   {C: 0.90, E: 0.10},
    "write expression": {AP: 0.70, U: 0.30},
    "write equation":   {AP: 0.50, R: 0.30, U: 0.20},
    "write definition": {R: 1.0},
    "give example":     {U: 0.80, AP: 0.20},
    "give account":     {U: 0.70, R: 0.30},
    "give reason":      {U: 0.55, E: 0.45},
    "give opinion":     {E: 1.0},
    "give definition":  {R: 1.0},
    "draw diagram":     {U: 0.50, R: 0.30, AP: 0.20},
    "draw graph":       {AP: 0.80, AN: 0.20},
    "draw flowchart":   {AP: 0.70, C: 0.30},
    "carry out":        {AP: 1.0},
    "work out":         {AP: 1.0},
    "find out":         {AP: 0.60, R: 0.40},
    "point out":        {AN: 0.55, R: 0.45},
    "break down":       {AN: 1.0},
    "come up":          {C: 1.0},             # "come up with"
    "make use":         {AP: 1.0},
    "set up":           {AP: 0.50, C: 0.50},
    "bring out":        {AN: 0.50, U: 0.50},  # "bring out the differences"
    "account for":      {U: 0.50, AN: 0.50},
    "comment on":       {E: 0.60, U: 0.40},
    "make decision":    {E: 1.0},
    "make judgment":    {E: 1.0},
    "make judgement":   {E: 1.0},
    "make prediction":  {U: 0.60, AN: 0.40},
    "draw conclusion":  {E: 0.50, AN: 0.50},
    # mathematical "evaluate" means compute
    "evaluate integral":    {AP: 1.0},
    "evaluate expression":  {AP: 1.0},
    "evaluate limit":       {AP: 1.0},
    "evaluate determinant": {AP: 1.0},
    "evaluate derivative":  {AP: 1.0},
    "evaluate function":    {AP: 1.0},
    "evaluate sum":         {AP: 1.0},
}

# Question frames for items whose main predicate is not a Bloom verb ("What is a stack?").
# Ordered: first match wins. Patterns run on the lower-cased cleaned question.
FRAMES: list[tuple[str, re.Pattern[str], dict[Level, float]]] = [
    (name, re.compile(rx), dist)
    for name, rx, dist in [
        ("what_if", r"\bwhat (?:would|will|might|could) happen\b|\bwhat if\b", {AN: 0.45, U: 0.35, C: 0.20}),
        ("how_would_you_design", r"\bhow (?:would|will|could|can|might) you (?:design|create|develop|build|plan|improve|propose)", {C: 1.0}),
        ("how_would_you", r"\bhow (?:would|will|could|can|might|do|should) you\b", {AP: 0.55, C: 0.20, E: 0.15, AN: 0.10}),
        ("agree", r"\b(?:do you agree|would you agree|to what extent|is it (?:justified|fair|right|wise|better)|do you think|in your opinion|what is your (?:opinion|view))\b", {E: 0.90, AN: 0.10}),
        ("which_better", r"\b(?:which|what) (?:\w+ )?(?:is|are|would be) (?:the )?(?:better|best|more (?:suitable|appropriate|efficient|effective)|most (?:suitable|appropriate|efficient|effective))\b", {E: 0.80, AN: 0.20}),
        ("should", r"^(?:should|would it be|is it better)\b", {E: 0.80, AN: 0.20}),
        ("difference", r"\b(?:difference|differences|distinguish) between\b", {AN: 0.50, U: 0.45, R: 0.05}),
        ("relationship", r"\b(?:relationship|relation|connection|link) between\b", {AN: 0.70, U: 0.30}),
        ("advantages", r"\b(?:advantages?|disadvantages?|merits?|demerits?|pros and cons|limitations?|drawbacks?|benefits?)\b", {U: 0.50, R: 0.25, AN: 0.15, E: 0.10}),
        ("what_meant", r"\bwhat (?:is|are|do you) (?:meant|mean|understand) by\b", {U: 0.55, R: 0.45}),
        ("why", r"^why\b|\bwhy (?:is|are|does|do|did|was|were|should|would|can)\b", {U: 0.60, AN: 0.30, E: 0.10}),
        ("how_does", r"^how (?:does|do|is|are|did|can|was|were)\b", {U: 0.70, AP: 0.20, AN: 0.10}),
        ("how_many", r"^how (?:many|much|long|far|often)\b", {R: 0.50, AP: 0.50}),
        ("what_is", r"^(?:what|which) (?:is|are|was|were)\b", {R: 0.65, U: 0.35}),
        ("who_when_where", r"^(?:who|whom|whose|when|where)\b", {R: 0.95, U: 0.05}),
        ("what", r"^what\b", {R: 0.45, U: 0.45, AP: 0.10}),
        ("yes_no", r"^(?:is|are|can|does|do|was|were|has|have|will)\b", {R: 0.40, U: 0.40, E: 0.20}),
    ]
]


def _to_vector(dist: dict[Level, float]) -> np.ndarray:
    v = np.zeros(N_LEVELS)
    for lv, w in dist.items():
        v[int(lv) - 1] += w
    return v / v.sum()


def _build_verbs() -> dict[str, np.ndarray]:
    table: dict[str, dict[Level, float]] = {}
    for level, procs in COGNITIVE_PROCESSES.items():
        for head, alts in procs.items():
            for v in (head, *alts):
                if " " not in v:
                    table.setdefault(v, _one_hot(level))
    for level, words in _SINGLE.items():
        for w in words.split():
            if "_" not in w:
                table.setdefault(w, _one_hot(level))
    table.update(_AMBIGUOUS)
    return {v: _to_vector(d) for v, d in table.items()}


VERBS: dict[str, np.ndarray] = _build_verbs()
PHRASE_VECTORS: dict[str, np.ndarray] = {p: _to_vector(d) for p, d in PHRASES.items()}
FRAME_VECTORS: list[tuple[str, re.Pattern[str], np.ndarray]] = [(n, rx, _to_vector(d)) for n, rx, d in FRAMES]
UNIFORM = np.full(N_LEVELS, 1.0 / N_LEVELS)


@lru_cache(maxsize=4096)
def verb_prior(lemma: str) -> np.ndarray | None:
    v = VERBS.get(lemma.lower())
    return None if v is None else v.copy()


def phrase_prior(verb: str, particle_or_object: str) -> np.ndarray | None:
    v = PHRASE_VECTORS.get(f"{verb.lower()} {particle_or_object.lower()}")
    return None if v is None else v.copy()


def match_frame(text_lower: str) -> tuple[str, np.ndarray] | None:
    for name, rx, vec in FRAME_VECTORS:
        if rx.search(text_lower):
            return name, vec.copy()
    return None


def is_bloom_verb(lemma: str) -> bool:
    return lemma.lower() in VERBS


def levels_for(lemma: str, threshold: float = 0.2) -> list[Level]:
    """Levels a verb plausibly signals (weight >= threshold), most likely first."""
    v = verb_prior(lemma)
    if v is None:
        return []
    order = np.argsort(-v)
    return [Level(int(i) + 1) for i in order if v[i] >= threshold]
