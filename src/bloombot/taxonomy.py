"""The revised Bloom taxonomy (Anderson & Krathwohl, 2001) and label normalisation.

Datasets label questions in three incompatible ways:

* revised names   Remember, Understand, Apply, Analyze, Evaluate, Create
* original names  Knowledge, Comprehension, Application, Analysis, Synthesis, Evaluation
* numeric codes   1-6, BT1-BT6, L1-L6, K1-K6

The original order puts Synthesis (level 5) *before* Evaluation (level 6); the revised order
swaps them (Evaluate = 5, Create = 6). Names are unambiguous, but a bare code such as "BT5" is
not: it means Evaluate in the revised scheme and Synthesis -> Create in the original one. So
:func:`parse_label` never guesses the scheme for numeric codes; the caller must say which one
the source uses.
"""

from __future__ import annotations

import re
from enum import IntEnum
from typing import Literal

Scheme = Literal["revised", "original"]


class Level(IntEnum):
    REMEMBER = 1
    UNDERSTAND = 2
    APPLY = 3
    ANALYZE = 4
    EVALUATE = 5
    CREATE = 6

    @property
    def label(self) -> str:
        return self.name.capitalize()

    @property
    def is_lower_order(self) -> bool:
        """LOTS = Remember/Understand/Apply; HOTS = Analyze/Evaluate/Create."""
        return self <= Level.APPLY

    @classmethod
    def from_label(cls, label: str) -> Level:
        return cls[label.strip().upper()]


LEVELS: tuple[Level, ...] = tuple(Level)
LABELS: tuple[str, ...] = tuple(lv.label for lv in LEVELS)
N_LEVELS = len(LEVELS)

DESCRIPTIONS: dict[Level, str] = {
    Level.REMEMBER: "Retrieve relevant knowledge from long-term memory (recognise, recall).",
    Level.UNDERSTAND: "Construct meaning: interpret, exemplify, classify, summarise, infer, compare, explain.",
    Level.APPLY: "Carry out or use a procedure in a given situation (execute, implement).",
    Level.ANALYZE: "Break material into parts and determine how they relate (differentiate, organise, attribute).",
    Level.EVALUATE: "Make judgements based on criteria and standards (check, critique).",
    Level.CREATE: "Put elements together into a new coherent whole or original product (generate, plan, produce).",
}

_NAMES: dict[str, Level] = {
    # revised
    "remember": Level.REMEMBER, "remembering": Level.REMEMBER,
    "understand": Level.UNDERSTAND, "understanding": Level.UNDERSTAND,
    "apply": Level.APPLY, "applying": Level.APPLY,
    "analyze": Level.ANALYZE, "analyse": Level.ANALYZE, "analyzing": Level.ANALYZE, "analysing": Level.ANALYZE,
    "evaluate": Level.EVALUATE, "evaluating": Level.EVALUATE,
    "create": Level.CREATE, "creating": Level.CREATE,
    # original (Bloom, 1956)
    "knowledge": Level.REMEMBER,
    "comprehension": Level.UNDERSTAND,
    "application": Level.APPLY,
    "analysis": Level.ANALYZE,
    "synthesis": Level.CREATE,
    "evaluation": Level.EVALUATE,
}

_ORIGINAL_ORDER: dict[int, Level] = {
    1: Level.REMEMBER, 2: Level.UNDERSTAND, 3: Level.APPLY,
    4: Level.ANALYZE, 5: Level.CREATE, 6: Level.EVALUATE,  # 5 = Synthesis, 6 = Evaluation
}

_CODE = re.compile(r"^(?:bt|bl|l|k|level)?\s*[-_ ]?\s*([1-6])$")


def parse_label(value: object, scheme: Scheme | None = None) -> Level | None:
    """Map a raw dataset label to a :class:`Level`; ``None`` if it cannot be read.

    Names are read regardless of ``scheme``. Numeric codes need an explicit ``scheme``;
    without one they raise, because guessing silently swaps Evaluate and Create.
    """
    if value is None:
        return None
    if isinstance(value, Level):
        return value
    if isinstance(value, float):
        if value != value:  # NaN
            return None
        value = int(value)
    text = str(value).strip().lower()
    if not text:
        return None
    if text in _NAMES:
        return _NAMES[text]
    match = _CODE.match(text)
    if match is None:
        return None
    code = int(match.group(1))
    if scheme is None:
        raise ValueError(f"numeric Bloom code {value!r} is ambiguous; pass scheme='revised' or 'original'")
    return Level(code) if scheme == "revised" else _ORIGINAL_ORDER[code]
