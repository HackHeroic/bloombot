"""Typed data passed between pipeline stages and returned by the API.

These models are the contract with any frontend: the API serialises exactly these.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator

from bloombot.taxonomy import LABELS, Level

LevelName = Literal["Remember", "Understand", "Apply", "Analyze", "Evaluate", "Create"]


def level_name(v: int | Level) -> str:
    return LABELS[int(v) - 1]


# --- questions and papers -------------------------------------------------------------------------
class Question(BaseModel):
    id: str = Field(description="stable id inside the paper, e.g. '2b' or '3-or'")
    number: str = Field(default="", description="numbering as printed, e.g. 'Q2 (b)'")
    text: str
    context: str | None = Field(default=None, description="shared stem of a multi-part question")
    marks: float | None = None
    section: str | None = None
    group: str | None = Field(default=None, description="id shared by OR-alternatives")
    option: str | None = Field(default=None, description="which alternative of the group this item belongs to")
    course_outcome: str | None = None
    tagged_level: LevelName | None = Field(default=None, description="level printed on the paper, if any")
    locked: bool = Field(default=False, description="teacher marked it as not to be rewritten")


class Paper(BaseModel):
    title: str = ""
    questions: list[Question]
    total_marks: float | None = None
    warnings: list[str] = Field(default_factory=list)


# --- classification --------------------------------------------------------------------------------
class DirectiveOut(BaseModel):
    verb: str
    kind: str
    phrase: str | None = None
    negated: bool = False


class Neighbour(BaseModel):
    question: str
    level: LevelName
    similarity: float
    source: str | None = None


class Vote(BaseModel):
    agent: str
    level: LevelName
    confidence: float
    proba: list[float]
    reasoning: str | None = None


class Prediction(BaseModel):
    question: str
    level: LevelName
    level_id: int
    confidence: float
    proba: dict[str, float]
    needs_review: bool
    lower_order: bool
    directives: list[DirectiveOut] = Field(default_factory=list)
    frame: str | None = None
    evidence: str = "none"
    votes: list[Vote] = Field(default_factory=list)
    neighbours: list[Neighbour] = Field(default_factory=list)
    explanation: list[tuple[str, float]] = Field(default_factory=list)


# --- blueprint and paper report ----------------------------------------------------------------
class LevelBound(BaseModel):
    levels: list[LevelName]
    min_share: float | None = Field(default=None, ge=0, le=1)
    max_share: float | None = Field(default=None, ge=0, le=1)
    label: str = ""

    def describe(self) -> str:
        name = self.label or "+".join(self.levels)
        parts = []
        if self.min_share is not None:
            parts.append(f">= {self.min_share:.0%}")
        if self.max_share is not None:
            parts.append(f"<= {self.max_share:.0%}")
        return f"{name} {' and '.join(parts)} of marks"


class Blueprint(BaseModel):
    id: str
    name: str
    description: str = ""
    source: str = ""
    bounds: list[LevelBound]
    weight_by: Literal["marks", "count"] = "marks"


class BoundCheck(BaseModel):
    bound: LevelBound
    description: str
    share: float
    ok: bool
    gap: float = Field(description="share to move in (+) or out (-) of these levels to comply")


class QuestionAssessment(BaseModel):
    question: Question
    prediction: Prediction


class PaperReport(BaseModel):
    paper: Paper
    items: list[QuestionAssessment]
    distribution: dict[str, float] = Field(description="share of weight per level")
    weight_by: str
    checks: list[BoundCheck] = Field(default_factory=list)
    compliant: bool | None = None
    needs_review: list[str] = Field(default_factory=list)


# --- rewriting -----------------------------------------------------------------------------------------
class CheckResult(BaseModel):
    name: str
    passed: bool
    detail: str = ""
    score: float | None = None


class Attempt(BaseModel):
    number: int
    question: str
    composer: str
    rationale: str | None = None
    checks: list[CheckResult]
    passed: bool
    predicted_level: LevelName | None = None


class RewriteJob(BaseModel):
    question_id: str
    original: str
    from_level: LevelName
    to_level: LevelName
    marks: float | None = None
    reason: str = ""


class RewriteResult(BaseModel):
    job: RewriteJob
    accepted: bool
    final_question: str | None
    attempts: list[Attempt]
    needs_teacher: bool


class RebalanceResult(BaseModel):
    before: PaperReport
    plan: list[RewriteJob]
    rewrites: list[RewriteResult]
    after: PaperReport
    final_paper: Paper


# --- audit trail -----------------------------------------------------------------------------------------
class TraceEvent(BaseModel):
    at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    agent: str
    action: str
    detail: dict[str, Any] = Field(default_factory=dict)
    ms: float | None = None

    @field_validator("detail")
    @classmethod
    def _short(cls, v: dict[str, Any]) -> dict[str, Any]:
        return {k: (s[:500] + "...") if isinstance(s, str) and len(s) > 500 else s for k, s in v.items()}
