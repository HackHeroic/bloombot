"""HTTP API. Run with ``bloombot serve`` (or ``uvicorn bloombot.api.app:app``); docs at /docs.

All responses use the models in :mod:`bloombot.pipeline.schemas`, which are the contract for
any frontend. Long operations (rebalancing a paper with an LLM) run as background jobs: the
POST returns a job id and the client polls ``GET /v1/jobs/{id}``.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Any

from fastapi import Depends, FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from bloombot import __version__
from bloombot.api.services import JobState, LabelledExample, LiveTrace, Profile, Services
from bloombot.pipeline import blueprint as bp_mod
from bloombot.pipeline.conductor import Conductor
from bloombot.pipeline.intake import parse_file, parse_text
from bloombot.pipeline.rewrite import Rewriter
from bloombot.pipeline.schemas import (
    Blueprint, LevelName, Paper, PaperReport, Prediction, RebalanceResult, RewriteJob, RewriteResult,
)
from bloombot.taxonomy import DESCRIPTIONS, LEVELS, Level
from bloombot.text.linguistics import default_analyzer


# --- request bodies ---------------------------------------------------------------------------------------
class ClassifyRequest(BaseModel):
    questions: list[str] = Field(min_length=1, max_length=500)
    use_llm: bool | None = Field(default=None, description="None = server policy (LLM only when unsure)")
    profile_id: str | None = None
    explain: bool = False


class ClassifyResponse(BaseModel):
    predictions: list[Prediction]
    model: str
    llm_used: bool


class AnalyzeRequest(BaseModel):
    question: str


class ParseTextRequest(BaseModel):
    text: str
    title: str = ""
    bt_scheme: str | None = Field(default="revised", description="how printed BT1-BT6 codes are numbered")


class AssessRequest(BaseModel):
    paper: Paper
    blueprint_id: str | None = None
    blueprint: Blueprint | None = None
    use_llm: bool | None = None
    profile_id: str | None = None


class PlanRequest(BaseModel):
    report: PaperReport
    blueprint_id: str | None = None
    blueprint: Blueprint | None = None


class RewriteRequest(BaseModel):
    question: str
    target_level: LevelName
    from_level: LevelName | None = None
    context: str | None = None
    marks: float | None = None
    other_questions: list[str] = Field(default_factory=list)
    profile_id: str | None = None


class ProfileRequest(BaseModel):
    name: str
    examples: list[LabelledExample] = Field(min_length=5)


# --- app --------------------------------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.services = Services()
    yield


def services(request: Request) -> Services:
    return request.app.state.services


def _blueprint(s: Services, bp_id: str | None, bp: Blueprint | None) -> Blueprint | None:
    if bp is not None:
        return bp
    if bp_id is None:
        return None
    try:
        return s.blueprints()[bp_id]
    except KeyError:
        raise HTTPException(404, f"unknown blueprint {bp_id!r}") from None


def _assessor(s: Services, profile_id: str | None):
    try:
        return s.assessor_for(profile_id)
    except KeyError:
        raise HTTPException(404, f"unknown profile {profile_id!r}") from None


def create_app() -> FastAPI:
    app = FastAPI(title="BloomBot API", version=__version__, lifespan=lifespan,
                  description="Classify exam questions by revised Bloom level, check papers against a "
                              "blueprint, and rewrite questions to a target level with automatic checks.")
    app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

    @app.get("/health")
    def health(s: Services = Depends(services)) -> dict[str, Any]:
        return {"status": "ok", "version": __version__}

    @app.get("/v1/meta")
    def meta(s: Services = Depends(services)) -> dict[str, Any]:
        a = s.assessor
        card = s.datacard()
        return {
            "version": __version__,
            "lexis": a.lexis.name,
            "llm": {"model": s.settings.llm, "available": s.llm_available, "policy": s.settings.llm_policy},
            "levels": [{"id": int(lv), "name": lv.label, "description": DESCRIPTIONS[lv],
                        "lower_order": lv.is_lower_order} for lv in LEVELS],
            "data": {"total": card.get("total_rows"),
                     "sources": {k: {"rows": v["rows"], "role": v["role"], "licence": v["licence"]}
                                 for k, v in card.get("sources", {}).items()}},
            "thresholds": {"review_confidence": s.settings.review_confidence,
                           "llm_trigger_confidence": s.settings.llm_trigger_confidence},
        }

    @app.get("/v1/blueprints")
    def blueprints(s: Services = Depends(services)) -> list[Blueprint]:
        return list(s.blueprints().values())

    @app.post("/v1/analyze")
    def analyze(req: AnalyzeRequest) -> dict[str, Any]:
        a = default_analyzer()(req.question)
        d = a.to_dict()
        d.pop("tokens", None)
        d["prior"] = {lv.label: round(p, 4) for lv, p in zip(LEVELS, a.prior)}
        return d

    @app.post("/v1/classify")
    def classify(req: ClassifyRequest, s: Services = Depends(services)) -> ClassifyResponse:
        a = _assessor(s, req.profile_id)
        preds = a.assess(req.questions, use_llm=req.use_llm, explain=req.explain)
        used = any(v.agent == "Socrates" for p in preds for v in p.votes)
        return ClassifyResponse(predictions=preds, model=a.lexis.name, llm_used=used)

    @app.post("/v1/papers/parse")
    async def parse_upload(file: UploadFile = File(...), bt_scheme: str = Form("revised")) -> Paper:
        data = await file.read()
        if len(data) > 20 * 1024 * 1024:
            raise HTTPException(413, "file larger than 20 MB")
        return parse_file(data, file.filename or "paper.txt", bt_scheme or None)

    @app.post("/v1/papers/parse-text")
    def parse_body(req: ParseTextRequest) -> Paper:
        return parse_text(req.text, req.title, req.bt_scheme)

    @app.post("/v1/papers/assess")
    def assess_paper(req: AssessRequest, s: Services = Depends(services)) -> PaperReport:
        if not req.paper.questions:
            raise HTTPException(422, "the paper has no questions")
        c = Conductor(_assessor(s, req.profile_id), s.settings)
        return c.assess_paper(req.paper, _blueprint(s, req.blueprint_id, req.blueprint), req.use_llm)

    @app.post("/v1/papers/plan")
    def plan(req: PlanRequest, s: Services = Depends(services)) -> list[RewriteJob]:
        bp = _blueprint(s, req.blueprint_id, req.blueprint)
        if bp is None:
            raise HTTPException(422, "a blueprint is required")
        return bp_mod.plan(req.report, bp)

    @app.post("/v1/rewrite")
    def rewrite(req: RewriteRequest, s: Services = Depends(services)) -> RewriteResult:
        a = _assessor(s, req.profile_id)
        from_level = req.from_level or a.assess([req.question])[0].level
        job = RewriteJob(question_id="q", original=req.question, from_level=from_level, to_level=req.target_level,
                         marks=req.marks, reason="requested")
        rw = Rewriter(a, a.llm.client if a.llm else None, s.settings)
        return rw.run(job, req.other_questions, req.context)

    @app.post("/v1/papers/rebalance", status_code=202)
    def rebalance(req: AssessRequest, s: Services = Depends(services)) -> JobState:
        bp = _blueprint(s, req.blueprint_id, req.blueprint)
        if bp is None:
            raise HTTPException(422, "a blueprint is required")
        a = _assessor(s, req.profile_id)

        def work(job: JobState) -> RebalanceResult:
            c = Conductor(a, s.settings, trace=LiveTrace(job.trace))
            return c.rebalance(req.paper, bp, req.use_llm)

        return s.submit("rebalance", work)

    @app.get("/v1/jobs/{job_id}")
    def job(job_id: str, s: Services = Depends(services)) -> JobState:
        j = s.get_job(job_id)
        if j is None:
            raise HTTPException(404, "unknown job")
        return j

    @app.get("/v1/profiles")
    def profiles(s: Services = Depends(services)) -> list[Profile]:
        return s.list_profiles()

    @app.post("/v1/profiles", status_code=201)
    def create_profile(req: ProfileRequest, s: Services = Depends(services)) -> Profile:
        return s.create_profile(req.name, req.examples)

    @app.delete("/v1/profiles/{profile_id}", status_code=204)
    def delete_profile(profile_id: str, s: Services = Depends(services)) -> None:
        if not s.delete_profile(profile_id):
            raise HTTPException(404, "unknown profile")

    @app.get("/v1/experiments")
    def experiments(s: Services = Depends(services)) -> dict[str, Any]:
        return s.experiments()

    @app.get("/v1/experiments/tables")
    def experiment_tables(s: Services = Depends(services)) -> list[str]:
        return s.experiment_tables()

    @app.get("/v1/experiments/tables/{name}")
    def experiment_table(name: str, s: Services = Depends(services)) -> Any:
        t = s.experiment_table(name)
        if t is None:
            raise HTTPException(404, f"unknown table {name!r}")
        return t

    return app


app = create_app()
