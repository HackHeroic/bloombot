"""Long-lived services behind the API: models, institution profiles and background jobs."""

from __future__ import annotations

import json
import logging
import threading
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from functools import cached_property
from pathlib import Path
from typing import Any, Callable

import pandas as pd
from pydantic import BaseModel, Field

from bloombot.config import Settings, get_settings
from bloombot.models.base import BloomClassifier
from bloombot.pipeline.assess import Assessor, Cancelled
from bloombot.pipeline.blueprint import load_blueprints
from bloombot.pipeline.schemas import Blueprint, LevelName
from bloombot.taxonomy import Level

log = logging.getLogger(__name__)


class LabelledExample(BaseModel):
    question: str
    level: LevelName


class Profile(BaseModel):
    id: str
    name: str
    examples: list[LabelledExample]
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class LiveTrace(list):
    """A pipeline trace that also copies each event into a job as it happens, so clients polling
    GET /v1/jobs/{id} can follow the agents live instead of waiting for the result."""

    def __init__(self, sink: list[dict]):
        super().__init__()
        self.sink = sink

    def append(self, event) -> None:
        super().append(event)
        self.sink.append(event.model_dump(mode="json"))


class JobState(BaseModel):
    id: str
    kind: str
    status: str = "queued"  # queued | running | done | failed | cancelled
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    finished_at: datetime | None = None
    result: Any = None
    error: str | None = None
    trace: list[dict] = Field(default_factory=list)


class Services:
    def __init__(self, settings: Settings | None = None):
        self.settings = settings or get_settings()
        self._lock = threading.Lock()
        self._load_lock = threading.Lock()
        self._profiles: dict[str, Assessor] = {}
        self.jobs: dict[str, JobState] = {}
        # Ollama already serialises LLM calls; the pool only needs to be wide enough that a long LLM
        # job (or a cancelled one finishing its last calls) never blocks a fast one behind it.
        self._pool = ThreadPoolExecutor(max_workers=8)
        self.cancels: dict[str, threading.Event] = {}

    # -- models ---------------------------------------------------------------------------------------------
    @cached_property
    def assessor(self) -> Assessor:
        with self._load_lock:  # a request arriving during warm-up waits instead of loading twice
            if "assessor" not in self.__dict__:
                self.__dict__["assessor"] = Assessor.load(self.settings)
            return self.__dict__["assessor"]

    def warm_up(self) -> None:
        """Load the models in the background, so the first request does not wait ~20 s."""
        threading.Thread(target=lambda: self.assessor, name="warm-up", daemon=True).start()

    def blueprints(self) -> dict[str, Blueprint]:
        return load_blueprints(self.settings.root / "configs" / "blueprints")

    @property
    def llm_available(self) -> bool:
        return self.assessor.llm is not None

    # -- profiles -------------------------------------------------------------------------------------------
    @property
    def profile_dir(self) -> Path:
        return self.settings.artifacts / "profiles"

    def list_profiles(self) -> list[Profile]:
        if not self.profile_dir.exists():
            return []
        return [Profile.model_validate_json((d / "profile.json").read_text())
                for d in sorted(self.profile_dir.iterdir()) if (d / "profile.json").exists()]

    def create_profile(self, name: str, examples: list[LabelledExample]) -> Profile:
        prof = Profile(id=uuid.uuid4().hex[:12], name=name, examples=examples)
        adapted = self.assessor.with_examples([e.question for e in examples],
                                              [int(Level.from_label(e.level)) for e in examples], name)
        d = self.profile_dir / prof.id
        d.mkdir(parents=True, exist_ok=True)
        (d / "profile.json").write_text(prof.model_dump_json(indent=2))
        adapted.lexis.save(d / "lexis.joblib")
        with self._lock:
            self._profiles[prof.id] = adapted
        return prof

    def delete_profile(self, pid: str) -> bool:
        d = self.profile_dir / pid
        if not d.exists():
            return False
        for f in d.iterdir():
            f.unlink()
        d.rmdir()
        self._profiles.pop(pid, None)
        return True

    def assessor_for(self, profile_id: str | None) -> Assessor:
        if not profile_id:
            return self.assessor
        with self._lock:
            if profile_id in self._profiles:
                return self._profiles[profile_id]
        d = self.profile_dir / profile_id
        if not (d / "profile.json").exists():
            raise KeyError(profile_id)
        prof = Profile.model_validate_json((d / "profile.json").read_text())
        base = self.assessor
        bank = base.bank.extended([e.question for e in prof.examples],
                                  [int(Level.from_label(e.level)) for e in prof.examples], prof.name)
        adapted = Assessor(BloomClassifier.load(d / "lexis.joblib"), bank, self.settings, base.llm)
        with self._lock:
            self._profiles[profile_id] = adapted
        return adapted

    # -- jobs -----------------------------------------------------------------------------------------------
    def submit(self, kind: str, fn: Callable[[JobState], Any]) -> JobState:
        job = JobState(id=uuid.uuid4().hex[:12], kind=kind)
        self.jobs[job.id] = job
        self.cancels[job.id] = threading.Event()

        def run():
            job.status = "running"
            try:
                job.result = fn(job)
                job.status = "done"
            except Cancelled:
                job.status, job.error = "cancelled", "cancelled by the client"
            except Exception as exc:  # reported to the client, not swallowed
                log.exception("job %s failed", job.id)
                job.status, job.error = "failed", f"{type(exc).__name__}: {exc}"
                job.trace.append({"error": traceback.format_exc(limit=5)})
            finally:
                job.finished_at = datetime.now(timezone.utc)
                self.cancels.pop(job.id, None)
                self._persist(job)

        self._pool.submit(run)
        return job

    def cancel_job(self, jid: str) -> bool:
        ev = self.cancels.get(jid)
        if ev is not None:
            ev.set()
        return ev is not None

    def _persist(self, job: JobState) -> None:
        d = self.settings.artifacts / "jobs"
        d.mkdir(parents=True, exist_ok=True)
        (d / f"{job.id}.json").write_text(job.model_dump_json())

    def get_job(self, jid: str) -> JobState | None:
        if jid in self.jobs:
            return self.jobs[jid]
        path = self.settings.artifacts / "jobs" / f"{jid}.json"
        return JobState.model_validate_json(path.read_text()) if path.exists() else None

    def experiments(self) -> dict:
        path = self.settings.root / "results" / "summary.json"
        return json.loads(path.read_text()) if path.exists() else {}

    def experiment_tables(self) -> list[str]:
        d = self.settings.root / "results" / "tables"
        return sorted(f.stem for f in d.glob("*.csv")) + sorted(f.stem for f in d.glob("*.json"))

    def experiment_table(self, name: str):
        """One result table as JSON records (CSV) or as stored (JSON); None if unknown."""
        if name not in self.experiment_tables():
            return None
        d = self.settings.root / "results" / "tables"
        if (d / f"{name}.csv").exists():
            df = pd.read_csv(d / f"{name}.csv")
            return json.loads(df.to_json(orient="records"))
        return json.loads((d / f"{name}.json").read_text())

    def datacard(self) -> dict:
        path = self.settings.root / "data" / "processed" / "datacard.json"
        return json.loads(path.read_text()) if path.exists() else {}
