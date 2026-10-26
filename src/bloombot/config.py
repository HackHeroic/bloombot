"""Runtime settings. Every field can be overridden with an environment variable ``BLOOMBOT_<NAME>``."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="BLOOMBOT_", env_file=".env", extra="ignore")

    root: Path = ROOT
    artifacts: Path = ROOT / "artifacts"

    # Lexis: the fast classifier. "embedlex" needs the sentence encoder; "bloomlex" is pure CPU sklearn.
    lexis_model: Literal["embedlex", "bloomlex"] = "embedlex"
    encoder: str = "BAAI/bge-small-en-v1.5"

    # Socrates / Composer / Student / Critic: a local Ollama model. Empty string disables the LLM.
    llm: str = "qwen3:4b-instruct-2507-q4_K_M"
    ollama_host: str = "http://127.0.0.1:11434"
    llm_policy: Literal["never", "uncertain", "always"] = "uncertain"
    llm_shots: int = 6
    llm_cot: bool = True  # reason first, then answer; the only LLM setting that helped (docs/RESULTS.md, Arbiter)

    # Arbiter: chosen by the Arbiter study (linear pooling; LLM only when unsure; the Archivist's
    # vote did not help, so its neighbours are shown as evidence but weigh nothing by default)
    llm_trigger_confidence: float = 0.60  # call the LLM when Lexis is less sure than this
    review_confidence: float = 0.50       # below this the teacher is asked to confirm
    pool_rule: Literal["linear", "geometric"] = "linear"
    weight_lexis: float = 1.0
    weight_llm: float = 0.5
    weight_neighbours: float = 0.0
    neighbours: int = 8

    # Rewriting
    max_attempts: int = 3
    level_check_min_proba: float = 0.40
    topic_min_overlap: float = 0.25
    duplicate_max_similarity: float = 0.92

    # Institution profiles (k-shot adaptation)
    profile_example_weight: float = 10.0


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
