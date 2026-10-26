"""Small open LLMs (via a local Ollama server) as Bloom classifiers and as text generators.

Design choices:

* **Constrained output.** Ollama's ``format`` takes a JSON schema, so the model can only answer
  ``{"reasoning": str, "level": 1..6}``; there is nothing to parse or repair.
* **Probabilities from token log-probs, not self-reports.** The level is a single digit, so the
  log-probabilities of the alternatives 1-6 at that position give a distribution. An LLM's
  verbal confidence is known to be poorly calibrated; this is the model's actual belief.
* **Reasoning before the answer** when ``cot=True`` (chain-of-thought, Wei et al. 2022): the
  schema puts ``reasoning`` first, so it is generated before the level.
* **Retrieved few-shot examples** (RAG, Lewis et al. 2020): the k most similar labelled
  questions from the training sources - and, when adapting to an institution, from that
  institution's own labelled questions first.

The client speaks plain HTTP, so any Ollama-compatible server works. No GPU is required; a
3-4B model at 4-bit runs on a laptop CPU/Apple silicon.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import httpx
import numpy as np

from bloombot.models.base import BloomClassifier
from bloombot.taxonomy import DESCRIPTIONS, LABELS, LEVELS, N_LEVELS, Level

DEFAULT_LLM = os.environ.get("BLOOMBOT_LLM", "qwen3:4b-instruct-2507-q4_K_M")
DEFAULT_HOST = os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434")


class LLMError(RuntimeError):
    pass


@dataclass
class Completion:
    content: str
    data: Any
    logprobs: list[dict] | None
    eval_count: int = 0


class OllamaClient:
    def __init__(self, model: str = DEFAULT_LLM, host: str = DEFAULT_HOST, timeout: float = 180.0):
        self.model, self.host = model, host.rstrip("/")
        self._http = httpx.Client(timeout=timeout)

    def available(self) -> bool:
        try:
            tags = self._http.get(f"{self.host}/api/tags", timeout=3).json()
        except (httpx.HTTPError, ValueError):
            return False
        return any(m.get("name") == self.model or m.get("model") == self.model for m in tags.get("models", []))

    def chat(self, messages: list[dict], schema: dict | None = None, temperature: float = 0.0,
             max_tokens: int = 512, logprobs: bool = False, top_logprobs: int = 10, seed: int = 0) -> Completion:
        body: dict[str, Any] = {
            "model": self.model, "messages": messages, "stream": False,
            "options": {"temperature": temperature, "num_predict": max_tokens, "seed": seed},
        }
        if schema is not None:
            body["format"] = schema
        if logprobs:
            body["logprobs"] = True
            body["top_logprobs"] = top_logprobs
        try:
            r = self._http.post(f"{self.host}/api/chat", json=body)
            r.raise_for_status()
        except httpx.HTTPError as exc:
            raise LLMError(f"Ollama request failed ({self.model} at {self.host}): {exc}") from exc
        out = r.json()
        content = out.get("message", {}).get("content", "")
        data = None
        if schema is not None:
            try:
                data = json.loads(content)
            except json.JSONDecodeError:
                data = None
        return Completion(content, data, out.get("logprobs"), out.get("eval_count", 0))


# --- classification --------------------------------------------------------------------------
SYSTEM_PROMPT = """You are an expert in educational assessment. Classify an exam question by the \
cognitive process it requires, using the revised Bloom's taxonomy (Anderson & Krathwohl, 2001):

{levels}

Rules:
- Judge what the student must DO to answer, not which words appear. A Bloom verb inside a \
clause ("Explain how you would design ...") does not set the level; the main instruction does.
- If a question has several tasks, use the highest level it really demands.
- Questions answerable by recalling a fact, definition or list are Remember even if phrased \
with "explain" or "describe"; explaining a concept in one's own words is Understand.
- Using a known procedure or formula on a given case is Apply; producing something new is Create.
Answer with the level number (1-6)."""


def _levels_block() -> str:
    return "\n".join(f"{int(lv)}. {lv.label}: {DESCRIPTIONS[lv]}" for lv in LEVELS)


def classification_schema(cot: bool) -> dict:
    props: dict[str, Any] = {}
    if cot:
        props["reasoning"] = {"type": "string", "description": "one or two sentences: what must the student do?"}
    props["level"] = {"type": "integer", "enum": list(range(1, N_LEVELS + 1))}
    return {"type": "object", "properties": props, "required": list(props)}


_DIGIT = re.compile(r"^\s*[1-6]\s*$")
_LEVEL_TOKENS = frozenset(str(i) for i in range(1, N_LEVELS + 1))


_LEVEL_KEY = re.compile(r'"level"\s*:\s*$')


def max_tokens(cot: bool) -> int:
    """Answer budget. Reasoning runs to ~250 tokens; a budget of 200 cut off 9% of answers,
    which then fell back to a uniform distribution."""
    return 600 if cot else 20


def distribution_from_logprobs(logprobs: list[dict] | None, chosen: int | None, smoothing: float = 0.15) -> np.ndarray:
    """Read P(level) from the top log-probs at the token that is the value of ``"level"``.

    Scans from the end, and requires the text just before the digit to end with ``"level":``,
    so a digit inside the reasoning ("... a level 4 task ...") is never used. Falls back to a
    smoothed one-hot when log-probs are unavailable."""
    p = np.zeros(N_LEVELS)
    if logprobs:
        for i in range(len(logprobs) - 1, -1, -1):
            tok = logprobs[i]
            if not _DIGIT.match(tok.get("token", "")):
                continue
            prefix = "".join(t.get("token", "") for t in logprobs[max(0, i - 8):i])
            if not _LEVEL_KEY.search(prefix):
                continue
            for alt in tok.get("top_logprobs", []) or [tok]:
                s = alt.get("token", "").strip()
                if s in _LEVEL_TOKENS:  # ASCII only: "\u2082".isdigit() is True but int() fails
                    p[int(s) - 1] += math.exp(alt.get("logprob", -50))
            if p.sum() > 0:
                return p / p.sum()
            break
    if chosen is None:
        return np.full(N_LEVELS, 1.0 / N_LEVELS)
    p[:] = smoothing / (N_LEVELS - 1)
    p[chosen - 1] = 1 - smoothing
    return p


@dataclass
class Example:
    question: str
    level: int


class ExampleRetriever:
    """Nearest labelled questions by sentence-embedding cosine similarity."""

    def __init__(self, questions: Sequence[str], levels: Sequence[int], encoder: str | None = None,
                 priority: Sequence[bool] | None = None):
        from bloombot.models.encoder import DEFAULT_ENCODER, embedding_cache

        self.cache = embedding_cache(encoder or DEFAULT_ENCODER)
        self.questions, self.levels = list(questions), np.asarray(levels, int)
        self.priority = np.asarray(priority if priority is not None else [False] * len(self.questions))
        self.E = self.cache.encode(self.questions) if self.questions else np.zeros((0, 384), np.float32)

    def __call__(self, query: str, k: int = 6, exclude: str | None = None) -> list[Example]:
        if not self.questions:
            return []
        sims = self.E @ self.cache.encode([query])[0]
        sims = sims + 0.25 * self.priority  # an institution's own examples come first
        order = np.argsort(-sims)
        out = []
        for i in order:
            if exclude is not None and self.questions[i] == exclude:
                continue
            out.append(Example(self.questions[i], int(self.levels[i])))
            if len(out) == k:
                break
        return out[::-1]  # most similar last, nearest the query


_CACHE_LOCK = threading.Lock()
_CACHE: dict[str, dict] | None = None


def _cache_path() -> Path:
    return Path(os.environ.get("BLOOMBOT_LLM_CACHE", Path(__file__).resolve().parents[3] / "artifacts" / "llm_cache.jsonl"))


def _cache_get(key: str) -> dict | None:
    global _CACHE
    with _CACHE_LOCK:
        if _CACHE is None:
            _CACHE = {}
            path = _cache_path()
            if path.exists():
                for line in path.read_text().splitlines():
                    try:
                        rec = json.loads(line)
                        _CACHE[rec["key"]] = rec["value"]
                    except (ValueError, KeyError):
                        continue
        return _CACHE.get(key)


def _cache_put(key: str, value: dict) -> None:
    with _CACHE_LOCK:
        if _CACHE is not None:
            _CACHE[key] = value
        path = _cache_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a") as fh:
            fh.write(json.dumps({"key": key, "value": value}) + "\n")


class LLMClassifier(BloomClassifier):
    """Prompted open LLM. ``shots=0`` is zero-shot; ``shots>0`` uses retrieved examples."""

    def __init__(self, model: str = DEFAULT_LLM, shots: int = 0, cot: bool = False, workers: int = 4,
                 host: str = DEFAULT_HOST, use_logprobs: bool = True):
        self.model, self.shots, self.cot, self.workers = model, shots, cot, workers
        self.host, self.use_logprobs = host, use_logprobs
        self.trainable = shots > 0
        self.retriever: ExampleRetriever | None = None
        self.name = f"llm[{model.split(':')[0]}{':' + model.split(':')[1].split('-')[0] if ':' in model else ''}]" \
                    f"-{shots}shot{'-cot' if cot else ''}"
        self.last_outputs: list[dict] = []

    @property
    def client(self) -> OllamaClient:
        if getattr(self, "_client", None) is None:
            self._client = OllamaClient(self.model, self.host)
        return self._client

    def fit(self, texts, y, sample_weight=None):
        if self.shots > 0:
            prio = None if sample_weight is None else np.asarray(sample_weight) > np.median(sample_weight) * 2
            self.retriever = ExampleRetriever(list(texts), list(y), priority=prio)
        return self

    def messages(self, question: str) -> list[dict]:
        msgs = [{"role": "system", "content": SYSTEM_PROMPT.format(levels=_levels_block())}]
        if self.retriever is not None and self.shots > 0:
            for ex in self.retriever(question, self.shots, exclude=question):
                msgs.append({"role": "user", "content": f"Question: {ex.question}"})
                msgs.append({"role": "assistant", "content": json.dumps(
                    {**({"reasoning": "(example)"} if self.cot else {}), "level": ex.level})})
        msgs.append({"role": "user", "content": f"Question: {question}"})
        return msgs

    def classify_one(self, question: str) -> dict:
        msgs = self.messages(question)
        key = hashlib.sha1(json.dumps([self.model, self.cot, self.use_logprobs, msgs]).encode()).hexdigest()
        hit = _cache_get(key)
        if hit is not None and hit.get("level") is not None:
            return hit
        comp = self.client.chat(msgs, schema=classification_schema(self.cot),
                                max_tokens=max_tokens(self.cot), logprobs=self.use_logprobs)
        level = None
        if isinstance(comp.data, dict):
            try:
                level = int(comp.data.get("level"))
            except (TypeError, ValueError):
                level = None
            if level is not None and not 1 <= level <= N_LEVELS:
                level = None
        proba = distribution_from_logprobs(comp.logprobs, level)
        out = {"question": question, "level": level, "proba": proba.tolist(),
               "reasoning": (comp.data or {}).get("reasoning") if isinstance(comp.data, dict) else None}
        if level is not None:  # a failed answer (e.g. cut off) is retried next time, not cached
            _cache_put(key, out)
        return out

    def classify_many(self, texts, on_result=None) -> list[dict]:
        """Thread-safe: returns the outputs instead of storing them on the instance.
        ``on_result(i, out)`` is called as each answer arrives (for live progress)."""
        texts = list(texts)
        outs: list[dict] = [{} for _ in texts]
        with ThreadPoolExecutor(max_workers=self.workers) as pool:
            futures = {pool.submit(self.classify_one, t): i for i, t in enumerate(texts)}
            for f in as_completed(futures):
                i = futures[f]
                outs[i] = f.result()
                if on_result is not None:
                    on_result(i, outs[i])
        return outs

    def predict_proba(self, texts):
        outs = self.classify_many(texts)
        self.last_outputs = outs  # convenience for single-threaded experiment code only
        return np.array([o["proba"] for o in outs])

    def __getstate__(self):
        state = dict(self.__dict__)
        state.pop("_client", None)
        return state


def level_names() -> str:
    return ", ".join(f"{i + 1}={lab}" for i, lab in enumerate(LABELS))


__all__ = ["OllamaClient", "LLMClassifier", "LLMError", "ExampleRetriever", "Level", "level_names"]
