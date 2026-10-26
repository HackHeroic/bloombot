# HTTP API

Start it with `bloombot serve` (default `http://127.0.0.1:8000`). Interactive docs are at
`/docs`; the machine-readable spec is [`openapi.json`](openapi.json) (regenerate with
`python -c "import json; from bloombot.api.app import create_app; json.dump(create_app().openapi(), open('docs/openapi.json','w'), indent=1)"`).
All bodies are JSON and all shapes are defined in `src/bloombot/pipeline/schemas.py`.
Level names are always one of `Remember, Understand, Apply, Analyze, Evaluate, Create`.

## Endpoints

| Method | Path | Purpose |
|---|---|---|
| GET | `/health` | liveness |
| GET | `/v1/meta` | levels with descriptions, model names, LLM availability, data summary, thresholds |
| GET | `/v1/blueprints` | blueprint presets (`aicte-2018-example`, `balanced-ug`) |
| POST | `/v1/analyze` | linguistic analysis of one question (directives, frame, masked form, prior) |
| POST | `/v1/classify` | classify up to 500 questions |
| POST | `/v1/papers/parse` | multipart upload (PDF, DOCX, TXT, CSV) → `Paper` |
| POST | `/v1/papers/parse-text` | pasted text → `Paper` |
| POST | `/v1/papers/assess` | `Paper` (+ blueprint) → `PaperReport` |
| POST | `/v1/papers/plan` | `PaperReport` + blueprint → list of `RewriteJob` |
| POST | `/v1/rewrite` | rewrite one question to a target level, with checks (synchronous) |
| POST | `/v1/papers/rebalance` | full pipeline as a background job → `202 JobState` |
| GET | `/v1/jobs/{id}` | job status; `result` is a `RebalanceResult` when `status == "done"` |
| GET/POST | `/v1/profiles` | list / create an institution profile (k-shot adaptation) |
| DELETE | `/v1/profiles/{id}` | delete a profile |
| GET | `/v1/experiments` | `results/summary.json` (for a results dashboard) |
| GET | `/v1/experiments/tables` | names of the result tables |
| GET | `/v1/experiments/tables/{name}` | one result table as JSON records (e.g. `loso`, `adapt`, `concept_shift`, `arbiter`, `llm`, `rewrite`) |

`use_llm`: `null` = server policy (the LLM is asked only when the fast model is unsure),
`false` = never, `true` = when unsure. `profile_id` makes any assessment use an institution's
adapted model.

## Typical frontend flow

```
upload file ──► POST /v1/papers/parse ──► show questions; teacher edits text/marks, locks questions
            ──► POST /v1/papers/assess {paper, blueprint_id} ──► distribution chart, bound checks,
                                                                   per-question level + confidence,
                                                                   ⚑ needs_review list
            ──► POST /v1/papers/rebalance {paper, blueprint_id} ──► {id}
            ──► poll GET /v1/jobs/{id} until done ──► before/after, each rewrite with its checks
```

## Examples

```bash
curl -s localhost:8000/v1/classify -H 'content-type: application/json' \
  -d '{"questions": ["Explain how you would design a database schema.", "Evaluate the integral of 2x from 0 to 3."]}'
```

```jsonc
{
  "predictions": [{
    "question": "Explain how you would design a database schema.",
    "level": "Understand", "level_id": 2, "confidence": 0.71,
    "proba": {"Remember": 0.08, "Understand": 0.71, "Apply": 0.06, "Analyze": 0.09, "Evaluate": 0.02, "Create": 0.04},
    "needs_review": false, "lower_order": true,
    "directives": [{"verb": "explain", "kind": "imperative", "phrase": null, "negated": false}],
    "frame": null, "evidence": "directive",
    "votes": [{"agent": "Lexis", "level": "Understand", "confidence": 0.69, "proba": [...]},
              {"agent": "Archivist", "level": "Understand", "confidence": 0.52, "proba": [...]}],
    "neighbours": [{"question": "...", "level": "Understand", "similarity": 0.83, "source": "gani2023"}]
  }, ...],
  "model": "embedlex", "llm_used": false
}
```

```bash
# upload a paper
curl -s -F file=@data/papers/sample_os_paper.txt localhost:8000/v1/papers/parse
# create an institution profile from >= 5 labelled questions (more is better; 25-100 recommended)
curl -s localhost:8000/v1/profiles -H 'content-type: application/json' -d '{"name": "XYZ CSE", "examples": [
  {"question": "Explain paging.", "level": "Remember"}, ...]}'
```

## Notes for the UI

* Show `confidence` and the ⚑ `needs_review` flag; across institutions the model is right on
  roughly 6 of 10 questions exactly and 8–9 of 10 within one level, so the teacher stays in
  the loop. A profile with 25–100 of the institution's own labelled questions raises accuracy.
* `tagged_level` is the level printed on the paper (if any); highlighting disagreements with
  the predicted level is useful.
* OR-alternatives share a `group`; the distribution counts each alternative with
  weight / number of alternatives.
* `RewriteResult.attempts[].checks` explains every accept/reject decision; show it.
