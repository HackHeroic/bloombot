# Frontend (to be built)

Design brief / Claude Design prompt: [`../docs/FRONTEND_BRIEF.md`](../docs/FRONTEND_BRIEF.md).
Demo script and architecture: [`../docs/DEMO.md`](../docs/DEMO.md).

The backend is complete and stable; this folder is reserved for the web UI.

* Contract: [`../docs/API.md`](../docs/API.md) and [`../docs/openapi.json`](../docs/openapi.json)
  (generate a typed client from it, e.g. `npx openapi-typescript ../docs/openapi.json -o src/api.ts`).
* Run the API locally: `bloombot serve` → `http://127.0.0.1:8000` (CORS is open for development).

Suggested screens, each backed by existing endpoints:

| Screen | Endpoints |
|---|---|
| Upload & edit paper (fix parsed text, marks, lock questions) | `POST /v1/papers/parse`, `/parse-text` |
| Paper report: level distribution vs blueprint, per-question level, confidence, ⚑ review, printed tag vs predicted | `POST /v1/papers/assess`, `GET /v1/blueprints` |
| Rebalance: proposed rewrites with their checks, accept/edit/reject | `POST /v1/papers/rebalance`, `GET /v1/jobs/{id}`, `POST /v1/rewrite` |
| Single question: level, evidence (directive verbs, similar labelled questions) | `POST /v1/classify`, `POST /v1/analyze` |
| Institution profile: label 25-100 own questions to adapt the model | `GET/POST/DELETE /v1/profiles` |
| Results dashboard / How it works | `GET /v1/experiments`, `GET /v1/experiments/tables[/{name}]` (loso, adapt, concept_shift, arbiter, llm, rewrite, …) |
