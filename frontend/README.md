# BloomBot web UI

A Next.js app built from the Claude Design project (`design/`). It talks to the BloomBot API, and
when the API is not running it replays recorded runs, so it can always be demonstrated.

## Run it

```bash
# 1. the API (and, for the LLM parts, `ollama serve`)
cd .. && source .venv/bin/activate && bloombot serve        # http://127.0.0.1:8000

# 2a. development, with hot reload
cd frontend && npm install && npm run dev                    # http://localhost:3000

# 2b. or build once and let the API serve the UI itself
cd frontend && npm run build                                 # writes out/
#     → open http://127.0.0.1:8000 (bloombot serve mounts frontend/out at /)
```

The header shows where the data comes from:

| Chip | Meaning |
|---|---|
| **Live API** (green) | every screen calls the API: parse, assess, classify, rebalance (streamed), profiles |
| **Replay · recorded** (amber) | no API found; screens use the recorded runs in `src/designData.js` |
| **Connecting…** | waiting for the API; the first call after a restart loads the models (~20 s) |

Click the chip to open the connection panel. To point at another host, open the page with
`?api=http://host:8000` (remembered in the browser); `NEXT_PUBLIC_BLOOMBOT_API` sets the default at
build time.

### The hosted demo and "Connect to local BloomBot"

https://bloombot.muralimadhav.com is built with `NEXT_PUBLIC_BLOOMBOT_API=replay` (Cloudflare Pages,
redeployed on every push to `main`). It replays recorded runs and never contacts the visitor's
computer on its own, so ordinary visitors get no browser prompt. To use your own papers there:

1. start Ollama (the app, or `ollama serve`) for the LLM second opinion; optional
2. `cd ~/bloombot && source .venv/bin/activate && bloombot serve` (it says whether Ollama answered)
3. click the **Replay · recorded** chip, then **Connect to local BloomBot**; Chrome asks once to let
   the site reach your computer: Allow

The page then probes `http://127.0.0.1:8000/v1/meta`, switches to Live and remembers the server in
`localStorage` (as `?api=` does). On later visits it tries that server quietly; if it is down, the
page replays recorded runs with a small "Local BloomBot not running · Reconnect" notice. The panel
explains a failed connection: nothing listening, Chrome's local-network permission denied (how to
allow it), or Safari (which blocks an https page from calling http://127.0.0.1: use Chrome or Edge,
or open http://127.0.0.1:8000 directly). **Disconnect (back to replay)** forgets the server.

The LLM chip turns the second opinion off ("fast mode" sends `use_llm: false`). If Ollama is not
running, it says so and the API uses its template rewriter.

## What is live

| Screen | Live API calls |
|---|---|
| Check a paper | `POST /v1/papers/parse` (file) or `/parse-text` (paste, sample); edits and 🔒 locks are sent back; `POST /v1/papers/assess` with the chosen blueprint and profile |
| Rebalance | `POST /v1/papers/plan`, `POST /v1/papers/rebalance`, then `GET /v1/jobs/{id}` every 0.7 s: each agent step appears as it happens and rewrite cards land when they pass the Verifier |
| Classify | `POST /v1/classify` + `POST /v1/analyze` for any typed question; the five example chips show recorded results instantly |
| College profile | labels collected with keys 1–6; "Use in assessments" calls `POST /v1/profiles` (≥ 5 different questions) and sends `profile_id` with every later call |
| How it works, Evaluation, Data | static: the verified numbers from `docs/RESULTS.md` and `docs/DATA.md` |

Accept / Edit / Reject decisions and the exports (paper `.md`, report `.md` / `.json`) happen in
the browser; only accepted and edited rewrites are written into the exported paper.

## How the code is organised

```
design/                 the Claude Design files (BloomBot.dc.html + components + bloombot-data.js)
tools/dc2jsx.py         converts a .dc.html template into a React view, keeping every style verbatim
tools/sync_design.sh    npm run design: regenerates src/views, src/components and src/designData.js
src/views/AppView.jsx   GENERATED markup of every screen
src/components/         GENERATED DistBar, LevelLadder, RewriteCard (design logic included)
src/designData.js       GENERATED static content and recorded runs
src/App.jsx             the screen logic: the design's logic class plus the API calls
src/adapt.js            API responses → the shapes the screens read
src/api.js              fetch wrapper and API discovery
src/app/                Next.js entry (layout, page, global CSS from the design's <helmet>)
```

To take a new version of the design: replace the files in `design/`, re-apply the few live-data
bindings in `BloomBot.dc.html` (`git diff design/` shows them: the API chip, error banner, paper
title, verdict, plan reason and before/after icons), then `npm run design`. Don't edit the
generated files by hand.

Contract: [`../docs/API.md`](../docs/API.md). Demo script: [`../docs/DEMO.md`](../docs/DEMO.md).
Real API responses for every screen: [`fixtures/`](fixtures) (`python scripts/make_fixtures.py`).
