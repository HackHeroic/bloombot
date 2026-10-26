"""Record real runs of every sample paper for the web UI's offline (replay) mode.

For each sample in data/papers and each blueprint: the parsed paper, the assessment, the plan and
the full rebalance job (with its timed trace). Also writes the licence-safe dataset sample for the
Data page. Needs a running API (with Ollama, for the LLM votes and rewrites):

    bloombot serve &
    python scripts/record_samples.py            # writes frontend/src/recorded/*.json
"""
import json
import sys
import time
from pathlib import Path

import httpx

from bloombot.config import get_settings
from bloombot.data.explore import bundle

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "frontend" / "src" / "recorded"
API = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8000"
KEEP_NEIGHBOURS = 2  # the recorded reports only need a couple of examples per question


def slim(report: dict) -> dict:
    for it in report["items"]:
        p = it["prediction"]
        p["neighbours"] = p["neighbours"][:KEEP_NEIGHBOURS]
        p["explanation"] = []
    return report


def wait(c: httpx.Client, job_id: str) -> dict:
    while True:
        j = c.get(f"/v1/jobs/{job_id}").json()
        if j["status"] in ("done", "failed", "cancelled"):
            return j
        time.sleep(1)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    with httpx.Client(base_url=API, timeout=600) as c:
        meta = c.get("/v1/meta").json()
        blueprints = c.get("/v1/blueprints").json()
        samples = c.get("/v1/samples").json()
        runs = {}
        for s in samples:
            paper = c.get(f"/v1/samples/{s['id']}").json()
            run = {"paper": paper, "assess": {}, "plan": {}, "rebalance": {}}
            for bp in blueprints:
                body = {"paper": paper, "blueprint_id": bp["id"]}
                t0 = time.time()
                report = c.post("/v1/papers/assess", json=body).json()
                run["assess"][bp["id"]] = slim(report)
                run["plan"][bp["id"]] = c.post("/v1/papers/plan", json={"report": report, "blueprint_id": bp["id"]}).json()
                job = wait(c, c.post("/v1/papers/rebalance", json=body).json()["id"])
                if job["status"] != "done":
                    raise SystemExit(f"{s['id']} / {bp['id']}: rebalance {job['status']}: {job['error']}")
                for k in ("before", "after"):
                    slim(job["result"][k])
                run["rebalance"][bp["id"]] = job
                print(f"{s['id']:<6} {bp['id']:<20} {len(run['plan'][bp['id']])} rewrites, "
                      f"{len(job['trace'])} events, {time.time() - t0:.0f} s")
            runs[s["id"]] = run
    rec = {"recorded_with": {"llm": meta["llm"], "lexis": meta["lexis"], "at": time.strftime("%Y-%m-%d")},
           "samples": samples, "blueprints": blueprints, "runs": runs}
    (OUT / "samples.json").write_text(json.dumps(rec, ensure_ascii=False, separators=(",", ":")))
    data = bundle(get_settings().root / "data" / "processed")
    (OUT / "dataset.json").write_text(json.dumps(data, ensure_ascii=False, separators=(",", ":")))
    for f in ("samples.json", "dataset.json"):
        print(f"wrote {OUT / f} ({(OUT / f).stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
