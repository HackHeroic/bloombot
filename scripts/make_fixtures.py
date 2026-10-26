"""Write real API responses to frontend/fixtures/ so the UI can be built and demoed offline.

Run with the trained model (and Ollama, for the LLM votes and rewrites):
    python scripts/make_fixtures.py
"""
import json
import time
from pathlib import Path

from fastapi.testclient import TestClient

from bloombot.api.app import create_app

OUT = Path(__file__).resolve().parents[1] / "frontend" / "fixtures"
EXAMPLES = [
    "Explain how engineers evaluate the safety of an old bridge.",
    "Evaluate whether a microkernel is a better choice than a monolithic kernel for a pacemaker.",
    "Evaluate the integral of 2x from 0 to 3.",
    "What would happen if the page size were doubled?",
    "Design a database schema for a hospital appointment system.",
]


def save(name, obj):
    (OUT / f"{name}.json").write_text(json.dumps(obj, indent=1, ensure_ascii=False) + "\n")
    print("wrote", name)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    with TestClient(create_app()) as c:
        dump(c)


def dump(c):
    save("meta", c.get("/v1/meta").json())
    save("classify", c.post("/v1/classify", json={"questions": EXAMPLES}).json())
    save("analyze", [c.post("/v1/analyze", json={"question": q}).json() for q in EXAMPLES])
    save("blueprints", c.get("/v1/blueprints").json())
    paper_txt = Path(__file__).resolve().parents[1] / "data" / "papers" / "sample_os_paper.txt"
    with paper_txt.open("rb") as f:
        paper = c.post("/v1/papers/parse", files={"file": (paper_txt.name, f, "text/plain")}).json()
    save("paper_parsed", paper)
    save("paper_assessed", c.post("/v1/papers/assess", json={"paper": paper, "blueprint_id": "balanced-ug"}).json())
    job = c.post("/v1/papers/rebalance", json={"paper": paper, "blueprint_id": "balanced-ug"}).json()
    while (state := c.get(f"/v1/jobs/{job['id']}").json())["status"] not in ("done", "failed"):
        time.sleep(2)
    save("rebalance_job", state)
    save("experiments", c.get("/v1/experiments").json())
    save("experiment_tables", c.get("/v1/experiments/tables").json())


if __name__ == "__main__":
    main()
