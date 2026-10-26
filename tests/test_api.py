from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from bloombot.api.app import create_app
from bloombot.api.services import Services

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def client(assessor, settings):
    app = create_app()
    with TestClient(app) as c:
        svc = Services(settings.model_copy(update={"root": ROOT}))
        svc.__dict__["assessor"] = assessor  # pre-fill the cached_property: no model files needed
        app.state.services = svc
        yield c


def test_health_and_meta(client):
    assert client.get("/health").json()["status"] == "ok"
    meta = client.get("/v1/meta").json()
    assert len(meta["levels"]) == 6 and meta["llm"]["available"] is False


def test_classify(client):
    r = client.post("/v1/classify", json={"questions": ["Define a queue.", "Design a new scheduling algorithm."]})
    assert r.status_code == 200
    preds = r.json()["predictions"]
    assert [p["level"] for p in preds] == ["Remember", "Create"]
    assert abs(sum(preds[0]["proba"].values()) - 1) < 1e-3
    assert preds[0]["directives"][0]["verb"] == "define"


def test_analyze(client):
    r = client.post("/v1/analyze", json={"question": "Explain how you would design a schema."}).json()
    assert r["directives"][0]["lemma"] == "explain" and "design" in r["embedded_bloom_verbs"]


def test_paper_flow(client):
    text = (ROOT / "data" / "papers" / "sample_os_paper.txt").read_text()
    paper = client.post("/v1/papers/parse-text", json={"text": text}).json()
    assert len(paper["questions"]) == 11
    rep = client.post("/v1/papers/assess", json={"paper": paper, "blueprint_id": "aicte-2018-example"}).json()
    assert set(rep["distribution"]) == {"Remember", "Understand", "Apply", "Analyze", "Evaluate", "Create"}
    assert rep["checks"][0]["description"].startswith("Knowledge-oriented")
    jobs = client.post("/v1/papers/plan", json={"report": rep, "blueprint_id": "aicte-2018-example"}).json()
    assert isinstance(jobs, list)


def test_upload_and_rewrite(client):
    files = {"file": ("p.txt", b"Q1. Define a stack. [5]\nQ2. List the types of memory. [5]\n", "text/plain")}
    paper = client.post("/v1/papers/parse", files=files).json()
    assert [q["marks"] for q in paper["questions"]] == [5, 5]
    r = client.post("/v1/rewrite", json={"question": "Define a stack.", "target_level": "Apply"}).json()
    assert r["job"]["to_level"] == "Apply" and r["attempts"]


def test_unknown_blueprint_404(client):
    r = client.post("/v1/papers/assess", json={"paper": {"questions": [{"id": "1", "text": "Define x."}]},
                                               "blueprint_id": "nope"})
    assert r.status_code == 404


def test_experiment_tables(client):
    names = client.get("/v1/experiments/tables").json()
    assert client.get("/v1/experiments/tables/nope").status_code == 404
    if "loso" not in names:
        pytest.skip("results/tables not generated")
    rows = client.get("/v1/experiments/tables/loso").json()
    assert rows and {"model", "test", "weighted_f1"} <= set(rows[0])
    assert client.get("/v1/experiments/tables/../summary").status_code == 404
