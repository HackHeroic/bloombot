"""Command line: ``bloombot <group> <command>``. Every experiment in the report is one command."""

from __future__ import annotations

import json
import logging
import warnings
from pathlib import Path
from typing import Optional

import typer

app = typer.Typer(add_completion=False, help="BloomBot: Bloom-level classification and paper rebalancing.")
data_app = typer.Typer(help="Download and build the datasets.")
app.add_typer(data_app, name="data")


def _setup(verbose: bool = False) -> None:
    warnings.filterwarnings("ignore")
    logging.basicConfig(level=logging.INFO if verbose else logging.WARNING, format="%(levelname)s %(message)s")


@data_app.command("build")
def data_build(seed_dir: Optional[Path] = typer.Option(None, help="folder with already-downloaded files"),
               verbose: bool = True) -> None:
    """Download all sources, clean, de-duplicate and write data/processed/."""
    _setup(verbose)
    from bloombot.config import get_settings
    from bloombot.data.build import build

    df = build(get_settings().root, seed_dir)
    typer.echo(df.groupby(["source", "label"]).size().unstack(fill_value=0).to_string())
    typer.echo(f"total: {len(df)}")


@data_app.command("card")
def data_card() -> None:
    """Print the data card (sources, licences, counts, filtering)."""
    from bloombot.config import get_settings

    typer.echo((get_settings().root / "data" / "processed" / "datacard.json").read_text())


@app.command()
def train(model: str = typer.Option(None, help="embedlex (default) or bloomlex")) -> None:
    """Train the production classifier (Lexis) on all training sources and save it to artifacts/."""
    _setup()
    from bloombot.config import get_settings
    from bloombot.pipeline.assess import train_lexis

    s = get_settings()
    if model:
        s = s.model_copy(update={"lexis_model": model})
    lexis, bank = train_lexis(s)
    typer.echo(f"saved {lexis.name} trained on {len(bank.questions)} questions to {s.artifacts / 'models'}")


@app.command()
def classify(questions: list[str], llm: bool = typer.Option(False, help="ask the LLM when unsure")) -> None:
    """Classify one or more questions."""
    _setup()
    from bloombot.pipeline.assess import Assessor

    a = Assessor.load(use_llm=llm)
    for p in a.assess(questions, use_llm=llm):
        flag = "  [review]" if p.needs_review else ""
        verbs = ", ".join(d.verb for d in p.directives) or p.frame or "-"
        typer.echo(f"{p.level:<10} {p.confidence:.2f}{flag}  ({verbs})  {p.question}")


@app.command()
def samples() -> None:
    """List the sample papers (use an id in place of a path: bloombot paper os)."""
    from bloombot.pipeline.samples import list_samples

    for s in list_samples():
        typer.echo(f"{s.id:<8} {s.description}\n         {s.file}")


@app.command()
def paper(path: Path, blueprint: str = typer.Option("aicte-2018-example"), rebalance: bool = False,
          llm: bool = typer.Option(False), out: Optional[Path] = None,
          duration: Optional[float] = typer.Option(None, help="exam duration in minutes (if not printed)"),
          pace: str = typer.Option("typical", help="students' pace: faster, typical or slower")) -> None:
    """Assess a question paper (PDF/DOCX/TXT/CSV, or a sample id from `bloombot samples`) against a
    blueprint; optionally rebalance it."""
    _setup()
    from bloombot.pipeline.blueprint import load_blueprints
    from bloombot.pipeline.conductor import Conductor
    from bloombot.pipeline.intake import parse_file
    from bloombot.pipeline.report import render_markdown

    bp = load_blueprints()[blueprint]
    if not path.exists():
        from bloombot.pipeline.samples import sample_path

        path = sample_path(str(path)) or path
    p = parse_file(path.read_bytes(), path.name)
    if duration:
        p.duration_minutes = duration
    c = Conductor.load(use_llm=llm)
    if rebalance:
        res = c.rebalance(p, bp, use_llm=llm, pace=pace)
        md = render_markdown(res.before, bp, res)
        payload = res.model_dump(mode="json")
    else:
        rep = c.assess_paper(p, bp, use_llm=llm, pace=pace)
        md = render_markdown(rep, bp)
        payload = rep.model_dump(mode="json")
    typer.echo(md)
    if out:
        out.mkdir(parents=True, exist_ok=True)
        (out / "report.md").write_text(md)
        (out / "report.json").write_text(json.dumps(payload, indent=2))
        (out / "audit_trail.json").write_text(json.dumps([t.model_dump(mode="json") for t in c.trace], indent=2))
        typer.echo(f"written to {out}")


@app.command("eval")
def evaluate(protocols: str = typer.Option("indomain,transfer,loso,heldout,adapt,probes"),
             models: str = typer.Option("rule,tfidf_svm,pos_tfidf_svm,bloomlex,embed_lr,embedlex"),
             llm_models: str = typer.Option("", help="comma-separated LLM specs, e.g. 'zero,rag,rag-cot'"),
             llm_per_source: int = typer.Option(120, help="questions per source for LLM runs"),
             out: Path = typer.Option(Path("results"))) -> None:
    """Run the experiments and write results/ (tables, predictions, summary.json)."""
    _setup()
    from bloombot.eval.runner import run_all

    run_all([p.strip() for p in protocols.split(",") if p.strip()],
            [m.strip() for m in models.split(",") if m.strip()],
            [m.strip() for m in llm_models.split(",") if m.strip()], llm_per_source, out)


@app.command()
def results(out: Path = typer.Option(Path("docs/figures"))) -> None:
    """Regenerate the figures from results/."""
    _setup()
    from bloombot.config import get_settings
    from bloombot.eval.figures import make_all

    root = get_settings().root
    make_all(root, root / "results", root / out)
    typer.echo(f"figures written to {out}")


@app.command()
def serve(host: str = "127.0.0.1", port: int = 8000, reload: bool = False) -> None:
    """Start the HTTP API (docs at /docs)."""
    import os

    import uvicorn

    from bloombot.config import get_settings
    from bloombot.models.llm import llm_client

    s = get_settings()
    if s.llm and s.llm_policy != "never":
        if llm_client(s.llm, s.ollama_host).available():
            typer.echo(f"LLM: {s.llm}" + ("" if s.llm.startswith("claude-") else f" at {s.ollama_host}"))
        else:
            typer.echo(f"Ollama not running (or {s.llm} not pulled) at {s.ollama_host}: start the Ollama app or run "
                       f"`ollama serve`; continuing without the LLM. The page's LLM chip shows when it is back.")
    os.environ.setdefault("BLOOMBOT_WARMUP", "1")
    uvicorn.run("bloombot.api.app:app", host=host, port=port, reload=reload)


if __name__ == "__main__":
    app()
