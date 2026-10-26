"""Deploy BloomBot (API + web UI) to a Hugging Face Docker Space.

Assembles a staging folder (code, configs, sample papers, the trained model, the web UI built to
call its own origin, the public part of the dataset, and a licence notice) and uploads it. The
Space builds deploy/huggingface/Dockerfile and runs in fast mode with nothing stored.

    hf auth login                                   # once, in your own terminal
    python scripts/deploy_space.py                  # → https://huggingface.co/spaces/Madhav-Mit2026/bloombot
    python scripts/deploy_space.py --dry-run        # stage only, print the folder

Never shipped: artifacts/jobs, artifacts/profiles and the LLM cache (they hold papers and labels),
and the non-public sources' questions (the Indian school-board test set is counts only).
"""
import argparse
import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import pandas as pd

from bloombot.data.explore import licence_url
from bloombot.data.sources import SOURCES

ROOT = Path(__file__).resolve().parents[1]
DEPLOY = ROOT / "deploy" / "huggingface"
IGNORE = shutil.ignore_patterns("__pycache__", "*.pyc", ".DS_Store")


def build_ui(dest: Path) -> None:
    """Build the UI for the Space into dest without disturbing frontend/out (a local server may be using it)."""
    fe = ROOT / "frontend"
    backup = Path(tempfile.mkdtemp()) / "out"
    if (fe / "out").exists():
        shutil.copytree(fe / "out", backup)
    try:
        env = {**os.environ, "NEXT_PUBLIC_BLOOMBOT_API": "same-origin", "NEXT_PUBLIC_BLOOMBOT_HOSTED": "1"}
        subprocess.run(["npm", "run", "build"], cwd=fe, env=env, check=True, stdout=subprocess.DEVNULL)
        shutil.copytree(fe / "out", dest)
    finally:
        if backup.exists():
            shutil.rmtree(fe / "out", ignore_errors=True)
            shutil.copytree(backup, fe / "out")


def notice() -> str:
    lines = ["# Data notice", "",
             "The trained model's example bank (`artifacts/models/bank.parquet`) and the dataset explorer "
             "(`data/processed/questions.parquet`) contain questions from these sources, each under its own licence. "
             "Labels were converted to the six revised Bloom levels and the text was cleaned "
             "(see https://github.com/HackHeroic/bloombot/blob/main/docs/DATA.md).", "",
             "| Source | Citation | Licence | Shipped here |", "|---|---|---|---|"]
    for s in SOURCES:
        shipped = "questions" if s.public and s.role == "train" else "counts only"
        url = licence_url(s.licence)
        lic = f"[{s.licence}]({url})" if url else s.licence
        lines.append(f"| {s.title} | {s.citation} | {lic} | {shipped} |")
    lines += ["", "Full licence texts: [LICENSES/MIT.txt](LICENSES/MIT.txt), "
              "[LICENSES/Apache-2.0.txt](LICENSES/Apache-2.0.txt); CC BY 4.0: "
              "https://creativecommons.org/licenses/by/4.0/legalcode", ""]
    return "\n".join(lines)


def stage(dest: Path) -> None:
    for f in ("pyproject.toml",):
        shutil.copy2(ROOT / f, dest / f)
    for d in ("src", "configs", "data/papers", "artifacts/models", "results/tables"):
        shutil.copytree(ROOT / d, dest / d, ignore=IGNORE)
    shutil.copy2(ROOT / "results" / "summary.json", dest / "results" / "summary.json")
    proc = dest / "data" / "processed"
    proc.mkdir(parents=True)
    shutil.copy2(ROOT / "data" / "processed" / "datacard.json", proc / "datacard.json")
    public = [s.id for s in SOURCES if s.public]
    df = pd.read_parquet(ROOT / "data" / "processed" / "questions.parquet")
    df[df["source"].isin(public)].to_parquet(proc / "questions.parquet", index=False)
    build_ui(dest / "frontend" / "out")
    for f in DEPLOY.iterdir():  # Dockerfile, the Space card (README.md), .dockerignore, LICENSES/
        (shutil.copytree if f.is_dir() else shutil.copy2)(f, dest / f.name)
    (dest / "NOTICE.md").write_text(notice())
    assert not (dest / "artifacts" / "jobs").exists() and not (dest / "artifacts" / "profiles").exists()
    assert set(pd.read_parquet(proc / "questions.parquet")["source"]) <= set(public)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default="Madhav-Mit2026/bloombot")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    dest = Path(tempfile.mkdtemp(prefix="bloombot-space-"))
    stage(dest)
    size = sum(f.stat().st_size for f in dest.rglob("*") if f.is_file())
    print(f"staged {dest} ({size / 1e6:.1f} MB)")
    if args.dry_run:
        return
    from huggingface_hub import HfApi

    api = HfApi()
    api.create_repo(args.repo, repo_type="space", space_sdk="docker", exist_ok=True)
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    api.upload_folder(repo_id=args.repo, repo_type="space", folder_path=dest,
                      commit_message=f"Deploy bloombot {commit}")
    print(f"uploaded: https://huggingface.co/spaces/{args.repo}  (app: https://{args.repo.replace('/', '-').lower()}.hf.space)")
    print(json.dumps({"runtime": str(api.get_space_runtime(args.repo).stage)}))


if __name__ == "__main__":
    main()
