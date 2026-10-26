# BloomBot

**Classify exam questions by revised Bloom level — reliably on question sources the model
has never seen — and help teachers rebalance a question paper by rewriting questions to a
target level, with automatic checks.** Open models only; runs on a laptop CPU (no GPU).

## Why

Colleges following Outcome-Based Education tag every exam question with a Bloom level, and
AICTE's *Examination Reform Policy* (2018) recommends that institutions cap lower-order
questions (its example: no more than 40% weightage on knowledge-oriented questions). Tagging by
hand is slow and inconsistent between teachers.

Published classifiers look accurate (0.73–0.94 on random splits of one dataset), but trained
on one dataset and tested on another they lose about 0.25–0.3 F1 (Faraji et al., 2026). We
reproduce that drop on 9,855 questions from 7 sources and find its main cause: **institutions
label the same question form differently** ("Explain …" is Understand in Malaysian exams but
Remember in 55% of Indian engineering questions). BloomBot is built around that fact.

## What it does

| | |
|---|---|
| **Classify** | A spaCy dependency parse finds the *directive* (what the student must do), not just any Bloom verb; question frames handle verb-less questions; an ordinal classifier combines these linguistic features with a sentence embedding. When the fast model is unsure, a local open LLM (Qwen3-4B via Ollama) reasons about the question with similar labelled examples and its answer is mixed in. Returns a probability per level, the evidence, and similar labelled questions. |
| **Adapt** | A teacher labels 25–100 of their own questions to create an institution profile; accuracy rises by +0.03 to +0.10 weighted F1. |
| **Check a paper** | Upload PDF/DOCX/TXT/CSV; the parser reads sub-parts, OR-alternatives, marks, CO and printed BT tags; the report shows the level distribution by marks against a blueprint (e.g. AICTE's example cap). |
| **Rebalance** | Plans the fewest rewrites that satisfy the blueprint; the LLM rewrites each question to its target level; every rewrite is re-classified by an independent model, checked for form, topic and duplication, and (for higher levels) rejected if memory alone could answer it — with up to 3 retries, then the teacher. |

## Results at a glance

Leave-one-source-out (train on 5 sources, test on the 6th), mean over 6 sources; Indian
school-board questions are never used for training. Details: [docs/RESULTS.md](docs/RESULTS.md).

| Model | Same source | One other source | **All other sources** | QWK | Indian boards (unseen) |
|---|---:|---:|---:|---:|---:|
| TF-IDF + SVM | 0.732 | 0.413 | 0.589 | 0.577 | 0.599 |
| POS-weighted TF-IDF + SVM | 0.743 | 0.435 | 0.590 | 0.568 | 0.599 |
| Sentence embedding + LR | 0.638 | 0.359 | 0.502 | 0.472 | 0.593 |
| Lexicon rule (no training) | 0.624 | 0.624 | 0.624 | 0.613 | 0.551 |
| **EmbedLex (ours)** | **0.764** | **0.524** | **0.641** | **0.701** | **0.630** |

Weighted F1 unless noted. EmbedLex vs TF-IDF + SVM across sources: +0.051 [0.041, 0.060],
p < 0.001. With 100 labelled questions from the target institution: 0.627 → 0.694 on average,
Indian engineering 0.435 → 0.527, Indian boards 0.578 → 0.674.

**Is a small LLM better?** Not on its own. On the same 418 held-out questions:

* Qwen3-4B zero-shot scores 0.45, with retrieved examples 0.51, and with retrieved examples
  plus reasoning 0.62. EmbedLex scores 0.67 at ~1 ms per question.
* The LLM's probabilities are overconfident (ECE 0.39–0.50).
* With RAG it follows a wrong level suggested by a "senior teacher" 47% of the time.

As a *second opinion* the LLM does help. When the fast model is unsure, BloomBot asks it to
reason and then mixes its answer in at half weight: 0.640 → 0.682 (+0.042 [0.005, 0.078],
with the rule chosen without seeing the test source).

The first design pooled the votes log-linearly, and that cost 0.15. **Rewriting:** 37 of 40
real questions were rewritten two levels up and passed every automatic check (80% on the
first try, 4.5 s each). [§6–8 of RESULTS](docs/RESULTS.md#6-small-open-llm-qwen3-4b-instruct-4-bit-ollama-laptop).

## Quick start

```bash
git clone https://github.com/HackHeroic/bloombot.git && cd bloombot
python -m venv .venv && source .venv/bin/activate
pip install -e ".[neural,dev]"
python -m spacy download en_core_web_sm

bloombot data build          # downloads the 7 public sources (~3 MB) and builds data/processed/
bloombot train               # trains the classifier (~30 s on a laptop)
pytest                       # 75 tests, ~6 s, offline

bloombot classify "Explain how you would design a database schema." "Evaluate the integral of 2x from 0 to 3."
bloombot samples             # the sample papers in data/papers
bloombot paper table --blueprint balanced-ug --rebalance --out results/demo   # a sample id or a file path
bloombot serve               # HTTP API on http://127.0.0.1:8000 (docs at /docs)
```

Web UI (Next.js, in `frontend/`): `cd frontend && npm install && npm run build`, then
`bloombot serve` also serves it at http://127.0.0.1:8000. It streams rebalance jobs live and falls
back to recorded runs of every sample paper when the API is not running. Hosted at
https://bloombot.muralimadhav.com, where **Connect to local BloomBot** switches it to your own
server ([frontend/README.md](frontend/README.md)).

Optional LLM (classification of hard cases, rewriting, verification):

```bash
ollama pull qwen3:4b-instruct-2507-q4_K_M      # 2.5 GB; any Ollama model works via BLOOMBOT_LLM
```

Without it everything still runs: assessment uses the fast models and rewriting falls back to
transparent templates.

## Reproduce the experiments

```bash
bloombot eval                                              # CPU protocols, ~7 min
bloombot eval --protocols arbiter,rewrite --models none \
              --llm-models zero,rag,rag-cot --llm-per-source 60   # LLM experiments, ~1.5 h on a laptop
bloombot results                                           # figures in docs/figures/
```

Outputs: `results/tables/*.csv`, `results/predictions/*.parquet` (every prediction with
probabilities), `results/summary.json`.

## Layout

```
src/bloombot/
  taxonomy.py         levels; label parsing that refuses ambiguous numeric codes
  text/               normalisation, Bloom lexicon (Anderson & Krathwohl), spaCy analysis, delexicalisation
  data/               source registry (provenance, licence, label scheme) and the build
  models/             rule, TF-IDF+SVM, POS-TF-IDF+SVM, BloomLex, EmbedLR/EmbedLex, LLM (Ollama)
  eval/               metrics, protocols (in-domain, transfer, LOSO, held-out, adaptation), probes, runner, figures
  pipeline/           Scribe (intake), assessment (Lexis/Archivist/Socrates/Arbiter), Architect (blueprint),
                      Composer + verifiers (rewrite), Conductor, report, schemas
  api/                FastAPI app and services (profiles, background jobs)
  cli.py              `bloombot ...`
configs/blueprints/   AICTE-2018 example cap, balanced template
data/probes/          contrast set (58 items); data/papers/ sample papers
docs/                 DESIGN, DATA (data card), RESULTS, API, openapi.json, labelling guide, figures
frontend/             Next.js web UI built from the Claude Design project (live API + recorded replay)
```

## Documentation

* [docs/DESIGN.md](docs/DESIGN.md) — why it is built this way, NLP pipeline, models, evaluation, agents
* [docs/DATA.md](docs/DATA.md) — sources, provenance, licences, label conversion, cleaning
* [docs/RESULTS.md](docs/RESULTS.md) — all results, failure analysis, corrections to cited claims
* [docs/DEMO.md](docs/DEMO.md) — demonstration guide: architecture diagram, one question through the NLP, evaluation, live demo script, panel Q&A
* [docs/API.md](docs/API.md) — HTTP contract for the frontend
* [docs/FRONTEND_BRIEF.md](docs/FRONTEND_BRIEF.md) — design brief (Claude Design prompt) for the web UI
* [docs/labelling_guide.md](docs/labelling_guide.md) — for teachers building a profile or checking rewrites

## Limitations

* Exact-level accuracy on an unseen institution is about 6 in 10 (8–9 in 10 within one level);
  keep the teacher in the loop and use a profile.
* Evaluate/Create are rare in Indian exam data, so per-level results there rest on few items.
* The contrast set and rewrite checks are automatic; rewrites need human validation
  (`results/tables/rewrite.csv` has columns for two reviewers).
* The PDF parser handles text PDFs in common layouts; scanned papers need OCR first.

## References

Anderson & Krathwohl (2001), *A Taxonomy for Learning, Teaching, and Assessing* ·
AICTE (2018), *Examination Reform Policy* ·
Faraji, Molavi, Rasoulkhani, Tavakoli & Kismihók (2026), *Cross-Dataset Bloom Question Classification: Supervised Models and Prompted LLMs*, arXiv:2606.13684 ·
Kumar, Gulwani & Singh (2025), arXiv:2511.10903 ·
Mohammed & Omar (2020), PLoS ONE 15(3): e0230442 ·
Yahya, Toukal & Osman (2012), Studies in Computational Intelligence 431 ·
Scaria, Chenna & Subramani (2024), AIED 2024 ·
Sangodiah et al. (2017), JATIT 95(12) · Gani et al. (2023), Educ. Inf. Technol. 28(12) ·
Das et al., Mendeley Data 10.17632/48nfn95jc8 ·
Frank & Hall (2001), ordinal classification · Gardner et al. (2020), contrast sets ·
Ribeiro et al. (2020), CheckList · Lewis et al. (2020), RAG · Wei et al. (2022), chain-of-thought ·
Honnibal et al., spaCy · Xiao et al. (2023), BGE embeddings · Qwen Team (2025), Qwen3.

Code: MIT. Datasets are downloaded from their publishers and keep their own licences
(see [docs/DATA.md](docs/DATA.md)); the Indian board set is CC BY-NC and used only for evaluation.
