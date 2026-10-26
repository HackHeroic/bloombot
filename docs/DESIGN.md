# Design

This document explains why BloomBot is built the way it is. For results see
[RESULTS.md](RESULTS.md); for the data see [DATA.md](DATA.md); for the HTTP contract see
[API.md](API.md).

## 1. What the problem really is

A Bloom level describes the **cognitive process a question demands** (Anderson & Krathwohl,
2001). Three facts shape everything else:

1. **The level is carried by the directive, not by any verb.** "Explain how you would *design*
   a schema" asks for an explanation (Understand); *design* is inside the complement. Bag-of-words
   models see both verbs equally, so they learn shortcuts that break on new data.
2. **Topic words do not transfer.** A model trained on data-science questions learns that
   "regression" goes with Apply; on economics papers that cue is useless or misleading.
   Sentence embeddings are worst at this: they mostly encode topic (weighted F1 0.50 across
   sources, the lowest of all models).
3. **Institutions label differently** (*concept shift*). In our data "Explain …" is
   Understand in 93% of Gani's Malaysian exam questions but **Remember in 55%** of the Indian
   engineering questions; "Design …" is Create almost everywhere but **Apply in 93%** of the
   Indian engineering questions ([figure](figures/fig_concept_shift.png)). No single global model can be right
   for both, so part of the cross-dataset drop is irreducible.

The design answers each fact: parse for the directive (1), delexicalise and add linguistic
features (2), and let an institution adapt the model with a few labelled questions (3).

## 2. No GPU needed

Questions are ~20 tokens and the data is ~10k items. Everything trains and runs on a laptop CPU:

| Component | Cost |
|---|---|
| spaCy parse of all 9.9k questions | ~8 s |
| BloomLex / TF-IDF models | seconds to train |
| BGE-small sentence encoder (33M params) | ~1 min to encode everything, cached |
| Qwen3-4B-Instruct (4-bit, via Ollama) | ~0.4 s/question zero-shot, ~5 s with reasoning, on a laptop (Apple silicon; slower on plain CPU) |

Fine-tuning a large transformer or an LLM (LoRA) was deliberately left out of the default
system: the cross-source ceiling is set by label conventions, not model capacity, and a
fine-tuned model memorises the conventions of its training sources harder. The LLM is used
where it adds something cheap models cannot: reasoning about unusual questions, and writing
new ones. Measured (RESULTS §6): on its own the 4B model is *worse* than the CPU classifier
across sources, but as a reasoning second opinion on the questions the CPU model is unsure
about it adds about +0.04 weighted F1.

## 3. NLP pipeline (`bloombot.text`)

| Step | Module | Why |
|---|---|---|
| Clean | `normalize.py` | strip numbering, marks, CO/BT tags (**a printed "BT4" is the label — leakage**), repair double-encoded UTF-8 |
| Parse | `linguistics.py` (spaCy `en_core_web_sm`) | dependency parse → directive predicates |
| Directive detection | root of an imperative (no subject or subject "you"), coordinated verbs (`conj`), complements of request frames ("you are required to *design*"), sequenced tasks ("…, and then *design*") | the level is set by what the student is told to do |
| Known parser failure | small models tag sentence-initial imperatives ("List", "State", "Name") as nouns; handled explicitly | otherwise the commonest Remember questions lose their verb |
| Verb–object & particles | "write *notes*" (Understand) vs "write a *program*" (Apply); "come *up* with" (Create); "evaluate the *integral*" (Apply) | verbs are ambiguous; their arguments disambiguate |
| Negation | "Do not just *list* …; *evaluate* …" drops *list* | contrast-set phenomenon |
| Question frames | "What would happen if …", "Do you agree …", "Which is better …" | questions without a Bloom verb |
| Compound tasks | levels combined as the distribution of the **maximum**: P(max ≤ k) = Π F_i(k) | OBE convention: a question is at the highest level it demands |
| Delexicalise | nouns/names/numbers → `<n>`, `<name>`, `<num>`; task cues ("advantages", "example", "diagram") kept | the model sees the cognitive form, not the subject |

The verb lexicon (`lexicon.py`) is built from the 19 cognitive processes of Anderson &
Krathwohl (2001, Table 5.1) plus common exam verbs. Ambiguous verbs carry a *distribution*
over levels, not a single label; it is a prior that the learned models can override.

## 4. Models (`bloombot.models`)

All models implement `fit(texts, y, sample_weight)` and `predict_proba(texts) -> (n, 6)`.

| Model | What it is | Role |
|---|---|---|
| `LexiconRule` | argmax of the parser's prior | training-free baseline; never "drops" because it never learns a source |
| `TfidfSVM` | word 1-2 + char 2-5 grams, linear SVM, Platt scaling | the classic strong baseline |
| `PosTfidfSVM` | TF weighted by part of speech (verbs most) | replication of the TFPOS-IDF baseline used by Faraji et al. (2026) |
| `EmbedLR` | frozen BGE-small embedding + LR | shows how much topic-heavy semantics transfer (badly) |
| **`BloomLex`** | directive/frame/negation/embedded-verb features + delexicalised n-grams + raw words + lexicon prior → **ordinal** logistic regression (Frank & Hall, 2001) | main CPU model |
| **`EmbedLex`** | BloomLex features + BGE-small embedding | production default (`Lexis`) |
| `LLMClassifier` | Qwen3-4B via Ollama, JSON-schema output, level probabilities from **token log-probs**, optional retrieval few-shot and chain-of-thought | `Socrates` |

Design choices worth noting:

* **Ordinal.** Levels are ordered; confusing Remember with Understand is less wrong than with
  Create. The ordinal model improves quadratic-weighted kappa (+0.03 in the ablation).
* **Source balancing.** Multi-source training weights each source equally; without it the
  3.5k Indian-engineering questions dominate (the single largest ablation effect, −0.029 F1).
* **Delexicalisation, measured.** On its own it helps (+0.023 wF1 when raw words are not
  used), but in the full model its information overlaps with the raw-word features (ablation
  effect 0.000 wF1, −0.005 QWK). Topic-swap flip rates of our models (12–14%) are similar to
  TF-IDF's (12–15%), so topic shortcuts are reduced only modestly; the bigger gains come from
  the parsed directive features, ordinal training and source balancing.
* **Probabilities, not self-reports.** The LLM writes the level as one digit; the
  log-probabilities of 1–6 at that position are its distribution. Verbal confidence is not used.

## 5. Evaluation (`bloombot.eval`)

| Protocol | Question it answers |
|---|---|
| `indomain` | grouped stratified 5-fold CV inside a source — the optimistic number papers report. Near-paraphrases (char-n-gram cosine ≥ 0.8) stay in one fold |
| `transfer` | train on one source, test on each other (Faraji et al., 2026) → the **drop** |
| `loso` | leave one source out, train on the rest — the realistic setting for a new college |
| `heldout` | train on all training sources, test on the Indian school-board set (never used for training) |
| `adapt` | k-shot convention adaptation, k ∈ {0, 10, 25, 50, 100}, 3 seeds, fixed test half |
| `ablation` | remove one BloomLex component at a time (LOSO) |
| `probes` | contrast set + CheckList-style invariance/directional tests |
| LLM | the same LOSO/held-out on a stratified subsample (CPU models scored on the same items), contrast set, sycophancy probe |
| `arbiter` | how to combine Lexis, Archivist and LLM (mixture vs log-linear, weights, cascade); per-item distributions saved; the rule is also chosen by nested leave-one-source-out selection so the reported gain is not flattered |
| `rewrite` | 40 real questions rewritten two levels up; acceptance, attempts, failed checks; CSV for human validation |

Metrics: weighted F1 (comparability), macro F1 (rare levels), accuracy, **within-one-level**,
**MAE**, **quadratic-weighted kappa**, LOTS/HOTS agreement, **ECE**; bootstrap 95% CIs and a
paired bootstrap for model comparisons.

## 6. The agent pipeline (`bloombot.pipeline`)

The roles from the proposal are kept, but as plain stages that pass typed data
(`schemas.py`), not as free-running chat agents. Each stage is testable on its own and the
expensive one (the LLM) runs only when needed.

```
Scribe ─► Lexis ─┬─► Arbiter ─► Architect ─► Composer ─► Level check ─► Critic ─► Student
          Archivist┘     ▲  (LLM only if Lexis is unsure            ▲                     │
          Socrates ──────┘   or disagrees with Archivist)           └─ retry with reasons ┘ (≤3)
```

| Agent | Implementation |
|---|---|
| Conductor | `conductor.py`; runs stages, keeps the audit trail (`TraceEvent`s) |
| Scribe | `intake.py`; PDF (pdfplumber) / DOCX / TXT / CSV → questions, sub-parts, OR-groups, marks, CO, printed BT tag |
| Lexis | trained EmbedLex (or BloomLex) |
| Archivist | cosine nearest neighbours in the labelled bank; institution examples first |
| Socrates | `LLMClassifier` with the Archivist's neighbours as few-shot examples; reasons before answering (chain-of-thought); asked only when Lexis is unsure or disagrees with the Archivist |
| Arbiter | weighted mixture p = Σ wᵢpᵢ / Σ wᵢ (Lexis 1, Socrates 0.5; the Archivist's neighbours are shown as evidence with weight 0). Chosen by the Arbiter study (RESULTS §6): log-linear pooling, tried first, let one overconfident voter override the others. Flags teacher review on low confidence or strong-voter disagreement |
| Architect | `blueprint.py`; distribution by marks (OR-alternatives share a slot), bound checks, greedy minimal rewrite plan |
| Composer | LLM rewrite with level definition, typical verbs, two retrieved exemplars at the target level, and feedback; template fallback offline |
| Level check | the rewrite is re-classified by the assessment layer, **not by the model that wrote it** |
| Critic | form, topic preservation (overlap of noun/adjective lemmas), not a near-duplicate |
| Student | for Analyze+: "could memory alone answer this?" — rejects fake higher-order questions |

## 7. Backend first, frontend later

`bloombot.api` is a FastAPI service whose request/response bodies are the pydantic models in
`pipeline/schemas.py`. A frontend needs only [API.md](API.md) and `docs/openapi.json`. Long
operations (rebalancing with an LLM) are background jobs that the client polls. CORS is open
for local development.

## 8. What changed from the earlier draft (and why)

| Earlier draft | Now | Reason |
|---|---|---|
| 5 sources, "all MIT" | 7 sources by **provenance**, licences recorded per source | Yahya/Mohammed are CC BY 4.0 (PLOS); ~60% of `bloombert_mix` came from an unlicensed repository |
| `indian_exams` (VMD7) as the Indian test | `indian_engineering` (3.5k, Devane v5 minus copies and synthetic rows) + `indian_boards` (test-only) | VMD7's file was mostly Yahya plus a few hundred rows; its Evaluate/Create rows were informal or broken; ~3.2k Indian exam questions were hidden inside `bloombert_mix`, which leaked into "held-out" tests |
| `bloombert_mix` | dropped | licence, plus 99.8% of rows lower-cased without punctuation — a model can identify the source from formatting |
| exact dedup only | exact + near-duplicate (cosine ≥ 0.9) across sources; paraphrase groups for CV | e.g. "cach memory" / "cache memory" |
| first Bloom verb rule | dependency-parsed directive + frames + negation + verb–object phrases | verb traps |
| Mohammed = 139, Kumar's 94% on Yahya | Mohammed = 141 (PLOS S1); Kumar et al. never name Yahya and use one split with augmentation | see the claims check in [RESULTS.md](RESULTS.md#corrections-to-claims-in-the-proposal) |
| AICTE "limit 40%" | presented as AICTE's *example*, not a rule | exact wording, pp. 22–23 |
| GPU required (DeBERTa, LoRA) | CPU-only | see §2 |
