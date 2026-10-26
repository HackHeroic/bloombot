# Frontend brief — prompt for Claude Design

Paste everything under the line below into Claude Design. It is self-contained: Claude Design
does not see this repository. Bring the resulting design artifact back to Claude Code, which
will implement it in `frontend/` against the real API ([API.md](API.md),
[openapi.json](openapi.json)).

---

## Prompt

Design the web app for **BloomBot**, an open, free tool that helps Indian college teachers
check and rebalance exam question papers by **Bloom's revised taxonomy**. It is also the
**demonstration** of a research project, so it must explain *how* the system works and *how
well*, not just what it outputs. Build an interactive, high-fidelity prototype that uses the
mock data below. The data has exactly the shapes the real API returns, so the design can be
wired up without reshaping anything.

### Who uses it

- **Faculty / HoD at an Indian engineering or arts college.** They have to prove to NBA/NAAC
  auditors that a paper meets an Outcome-Based-Education blueprint. Busy, not technical, on
  a laptop. They want to know three things:
  - Is my paper balanced?
  - Which questions are the problem?
  - Can you fix them, without me losing control?
- **Evaluators / a project panel** watching a demo. They want to see the architecture, the
  NLP, and honest evaluation numbers.

### Personality and visual direction

- Calm, trustworthy, academic, but **warm and modern**: a well-made research tool, not a
  corporate dashboard and not a toy.
- Generous whitespace and a clear typographic hierarchy. Pair one distinctive serif or
  humanist display face for headings with a clean sans for UI and a mono for code and
  evidence. Light and dark themes.
- **The six Bloom levels are ordered** (Remember < Understand < Apply < Analyze < Evaluate <
  Create). Encode them with **one ordered colour ramp**: light/cool for lower order, deep/warm
  for higher order. Never use six unrelated hues. Always show the level *name* (or its number
  1–6) next to the colour; never rely on colour alone. Lower order is Remember–Understand,
  higher order is Analyze–Create (Apply sits in the middle). That grouping should be visible
  in distribution bars.
- **Signature visual: the "level ladder".** A vertical 6-step ladder showing a question's
  probability on each level. It is used everywhere a prediction appears: large on the
  question page, compact as an inline sparkline in tables.
- **Uncertainty is a first-class citizen.** Show confidence, flag "⚑ needs teacher review",
  and when voters disagree, show them side by side. The product's honesty is its selling point.
- Motion is purposeful and limited: pipeline stages light up while a job runs; numbers
  count up once.
- Accessible (WCAG AA contrast, keyboard paths, visible focus). Desktop first (1280–1440),
  but the Classify page and the results pages must work on a phone.

### Information architecture (left rail or top nav)

1. **Check a paper** (home)
2. **Classify a question**
3. **Rebalance** (opens from a paper)
4. **College profile** (adaptation)
5. **How it works** (architecture and NLP explainer, for the demo)
6. **Evaluation** (results, for the demo)
7. **Data** (sources and data card)

A status chip in the header shows whether the local LLM is available
(`/v1/meta → llm.available`). Show "Fast mode (no LLM)" vs "LLM second opinion on".

### Screens

**1. Check a paper**

1. Drop zone for PDF / DOCX / TXT / CSV, or paste text.
2. **Parsed paper editor.** Sections, numbered questions and sub-parts (1a, 1b…), marks,
   CO, the printed BT tag, and OR-alternatives grouped visually ("answer either 11a or
   11b"). Every row is editable (text, marks), and a question can be 🔒 locked so it is
   never rewritten.
3. Pick a blueprint: "AICTE 2018 example (≤ 40% lower order)" or "Balanced UG".
4. **Assess**, which leads to the report:
   - a **distribution bar by marks** across the 6 levels, with the blueprint's bounds drawn
     on it as brackets or threshold markers, plus a pass/fail check list ("Lower order ≤ 40%
     of marks — 90% ✗, 50 points over")
   - one big verdict: *Compliant* / *Not compliant*
   - a question table with these columns: id, text, marks, predicted level (coloured chip +
     mini ladder), confidence, printed tag vs predicted (highlight disagreements), ⚑ review
   - a **review queue** of the flagged questions, to resolve first
   - a primary CTA: **"Rebalance this paper"**

**2. Classify a question** (also the demo's opening)

- A large text box with example chips, including the verb trap *"Explain how engineers
  evaluate the safety of an old bridge."*
- Result: the level ladder, the confidence, and the **evidence**:
  - the sentence with the **directive verb highlighted** and any embedded Bloom verbs marked
    as "trap: not the task"
  - the question frame, if any
  - the masked text, shown as a "what the model sees" toggle
  - the **votes** from Lexis, Archivist and Socrates (the LLM), each with its own small
    ladder, plus the LLM's reasoning text, collapsible
  - the **5 most similar labelled questions**, with their level, source and similarity
- A "View parse" drawer that shows the dependency parse as an arc diagram over the tokens
  (ROOT = directive).

**3. Rebalance** (a background job; poll `/v1/jobs/{id}`)

- While running: a **live pipeline timeline** built from the job's `trace` events. Stages:
  Scribe → Lexis → Archivist → Socrates → Arbiter → Conductor → Architect → Composer →
  Level check → Critic → Student. Each event has an agent, an action and the ms it took.
- The **plan**: "4 rewrites needed", each written as Q6 Remember → Apply.
- For each rewrite, a card with:
  - original vs rewrite (a word-level diff is nice)
  - target level
  - the attempts (1–3)
  - the **checks** as pass/fail pills, each with its detail: level (the new level's
    probability), form, topic (content-word overlap 0.xx), not duplicate (similarity),
    student ("could memory alone answer this?" and why)
  - Accept / Edit / Reject buttons
  - rejected rewrites are clearly handed back to the teacher
- **Before vs after**: two distribution bars and the bound checks flipping ✗ → ✓.
- Export: download the rebalanced paper and the report (Markdown/JSON).

**4. College profile**

- Explain: "Colleges label the same verb differently. Label 25–100 of your own past
  questions and BloomBot learns your convention."
- A fast labelling UI: a question card with 6 level buttons, keyboard shortcuts 1–6, and a
  progress ring to 25 / 50 / 100.
- An **expected gain chart** from the real experiment: k = 0, 10, 25, 50, 100 →
  0.627, 0.639, 0.656, 0.674, 0.694 mean weighted F1. Indian boards: 0.578 → 0.674.
- A list of profiles with their example counts; select one to use it in assessments.

**5. How it works** (the demo's architecture page; it should feel like an interactive
explainer, not a wall of text)

- **An interactive pipeline diagram** of the agents. Clicking a node opens a side panel with
  what the stage does, why it was designed that way, and its cost:
  - **Scribe:** parses papers; printed BT tags are the answer key, so they are stripped.
  - **Lexis:** EmbedLex ordinal classifier, about 1 ms per question on CPU.
  - **Archivist:** nearest labelled questions (BGE-small embeddings).
  - **Socrates:** Qwen3-4B local LLM with 6 retrieved examples, reasoning first. Asked only
    when Lexis is unsure: confidence < 0.6, or Lexis and the Archivist disagree.
  - **Arbiter:** a mixture of Lexis (weight 1) and the LLM (weight 0.5). An experiment
    showed that multiplying probabilities lost 0.15 F1.
  - **Architect:** marks-weighted blueprint check; the fewest rewrites that fix it.
  - **Composer:** the LLM rewrites to a target level.
  - **Level check / Critic / Student:** independent verifiers. The writer never grades
    itself. Up to 3 attempts, then the teacher.
- **"Follow one question"**: a stepper over the verb-trap example, showing at each step
  what the NLP produces (real data below):
  1. clean
  2. dependency parse
  3. directive found
  4. embedded verb flagged
  5. topic words masked
  6. lexicon prior
  7. classifier output
  8. unsure, so the LLM is asked
  9. the Arbiter's result
- A **"Why CPU, not GPU"** callout: questions are about 20 tokens and there are about 10k
  of them; training takes about 30 s on a laptop. The limit is the colleges' label
  conventions, not model size.

**6. Evaluation** (the demo's results page; data-dense but readable, with a narrative
running down the page as numbered findings)

1. **"Accuracy on one dataset is misleading."** Dumbbell chart, same source → other source:

   | Model | Same source | Other source | Drop |
   |---|---:|---:|---:|
   | TF-IDF + SVM | 0.732 | 0.413 | 0.320 |
   | POS-weighted SVM | 0.743 | 0.435 | 0.307 |
   | Embedding + LR | 0.638 | 0.359 | 0.279 |
   | Lexicon rule | 0.624 | 0.624 | 0.000 |
   | BloomLex | 0.754 | 0.522 | 0.232 |
   | **EmbedLex (ours)** | 0.764 | 0.524 | 0.239 |

2. **"Leave-one-source-out: train on 5 colleges/sources, test on the 6th."** Weighted F1:

   | Model | LOSO mean |
   |---|---:|
   | TF-IDF + SVM | 0.589 |
   | POS-weighted SVM | 0.590 |
   | Embedding + LR | 0.502 |
   | Lexicon rule | 0.624 |
   | BloomLex | 0.632 |
   | **EmbedLex** | **0.641** |

   EmbedLex: QWK 0.701. It beats TF-IDF by +0.051 [0.041, 0.060], p < 0.001. On unseen
   Indian school-board questions it scores 0.630 [0.608, 0.649], within one level 0.82.

   EmbedLex per held-out source (CI bars):

   | Source | wF1 [95% CI] |
   |---|---|
   | Yahya | 0.684 [0.643, 0.716] |
   | Mohammed | 0.649 [0.568, 0.729] |
   | Sangodiah | 0.675 [0.629, 0.717] |
   | Gani | 0.715 [0.689, 0.737] |
   | Scaria | 0.694 [0.672, 0.716] |
   | Indian engineering | 0.429 [0.412, 0.447] |

3. **"Why it can't be perfect: colleges disagree."** A small-multiples heatmap: verb ×
   source → share of each level. Rows are sources, columns are the 6 levels.

   | Verb | Source | Remember | Understand | Apply | Analyze | Evaluate | Create |
   |---|---|---:|---:|---:|---:|---:|---:|
   | explain | Gani | 0.00 | 0.93 | 0.02 | 0.03 | 0.02 | 0.00 |
   | explain | Indian engineering | 0.55 | 0.24 | 0.06 | 0.15 | 0.00 | 0.00 |
   | explain | Indian boards | 0.06 | 0.77 | 0.09 | 0.02 | 0.06 | 0.00 |
   | explain | Scaria | 0.01 | 0.72 | 0.09 | 0.18 | 0.00 | 0.00 |
   | design | Indian engineering | 0.07 | 0.00 | 0.93 | 0.00 | 0.00 | 0.00 |
   | design | Scaria | 0.00 | 0.00 | 0.02 | 0.00 | 0.01 | 0.96 |
   | write | Indian boards | 0.65 | 0.15 | 0.17 | 0.03 | 0.00 | 0.00 |
   | write | Indian engineering | 0.58 | 0.13 | 0.24 | 0.04 | 0.00 | 0.00 |

4. **"Fix: a few labelled questions per college."** Line chart, k = 0/10/25/50/100:

   | Target | k=0 | 10 | 25 | 50 | 100 |
   |---|---:|---:|---:|---:|---:|
   | Indian engineering | 0.435 | 0.451 | 0.474 | 0.505 | 0.527 |
   | Indian boards | 0.578 | 0.609 | 0.615 | 0.648 | 0.674 |
   | Gani | 0.718 | 0.728 | 0.747 | 0.763 | 0.792 |
   | Scaria | 0.680 | 0.694 | 0.718 | 0.731 | 0.752 |
   | Sangodiah | 0.702 | 0.707 | 0.713 | 0.720 | 0.724 |
   | Yahya | 0.644 | 0.655 | 0.663 | 0.675 | 0.695 |
   | **Mean** | 0.627 | 0.639 | 0.656 | 0.674 | 0.694 |

5. **"Reading the directive beats counting words."** Contrast-set accuracy (verb traps):

   | Model | Verb traps |
   |---|---:|
   | POS-weighted SVM | 0.2 |
   | TF-IDF | 0.6 |
   | Embedding | 0.4 |
   | BloomLex / EmbedLex | **0.9** |
   | LLM + reasoning | 0.8 |

   Topic-swap flip rate: TF-IDF 12–15%, ours 12–14%, embedding 24%.

6. **"Is a small LLM better?"** Same 418 questions:

   | Model | wF1 | Time / question |
   |---|---:|---:|
   | EmbedLex | 0.665 | ~1 ms |
   | LLM zero-shot | 0.446 | 0.4 s |
   | LLM + 6 examples | 0.508 | 0.6 s |
   | LLM + examples + reasoning | 0.621 | 5.1 s |

   Calibration error: LLM 0.39–0.50 vs EmbedLex 0.07.

   **Sycophancy:** told "a senior teacher thinks it's X" (a wrong level), the LLM switched
   to X 31% / 47% / 29% of the time (zero-shot / RAG / reasoning).

7. **"Combining them, chosen fairly."**

   | Combination | wF1 |
   |---|---:|
   | Lexis only | 0.640 |
   | **Lexis + LLM-reasoning as second opinion when unsure (69% of items)** | **0.684** |
   | Multiplying probabilities (first design) | 0.488 |

   With the rule chosen without seeing the test source: 0.682, +0.042 [0.005, 0.078],
   p = 0.01.

8. **"Rewrites."** 40 real questions rewritten two levels up:

   | Result | Value |
   |---|---|
   | Passed every automatic check | 37/40 (92.5%) |
   | First try | 80% |
   | Mean attempts | 1.3 |
   | Time per question | 4.5 s |
   | First-attempt failures | topic 4, student 3, level 1 |
   | By target | Remember→Apply 10/12, Understand→Analyze 20/21, Apply→Evaluate 7/7 |

   Show two or three example cards (below).

9. **Limits, stated plainly:** about 6 in 10 exactly right at a new college (8–9 in 10
   within one level), so the teacher stays in the loop. Rewrites still need human validation.

**7. Data**

A table of the 7 sources: name, rows, role (train/test), licence, origin.

| Source | Rows | Role | Licence | Origin |
|---|---:|---|---|---|
| yahya2012 | 597 | train | CC BY 4.0 | item bank |
| mohammed2020 | 136 | train | CC BY 4.0 | web |
| sangodiah2017 | 413 | train | MIT | university exam questions (Malaysia) |
| gani2023 | 1183 | train | MIT | university exam questions (Malaysia) |
| scaria2024 | 1728 | train | MIT | LLM-generated, kept where 2 experts agree |
| indian_engineering | 3520 | train | Apache 2.0 | Indian engineering university papers |
| indian_boards | 2278 | **test only** | CC BY-NC 3.0 | Indian school-board questions |

Total: 9,855.

### Mock data (the real API shapes)

**Real responses for every screen are in `frontend/fixtures/*.json`** (regenerate with `python scripts/make_fixtures.py`): five example questions with full votes, LLM reasoning and 5 neighbours, the parsed and assessed sample paper with full probabilities, the rebalance job with every check value and the 31-event trace, and the experiment tables. Use those rather than the excerpt below.

`POST /v1/classify` returns a `Prediction` per question. Lexis and the neighbours below are real output; the Archivist and Socrates values are illustrative (in the real run the LLM falls for this trap and votes Evaluate, so the pooled answer is Evaluate 0.37 and flagged ⚑; see `fixtures/classify.json`):

```json
{
  "question": "Explain how engineers evaluate the safety of an old bridge.",
  "level": "Understand", "level_id": 2, "confidence": 0.51, "needs_review": false, "lower_order": true,
  "proba": {"Remember": 0.01, "Understand": 0.51, "Apply": 0.00, "Analyze": 0.43, "Evaluate": 0.05, "Create": 0.00},
  "directives": [{"verb": "explain", "kind": "imperative", "phrase": null, "negated": false}],
  "frame": null, "evidence": "directive",
  "votes": [
    {"agent": "Lexis", "level": "Understand", "confidence": 0.51, "proba": [0.01, 0.51, 0.0, 0.43, 0.05, 0.0]},
    {"agent": "Archivist", "level": "Analyze", "confidence": 0.46, "proba": [0.1, 0.2, 0.12, 0.46, 0.08, 0.04]},
    {"agent": "Socrates", "level": "Understand", "confidence": 0.93, "proba": [0.02, 0.93, 0.01, 0.03, 0.01, 0.0],
     "reasoning": "The task is to explain a process engineers follow; the student describes, not judges."}
  ],
  "neighbours": [
    {"question": "Explain in detail the working of wheatstone bridge with equations.", "level": "Analyze", "similarity": 0.746, "source": "indian_engineering"},
    {"question": "Elaborate on detectors used in AC bridges.", "level": "Apply", "similarity": 0.721, "source": "indian_engineering"},
    {"question": "Describe the classification of fault and their engineering significance.", "level": "Remember", "similarity": 0.718, "source": "indian_engineering"}
  ]
}
```

`POST /v1/analyze` (for the "follow one question" stepper):

```json
{
  "text": "Explain how engineers evaluate the safety of an old bridge.",
  "directives": [{"lemma": "explain", "kind": "imperative", "negated": false}],
  "embedded_bloom_verbs": ["evaluate"], "frame": null, "wh_word": "how",
  "masked": "explain how <n> evaluate the <n> of an <adj> <n> .",
  "prior": {"Remember": 0.10, "Understand": 0.70, "Apply": 0.0, "Analyze": 0.15, "Evaluate": 0.05, "Create": 0.0},
  "evidence": "directive"
}
```

A parsed paper (`Paper.questions[]`): `{id: "1a", number: "Q1 (a)", text, marks: 2, section: "A",
group: null | "11" (OR-group), option: "a", course_outcome: "CO1", tagged_level: "Remember", locked: false}`.
The sample OS paper (50 marks) has these questions:

| id | Question | Marks | Predicted |
|---|---|---:|---|
| 1a | Define a process. | 2 | Remember 0.97 |
| 1b | List the states of a process. | 2 | Remember 0.99 |
| 1c | What is a context switch? | 2 | Remember 0.98 |
| 1d | State the necessary conditions for deadlock. | 2 | Remember 0.86 |
| 1e | What is thrashing? | 2 | Remember 0.97 |
| 2 | Explain the difference between preemptive and non-preemptive scheduling. | 10 | Understand 0.54 ⚑ (Lexis Understand 0.81, LLM Analyze 1.00) |
| 3 | Describe the producer-consumer problem and its solution using semaphores. | 10 | Understand 0.80 |
| 4a | Draw the Gantt chart for FCFS scheduling. | 5 | Remember 0.63 |
| 4b | Calculate the average waiting time for FCFS and SJF. | 5 | Apply 0.92 |
| 5 | Explain paging with a neat diagram. | 10 | Remember 0.62 |
| 6 | Write short notes on virtual memory and demand paging. | 10 | Remember 0.72 |

Distribution by marks: Remember 70%, Understand 20%, Apply 10%. Against **Balanced UG**:

| Bound | Share | Result |
|---|---:|---|
| Lower order ≤ 40% | 90% | ✗ |
| Apply + Analyze ≥ 40% | 10% | ✗ |
| Evaluate + Create ≥ 10% | 0% | ✗ |

The rebalance job planned 4 rewrites, all accepted on the first attempt:

| Q | Change | Rewrite |
|---|---|---|
| Q6 | Remember → Apply | "Apply demand paging to a system with 512 MB physical memory and 2 GB virtual memory, where a process initially loads 128 MB of code and data. Show how the system handles page faults…" |
| Q5 | Remember → Apply | "Apply paging to a memory management scenario with 4 pages and 8 frame slots; show how processes are loaded and pages are mapped to frames, including page table entries." |
| Q3 | Understand → Apply | "Implement a solution to the producer-consumer problem using semaphores in a shared memory environment where two threads produce and consume data from a buffer of size 5." |
| Q4b | Apply → Evaluate | "Critique the fairness of FCFS and SJF scheduling in the given scenario, assessing which algorithm better meets the criteria of minimizing average waiting time and avoiding starvation." |

After: Remember 30%, Understand 10%, Apply 50%, Evaluate 10%. All three bounds ✓, compliant.

A trace event looks like `{agent: "Composer", action: "drafted", ms: 4120, detail: {...}}`.

A rejected rewrite to show honestly:

- Original: "Describe MOST with suitable illustration."
- Rewrite: "Use the formula for calculating compound interest…"
- Topic check ✗ (content-word overlap 0.00), after 3 attempts. Handed back to the teacher.

### Deliverable

- An interactive prototype covering all 7 screens, with the flows connected: upload → report
  → rebalance → before/after.
- Both themes, and the Classify and Evaluation pages at phone width.
- A small component sheet: level chip, level ladder (large and inline), distribution bar
  with bounds, check pill, vote card, rewrite card, pipeline node, trace timeline, review
  flag.
- Use these exact labels and numbers. Do not invent other metrics.
