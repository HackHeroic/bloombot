"""Record the College profile page's question pool and the generic model's view of each question.

The pool is the design's own sample questions plus the Indian engineering questions in the web
UI's dataset sample (frontend/src/recorded/dataset.json), each with its real label from that
source. Each question is classified by the generic model (Lexis only: no LLM, no profile), so the
page can show, level by level, where a teacher's labels differ from the model, which is what a
college profile then shifts. The result is written into frontend/design/bloombot-data.js between
the profile-pool markers; run `npm run design` in frontend/ afterwards.

Usage (with `bloombot serve` running): python scripts/record_profile_pool.py [http://127.0.0.1:8001]
"""

from __future__ import annotations

import ast
import json
import random
import re
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_JS = ROOT / "frontend" / "design" / "bloombot-data.js"
DATASET = ROOT / "frontend" / "src" / "recorded" / "dataset.json"
LEVELS = ["Remember", "Understand", "Apply", "Analyze", "Evaluate", "Create"]
# Indian engineering: questions per level in the whole source (docs/DATA.md)
NATURAL = [n / 3520 for n in (1822, 416, 764, 503, 13, 2)]
START, END = "/* profile-pool:start */", "/* profile-pool:end */"


def classify(api: str, questions: list[str]) -> list[list[float]]:
    out = []
    for i in range(0, len(questions), 25):
        body = json.dumps({"questions": questions[i:i + 25], "use_llm": False}).encode()
        req = urllib.request.Request(f"{api}/v1/classify", data=body, headers={"content-type": "application/json"})
        with urllib.request.urlopen(req, timeout=120) as r:
            preds = json.load(r)["predictions"]
        out += [[round(p["proba"][lv], 3) for lv in LEVELS] for p in preds]
    return out


def main() -> None:
    api = sys.argv[1].rstrip("/") if len(sys.argv) > 1 else "http://127.0.0.1:8001"
    text = DATA_JS.read_text()
    base = ast.literal_eval(re.search(r"const POOL=(\[.*?\]);", text, re.S).group(1))
    items = [x for x in json.loads(DATASET.read_text())["items"] if x["source"] == "indian_engineering"]
    seen, pool, labels = set(), [], []
    for q, lv in [(q, 0) for q in base] + [(x["question"], LEVELS.index(x["level"]) + 1) for x in items]:
        key = re.sub(r"\W+", " ", q.lower()).strip()
        if key in seen:
            continue
        seen.add(key)
        pool.append(q)
        labels.append(lv)
    # The design's questions first, then the labelled ones in the source's natural mix of levels
    # (the dataset sample is stratified, 20 per level; a college's past papers are not), so the
    # first 25-50 labels look like a real college's.
    rest = list(range(len(base), len(pool)))
    random.Random(7).shuffle(rest)
    by_level = {lv: [i for i in rest if labels[i] == lv] for lv in range(1, 7)}
    order, counts = list(range(len(base))), {lv: 0 for lv in range(1, 7)}
    while any(by_level.values()):
        n = sum(counts.values()) + 1
        lv = max((lv for lv in by_level if by_level[lv]), key=lambda lv: NATURAL[lv - 1] * n - counts[lv])
        order.append(by_level[lv].pop(0))
        counts[lv] += 1
    pool, labels = [pool[i] for i in order], [labels[i] for i in order]
    preds = classify(api, pool)
    block = (f"{START}\n// Profile page pool: question, real label at its source (0 = none), generic model's"
             f" probabilities. Written by scripts/record_profile_pool.py.\n"
             f"Object.assign(window.BB,{{POOL:{json.dumps(pool, ensure_ascii=False)},"
             f"POOL_LBL:{json.dumps(labels)},POOL_PRED:{json.dumps(preds)}}});\n{END}")
    if START in text:
        text = re.sub(re.escape(START) + r".*?" + re.escape(END), lambda _: block, text, flags=re.S)
    else:
        text = text.rstrip("\n") + "\n" + block + "\n"
    DATA_JS.write_text(text)
    agree = sum(1 for lv, p in zip(labels, preds) if lv and p.index(max(p)) + 1 == lv)
    known = sum(1 for lv in labels if lv)
    print(f"pool {len(pool)} questions, {known} with a real label; generic model agrees on {agree}/{known}")


if __name__ == "__main__":
    main()
