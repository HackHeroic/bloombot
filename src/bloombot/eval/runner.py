"""Run all experiments and write ``results/``.

    results/tables/*.csv        one table per protocol (long format: model x train x test)
    results/predictions/*.parquet  every prediction with its probabilities, for error analysis
    results/summary.json        headline numbers (served by the API at /v1/experiments)

LLM runs use a stratified subsample of each source (``llm_per_source``) because a 4B model
needs ~0.5-2 s per question on a laptop; CPU models are scored on the same subsample too, so
the comparison is like for like.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from bloombot.config import get_settings
from bloombot.data.build import load_questions
from bloombot.data.sources import SOURCES, TRAIN_SOURCES
from bloombot.eval import protocols as P
from bloombot.eval.metrics import bootstrap_ci, paired_bootstrap, scores
from bloombot.models.base import BloomClassifier, balanced_weights

TEST_ONLY = [s.id for s in SOURCES if s.role == "test"]


def model_factories() -> dict[str, Callable[[], BloomClassifier]]:
    from bloombot.models import encoder as E
    from bloombot.models import linear as L

    return {
        "rule": L.LexiconRule,
        "tfidf_svm": L.TfidfSVM,
        "pos_tfidf_svm": L.PosTfidfSVM,
        "bloomlex": lambda: L.BloomLex(ordinal=True, raw_text=True),
        "embed_lr": E.EmbedLR,
        "embedlex": lambda: E.EmbedLex(ordinal=True, raw_text=True),
    }


def ablation_factories() -> dict[str, Callable[[], BloomClassifier]]:
    from bloombot.models import linear as L

    def named(name, f):
        def make():
            m = f()
            m.name = name
            return m
        return make

    return {
        "bloomlex (full)": named("bloomlex (full)", lambda: L.BloomLex(ordinal=True, raw_text=True)),
        "- delexicalised text": named("- delexicalised text", lambda: L.BloomLex(ordinal=True, raw_text=True, delex=False)),
        "- raw words": named("- raw words", lambda: L.BloomLex(ordinal=True, raw_text=False)),
        "- ordinal": named("- ordinal", lambda: L.BloomLex(ordinal=False, raw_text=True)),
        "- lexicon prior": named("- lexicon prior", lambda: L.BloomLex(ordinal=True, raw_text=True, prior_weight=0.0)),
        "- source balancing": named("- source balancing", lambda: L.BloomLex(ordinal=True, raw_text=True)),
    }


def llm_factory(spec: str) -> Callable[[], BloomClassifier]:
    from bloombot.models.llm import LLMClassifier

    s = get_settings()
    shots = 0 if spec.startswith("zero") else s.llm_shots
    cot = spec.endswith("cot")
    return lambda: LLMClassifier(s.llm, shots=shots, cot=cot, host=s.ollama_host)


def _stratified(df: pd.DataFrame, n: int, seed: int = 0) -> pd.DataFrame:
    """Up to n questions per source, keeping each level's share (at least one per present level)."""
    parts = []
    for _, d in df.groupby("source"):
        if len(d) <= n:
            parts.append(d)
            continue
        share = d.level.value_counts(normalize=True)
        alloc = {lv: max(1, int(round(n * sh))) for lv, sh in share.items()}
        while sum(alloc.values()) > n:  # rounding overshoot: trim the largest level
            alloc[max(alloc, key=alloc.get)] -= 1
        parts += [d[d.level == lv].sample(min(k, int((d.level == lv).sum())), random_state=seed)
                  for lv, k in alloc.items()]
    return pd.concat(parts)


def _save(out: Path, name: str, rows: pd.DataFrame, preds: pd.DataFrame | None = None) -> None:
    (out / "tables").mkdir(parents=True, exist_ok=True)
    rows.to_csv(out / "tables" / f"{name}.csv", index=False)
    if preds is not None and len(preds):
        (out / "predictions").mkdir(parents=True, exist_ok=True)
        preds.to_parquet(out / "predictions" / f"{name}.parquet", index=False)


def _ci_rows(preds: pd.DataFrame, by: list[str]) -> pd.DataFrame:
    rows = []
    for key, g in preds.groupby(by):
        lo, hi = bootstrap_ci(g.level.to_numpy(), g.pred.to_numpy(), "weighted_f1", n_boot=300)
        rows.append({**dict(zip(by, key if isinstance(key, tuple) else (key,))), "wf1_lo": lo, "wf1_hi": hi})
    return pd.DataFrame(rows)


def risk_coverage(preds: pd.DataFrame) -> pd.DataFrame:
    """Accuracy of the predictions kept when everything below a confidence threshold goes to the teacher."""
    rows = []
    for t in np.round(np.arange(0.0, 0.96, 0.05), 2):
        keep = preds.confidence >= t
        rows.append({"threshold": t, "coverage": float(keep.mean()),
                     "accuracy": float((preds.pred == preds.level)[keep].mean()) if keep.any() else None,
                     "within_one": float(((preds.pred - preds.level).abs() <= 1)[keep].mean()) if keep.any() else None})
    return pd.DataFrame(rows)


def run_all(protocols: list[str], models: list[str], llm_specs: list[str], llm_per_source: int, out: Path) -> dict:
    s = get_settings()
    df = load_questions(s.root)
    train_sources = list(TRAIN_SOURCES)
    facs = {k: v for k, v in model_factories().items() if k in models}
    out.mkdir(parents=True, exist_ok=True)
    summary: dict = {"generated": time.strftime("%Y-%m-%d %H:%M"), "n_questions": int(len(df)),
                     "train_sources": train_sources, "test_only_sources": TEST_ONLY}

    def log(msg):
        print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

    if "indomain" in protocols or "transfer" in protocols:
        ind_rows, ind_preds, tr_rows, tr_preds = [], [], [], []
        for name, f in facs.items():
            if "indomain" in protocols:
                log(f"indomain {name}")
                r, p = P.indomain(df, f, train_sources)
                ind_rows += [x.row() for x in r]
                ind_preds.append(p)
            if "transfer" in protocols:
                log(f"transfer {name}")
                r, p = P.transfer(df, f, train_sources)
                tr_rows += [x.row() for x in r]
                tr_preds.append(p)
        if ind_rows:
            _save(out, "indomain", pd.DataFrame(ind_rows), pd.concat(ind_preds))
        if tr_rows:
            _save(out, "transfer", pd.DataFrame(tr_rows), pd.concat(tr_preds))
        if ind_rows and tr_rows:
            drop = P.drop_table(pd.DataFrame(ind_rows), pd.DataFrame(tr_rows))
            _save(out, "drop", drop)
            summary["drop"] = drop.groupby("model")[["indomain", "transfer_mean", "drop"]].mean().round(3).to_dict("index")

    loso_preds = None
    if "loso" in protocols or "heldout" in protocols:
        rows, preds = [], []
        for name, f in facs.items():
            if "loso" in protocols:
                log(f"loso {name}")
                r, p = P.loso(df, f, train_sources)
                rows += [x.row() for x in r]
                preds.append(p)
            if "heldout" in protocols:
                log(f"heldout {name}")
                r, p, _ = P.heldout(df, f, train_sources, TEST_ONLY)
                rows += [x.row() for x in r]
                preds.append(p)
        table = pd.DataFrame(rows)
        loso_preds = pd.concat(preds)
        ci = _ci_rows(loso_preds, ["model", "protocol", "source"])
        table = table.merge(ci.rename(columns={"source": "test"}), on=["model", "protocol", "test"], how="left")
        _save(out, "loso", table, loso_preds)
        lo = table[table.protocol == "loso"]
        summary["loso_mean"] = lo.groupby("model")[["weighted_f1", "macro_f1", "qwk", "within_one", "ece"]].mean().round(3).to_dict("index")
        summary["heldout"] = table[table.protocol == "heldout"].set_index("model")[["weighted_f1", "macro_f1", "qwk", "within_one"]].round(3).to_dict("index")
        # significance: best learned model vs the classic baseline, pooled over LOSO items
        lp = loso_preds[loso_preds.protocol == "loso"]
        if {"embedlex", "tfidf_svm"} <= set(lp.model):
            a = lp[lp.model == "embedlex"].sort_values("uid")
            b = lp[lp.model == "tfidf_svm"].sort_values("uid")
            summary["embedlex_vs_tfidf_svm"] = paired_bootstrap(a.level.to_numpy(), a.pred.to_numpy(), b.pred.to_numpy())
        main = "embedlex" if "embedlex" in set(lp.model) else lp.model.iloc[0]
        rc = risk_coverage(lp[lp.model == main])
        _save(out, f"risk_coverage_{main}", rc)
        summary["risk_coverage"] = {"model": main, "rows": rc.round(3).to_dict("records")}

    if "ablation" in protocols:
        rows = []
        for name, f in ablation_factories().items():
            log(f"ablation {name}")
            r, _ = P.loso(df, f, train_sources, balance=name != "- source balancing")
            rows += [x.row() for x in r]
        t = pd.DataFrame(rows)
        _save(out, "ablation", t)
        summary["ablation"] = t.groupby("model")[["weighted_f1", "macro_f1", "qwk"]].mean().round(3).to_dict("index")

    if "adapt" in protocols:
        rows = []
        for name in [m for m in ("bloomlex",) if m in facs]:
            log(f"adapt {name}")
            r, _ = P.adapt(df, facs[name], train_sources, train_sources + TEST_ONLY)
            rows.append(r)
        t = pd.concat(rows)
        _save(out, "adapt", t)
        summary["adapt"] = t.groupby(["model", "target", "k"]).weighted_f1.mean().round(3).unstack().to_dict("index")
        summary["adapt"] = {f"{m}|{tg}": v for (m, tg), v in summary["adapt"].items()}

    if "probes" in protocols:
        from bloombot.eval.probes import behavioural_report, contrast_report, load_contrast

        cs = load_contrast(s.root)
        train = df[df.source.isin(train_sources)]
        res = {}
        for name, f in facs.items():
            log(f"probes {name}")
            m = f()
            if m.trainable:
                m.fit(train.question.tolist(), train.level.to_numpy(), sample_weight=balanced_weights(train.source))
            c = contrast_report(m, cs)
            b = behavioural_report(m, df[df.source.isin(["indian_engineering", "gani2023"])], n=400)
            res[name] = {"contrast": c, "behaviour": b}
        (out / "tables").mkdir(parents=True, exist_ok=True)
        (out / "tables" / "probes.json").write_text(json.dumps(res, indent=2, default=str))
        summary["probes"] = {k: {"contrast_accuracy": round(v["contrast"]["accuracy"], 3),
                                 "ladder_spearman": round(v["contrast"]["ladder_mean_spearman"], 3),
                                 "by_category": {r["category"]: round(r["accuracy"], 2) for r in v["contrast"]["by_category"]},
                                 "format_flip": round(v["behaviour"]["inv_format_flip_rate"], 3),
                                 "topic_flip": round(v["behaviour"]["inv_topic_flip_rate"], 3),
                                 "verb_raised": round(v["behaviour"]["dir_verb_raised"], 3)} for k, v in res.items()}

    if llm_specs:
        summary["llm"] = run_llm(df, llm_specs, llm_per_source, train_sources, out, loso_preds, log)

    if "arbiter" in protocols:
        summary["arbiter"] = run_arbiter(df, llm_per_source, train_sources, out, log)

    if "rewrite" in protocols:
        summary["rewrite"] = run_rewrite(df, 40, out, log)

    path = out / "summary.json"
    old = json.loads(path.read_text()) if path.exists() else {}
    old.update(summary)
    path.write_text(json.dumps(old, indent=2, default=str))
    log(f"wrote {path}")
    return old


def run_rewrite(df: pd.DataFrame, n: int, out: Path, log, seed: int = 0) -> dict:
    """Rewrite n real lower-order questions two levels up and record what the checks decide.
    Writes a CSV with empty human_* columns for two teachers to validate the accepted rewrites."""
    from bloombot.pipeline.assess import Assessor
    from bloombot.pipeline.rewrite import Rewriter
    from bloombot.pipeline.schemas import RewriteJob, level_name

    a = Assessor.load(use_llm=True)
    if a.llm is None:
        raise RuntimeError("the rewrite experiment needs the LLM (Ollama) to be running")
    pool = df[df.source.isin(["indian_engineering", "gani2023", "sangodiah2017"]) & (df.level <= 3)
              & df.question.str.split().str.len().between(4, 30)]
    sample = pool.sample(n, random_state=seed)
    rw = Rewriter(a, a.llm.client)
    rows = []
    for _, r in sample.iterrows():
        tgt = min(6, int(r.level) + 2)
        t0 = time.time()
        res = rw.run(RewriteJob(question_id=r.uid, original=r.question, from_level=level_name(r.level),
                                to_level=level_name(tgt)))
        last = res.attempts[-1] if res.attempts else None
        rows.append({"uid": r.uid, "original": r.question, "from": level_name(r.level), "to": level_name(tgt),
                     "accepted": res.accepted, "attempts": len(res.attempts), "rewrite": res.final_question or (last.question if last else ""),
                     "first_try_failed": ",".join(c.name for c in res.attempts[0].checks if not c.passed) if res.attempts else "",
                     "last_failed": ",".join(c.name for c in last.checks if not c.passed) if last else "",
                     "student_rejections": sum(any(c.name == "student" and not c.passed for c in at.checks) for at in res.attempts),
                     "seconds": round(time.time() - t0, 1),
                     "human_a_level": "", "human_a_ok": "", "human_b_level": "", "human_b_ok": ""})
        log(f"rewrite {r.uid}: {'accepted' if res.accepted else 'rejected'} after {len(res.attempts)}")
    t = pd.DataFrame(rows)
    _save(out, "rewrite", t)
    fails = pd.Series([f for x in t.first_try_failed for f in x.split(",") if f]).value_counts().to_dict()
    return {"n": int(len(t)), "accepted_rate": round(float(t.accepted.mean()), 3),
            "first_try_rate": round(float(((t.attempts == 1) & t.accepted).mean()), 3),
            "mean_attempts": round(float(t.attempts.mean()), 2), "first_try_failures": fails,
            "student_rejected_any": int((t.student_rejections > 0).sum()),
            "mean_seconds": round(float(t.seconds.mean()), 1)}


def _arbiter_candidates(L, N, G: dict, unsure) -> dict[str, tuple[np.ndarray, float]]:
    """Every combination rule in the study: name -> (pooled distributions, share sent to the LLM)."""
    from bloombot.pipeline.assess import pool

    def lin(ds, ws):
        return np.array([pool(list(d), ws, "linear") for d in zip(*ds)])

    def geo(ds, ws):
        return np.array([pool(list(d), ws, "geometric") for d in zip(*ds)])

    c = {"Lexis only": (L, 0.0),
         "Lexis + Archivist, geometric 0.3": (geo([L, N], [1, 0.3]), 0.0),
         "Lexis + Archivist, linear 0.3": (lin([L, N], [1, 0.3]), 0.0)}
    for name, P in G.items():
        c[f"LLM {name} only"] = (P, 1.0)
        for w in (0.2, 0.3, 0.5, 1.0):
            c[f"linear Lexis+LLM-{name} w={w}"] = (lin([L, P], [1, w]), 1.0)
            c[f"geometric Lexis+LLM-{name} w={w}"] = (geo([L, P], [1, w]), 1.0)
        for w in (0.3, 0.5):
            c[f"cascade linear Lexis+LLM-{name} w={w}"] = (
                np.where(unsure[:, None], lin([L, P], [1, w]), L), float(unsure.mean()))
        c[f"cascade geometric Lexis+Archivist+LLM-{name} w=0.5"] = (
            np.where(unsure[:, None], geo([L, N, P], [1, 0.3, 0.5]), geo([L, N], [1, 0.3])), float(unsure.mean()))
    return c


def run_arbiter(df: pd.DataFrame, per_source: int, train_sources: list[str], out: Path, log) -> dict:
    """How should Lexis, the Archivist and the LLM be combined? Compared on the same held-out
    items (LOSO / held-out), with the share of questions sent to the LLM.

    Choosing the best rule on the items it is then scored on would flatter it, so the study also
    reports a *nested* score: for each source, the rule is chosen on the other sources' items
    and applied to this one."""
    from bloombot.models.encoder import EmbedLex, embedding_cache
    from bloombot.models.llm import LLMClassifier

    s = get_settings()
    specs = {"zero": (0, False), "rag": (s.llm_shots, False), "rag-cot": (s.llm_shots, True)}
    sub = _stratified(df[df.source.isin(train_sources + TEST_ONLY)], per_source)
    frames = []
    for held in train_sources + TEST_ONLY:
        test = sub[sub.source == held]
        train = df[df.source.isin([x for x in train_sources if x != held])]
        lex = EmbedLex(ordinal=True, raw_text=True).fit(train.question.tolist(), train.level.to_numpy(),
                                                        sample_weight=balanced_weights(train.source))
        f = test[["uid", "source", "question", "level"]].copy()
        f["lex"] = list(lex.predict_proba(test.question.tolist()))
        enc = embedding_cache(s.encoder)
        sims = enc.encode(test.question.tolist()) @ enc.encode(train.question.tolist()).T
        p_nb = []
        for row in sims:
            idx = np.argsort(-row)[: s.neighbours]
            w = np.exp((row[idx] - row[idx].max()) / 0.05)
            p = np.full(6, 0.5 / 6)
            for j, wj in zip(idx, w):
                p[train.level.iat[j] - 1] += wj
            p_nb.append(p / p.sum())
        f["nb"] = p_nb
        for name, (shots, cot) in specs.items():
            m = LLMClassifier(s.llm, shots=shots, cot=cot, host=s.ollama_host)
            if shots:
                m.fit(train.question.tolist(), train.level.to_numpy())
            f[name] = list(m.predict_proba(test.question.tolist()))
        log(f"arbiter {held}: {len(test)} items")
        frames.append(f)
    rec = pd.concat(frames, ignore_index=True)
    (out / "predictions").mkdir(parents=True, exist_ok=True)
    rec.assign(**{k: rec[k].map(list) for k in ["lex", "nb", *specs]}).to_parquet(out / "predictions" / "arbiter.parquet")

    y = rec.level.to_numpy()
    L, N = np.stack(rec.lex), np.stack(rec.nb)
    G = {k: np.stack(rec[k]) for k in specs}
    unsure = (L.max(1) < s.llm_trigger_confidence) | (L.argmax(1) != N.argmax(1))
    cands = _arbiter_candidates(L, N, G, unsure)
    rows = [{"combination": k, "llm_calls": round(share, 3), **scores(y, P.argmax(1) + 1, P)}
            for k, (P, share) in cands.items()]
    t = pd.DataFrame(rows)

    # nested choice: the rule (among those with an LLM, and among all) is picked without the held source
    src = rec.source.to_numpy()
    nested = {}
    for family, names in {"any rule": list(cands),
                          "rules without the LLM": [k for k in cands if "LLM" not in k],
                          "rules with LLM-rag": [k for k in cands if "LLM-rag " in k or k.endswith("LLM-rag")]}.items():
        pred, chosen = np.zeros(len(y), int), {}
        for held in np.unique(src):
            other = src != held
            best = max(names, key=lambda k: scores(y[other], cands[k][0][other].argmax(1) + 1)["weighted_f1"])
            pred[~other] = cands[best][0][~other].argmax(1) + 1
            chosen[held] = best
        nested[family] = {"weighted_f1": round(scores(y, pred)["weighted_f1"], 3), "chosen": chosen,
                          "vs_lexis": paired_bootstrap(y, pred, L.argmax(1) + 1)}
    best = t.sort_values("weighted_f1", ascending=False).combination.head(5).tolist()
    tests = {k: paired_bootstrap(y, cands[k][0].argmax(1) + 1, L.argmax(1) + 1) for k in best if k != "Lexis only"}
    _save(out, "arbiter", t)
    keep = ["weighted_f1", "macro_f1", "qwk", "within_one", "ece", "llm_calls"]
    return {"n": int(len(y)), "table": t.set_index("combination")[keep].round(3).to_dict("index"),
            "nested": nested, "top_vs_lexis": tests}


def run_llm(df, specs, per_source, train_sources, out: Path, cpu_preds, log) -> dict:
    """LOSO + held-out on a stratified subsample, plus the contrast set and the sycophancy probe."""
    from bloombot.eval.probes import contrast_report, load_contrast, sycophancy_report

    s = get_settings()
    sub = _stratified(df[df.source.isin(train_sources + TEST_ONLY)], per_source)
    rows, preds, extra = [], [], {}
    for spec in specs:
        make = llm_factory(spec)
        for held in train_sources + TEST_ONLY:
            test = sub[sub.source == held]
            if not len(test):
                continue
            pool = df[df.source.isin([x for x in train_sources if x != held])]
            m = make()
            t0 = time.time()
            if m.trainable:
                m.fit(pool.question.tolist(), pool.level.to_numpy())
            proba = m.predict_proba(test.question.tolist())
            secs = (time.time() - t0) / max(1, len(test))
            log(f"llm {spec} on {held}: {len(test)} questions, {secs:.2f}s each")
            y, pred = test.level.to_numpy(), proba.argmax(1) + 1
            proto = "heldout" if held in TEST_ONLY else "loso"
            rows.append({"protocol": proto, "model": m.name, "spec": spec, "test": held, "sec_per_q": secs,
                         **scores(y, pred, proba)})
            pf = test[["uid", "source", "question", "level"]].copy()
            pf["pred"], pf["confidence"], pf["model"], pf["protocol"] = pred, proba.max(1), m.name, proto
            for k in range(6):
                pf[f"p{k + 1}"] = proba[:, k]
            pf["reasoning"] = [o.get("reasoning") for o in m.last_outputs]
            preds.append(pf)
        m = make()
        if m.trainable:
            pool = df[df.source.isin(train_sources)]
            m.fit(pool.question.tolist(), pool.level.to_numpy())
        cs = load_contrast(s.root)
        extra[spec] = {"contrast": contrast_report(m, cs), "sycophancy": sycophancy_report(m, cs)}
    table = pd.DataFrame(rows)
    allp = pd.concat(preds)
    # CPU models on exactly the same items
    if cpu_preds is None and (out / "predictions" / "loso.parquet").exists():
        cpu_preds = pd.read_parquet(out / "predictions" / "loso.parquet")
    if cpu_preds is not None:
        same = cpu_preds[cpu_preds.uid.isin(set(sub.uid))]
        for (model, proto, src), g in same.groupby(["model", "protocol", "source"]):
            rows.append({"protocol": proto, "model": model, "spec": "cpu", "test": src,
                         **scores(g.level.to_numpy(), g.pred.to_numpy())})
        table = pd.DataFrame(rows)
    _save(out, "llm", table, allp)
    (out / "tables" / "llm_probes.json").write_text(json.dumps(extra, indent=2, default=str))
    lo = table[table.protocol == "loso"]
    return {
        "per_source": per_source,
        "loso_mean_same_items": lo.groupby("model")[["weighted_f1", "macro_f1", "qwk"]].mean().round(3).to_dict("index"),
        "heldout_same_items": table[table.protocol == "heldout"].set_index("model")[["weighted_f1", "qwk"]].round(3).to_dict("index"),
        "sec_per_question": table.dropna(subset=["sec_per_q"]).groupby("model").sec_per_q.mean().round(2).to_dict()
        if "sec_per_q" in table else {},
        "contrast": {k: round(v["contrast"]["accuracy"], 3) for k, v in extra.items()},
        "sycophancy_followed_hint": {k: round(v["sycophancy"]["followed_hint_rate"], 3) for k, v in extra.items()},
    }
