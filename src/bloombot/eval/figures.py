"""Figures and the Markdown results report, generated from ``results/``."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from bloombot.taxonomy import LABELS

SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]  # validated categorical slots 1-5
BLUES = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]

PRETTY = {"lexicon_rule": "Lexicon rule (no training)", "tfidf_svm": "TF-IDF + SVM", "pos_tfidf_svm": "POS-weighted TF-IDF + SVM",
          "embed_lr": "Sentence embedding + LR", "bloomlex": "BloomLex (ours)", "embedlex": "EmbedLex (ours)"}
SRC = {"yahya2012": "Yahya", "mohammed2020": "Mohammed", "sangodiah2017": "Sangodiah", "gani2023": "Gani",
       "scaria2024": "Scaria", "indian_engineering": "Indian engineering", "indian_boards": "Indian boards"}


def _style(ax, grid_axis="x"):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=9, length=0)
    ax.grid(axis=grid_axis, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def _fig(w, h):
    fig, ax = plt.subplots(figsize=(w, h), dpi=160)
    fig.patch.set_facecolor(SURFACE)
    return fig, ax


def fig_drop(results: Path, out: Path) -> None:
    drop = pd.read_csv(results / "tables" / "drop.csv").groupby("model")[["indomain", "transfer_mean"]].mean()
    loso = pd.read_csv(results / "tables" / "loso.csv")
    loso = loso[loso.protocol == "loso"].groupby("model").weighted_f1.mean()
    d = drop.join(loso.rename("loso")).sort_values("loso")
    fig, ax = _fig(7.5, 3.6)
    _style(ax)
    y = np.arange(len(d))
    for i, (_, r) in enumerate(d.iterrows()):
        ax.plot([r.transfer_mean, r.indomain], [i, i], color=GRID, linewidth=2, zorder=1)
    marks = [("indomain", "Same source (5-fold CV)", SERIES[0]), ("transfer_mean", "Trained on one other source", SERIES[1]),
             ("loso", "Trained on all other sources", SERIES[2])]
    for col, label, color in marks:
        ax.scatter(d[col], y, s=46, color=color, edgecolor=SURFACE, linewidth=1.5, zorder=3, label=label)
    for i, (_, r) in enumerate(d.iterrows()):
        if abs(r.indomain - r.transfer_mean) < 0.01:
            ax.annotate(f"no drop: {r.loso:.2f} in all three settings", (r.loso, i), xytext=(10, 0),
                        textcoords="offset points", va="center", fontsize=8, color=INK2)
    ax.set_yticks(y, [PRETTY.get(m, m) for m in d.index], color=INK, fontsize=9)
    ax.set_xlabel("Weighted F1 (mean over 6 sources)", color=INK2, fontsize=9)
    ax.set_xlim(0.3, 0.85)
    ax.legend(loc="lower right", frameon=False, fontsize=8, labelcolor=INK2)
    ax.set_title("Accuracy on a source the model was not trained on", loc="left", color=INK, fontsize=11)
    fig.tight_layout()
    fig.savefig(out / "fig_drop.png", facecolor=SURFACE)
    plt.close(fig)


def fig_adapt(results: Path, out: Path) -> None:
    t = pd.read_csv(results / "tables" / "adapt.csv")
    g = t.groupby(["target", "k"]).weighted_f1.mean().unstack(0)
    keep = [c for c in ["indian_engineering", "indian_boards", "gani2023", "scaria2024", "sangodiah2017"] if c in g]
    fig, ax = _fig(7.0, 3.8)
    _style(ax, "y")
    for c, color in zip(keep, SERIES):
        ax.plot(g.index, g[c], color=color, linewidth=2, marker="o", markersize=5, markeredgecolor=SURFACE)
        ax.annotate(f"{SRC.get(c, c)}  {g[c].iloc[-1]:.2f}", (g.index[-1], g[c].iloc[-1]), xytext=(6, 0),
                    textcoords="offset points", va="center", fontsize=8, color=INK2)
    ax.set_xticks(g.index)
    ax.set_xlabel("Questions labelled by the institution (k)", color=INK2, fontsize=9)
    ax.set_ylabel("Weighted F1 on its other questions", color=INK2, fontsize=9)
    ax.set_xlim(g.index.min() - 3, g.index.max() * 1.35)
    ax.set_title("k-shot adaptation to a new institution's labelling conventions", loc="left", color=INK, fontsize=11)
    fig.tight_layout()
    fig.savefig(out / "fig_adapt.png", facecolor=SURFACE)
    plt.close(fig)


def fig_concept_shift(root: Path, out: Path, verbs=("explain", "describe", "design", "write")) -> pd.DataFrame:
    from bloombot.data.build import load_questions
    from bloombot.text.linguistics import default_analyzer

    df = load_questions(root)
    df["verb"] = [a.verbs[0] if a.verbs else None for a in default_analyzer().batch(df.question.tolist())]
    fig, axes = plt.subplots(1, len(verbs), figsize=(3.1 * len(verbs), 3.3), dpi=160, sharey=False)
    fig.patch.set_facecolor(SURFACE)
    tables = []
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("blues", [SURFACE] + BLUES)
    for ax, v in zip(axes, verbs):
        sub = df[df.verb == v]
        ct = pd.crosstab(sub.source, sub.label, normalize="index").reindex(columns=list(LABELS), fill_value=0)
        n = sub.source.value_counts()
        ct = ct[n.reindex(ct.index) >= 15]
        ct.index = [SRC.get(s, s) for s in ct.index]
        ax.imshow(ct.values, cmap=cmap, vmin=0, vmax=1, aspect="auto")
        for i in range(ct.shape[0]):
            for j in range(ct.shape[1]):
                val = ct.values[i, j]
                if val >= 0.05:
                    ax.text(j, i, f"{val:.0%}", ha="center", va="center", fontsize=7, color=SURFACE if val > 0.55 else INK)
        ax.set_xticks(range(6), [lab[:3] for lab in LABELS], fontsize=8, color=INK2)
        ax.set_yticks(range(len(ct)), ct.index, fontsize=8, color=INK)
        ax.set_title(f'"{v.capitalize()} ..."', fontsize=10, color=INK, loc="left")
        for s in ax.spines.values():
            s.set_visible(False)
        ax.tick_params(length=0)
        tables.append(ct.assign(verb=v))
    fig.suptitle("The same directive verb is labelled differently by different sources (share of questions per level)",
                 x=0.01, ha="left", fontsize=10, color=INK)
    fig.tight_layout()
    fig.savefig(out / "fig_concept_shift.png", facecolor=SURFACE)
    plt.close(fig)
    return pd.concat(tables)


def fig_risk(results: Path, out: Path) -> None:
    files = sorted((results / "tables").glob("risk_coverage_*.csv"))
    if not files:
        return
    rc = pd.read_csv(files[0])
    fig, ax = _fig(5.5, 3.4)
    _style(ax, "y")
    ax.plot(rc.coverage, rc.accuracy, color=SERIES[0], linewidth=2, marker="o", markersize=4, markeredgecolor=SURFACE)
    ax.plot(rc.coverage, rc.within_one, color=SERIES[1], linewidth=2, marker="o", markersize=4, markeredgecolor=SURFACE)
    ax.annotate("exact level", (rc.coverage.iloc[-1], rc.accuracy.iloc[-1]), xytext=(6, -2), textcoords="offset points",
                fontsize=8, color=INK2)
    ax.annotate("within one level", (rc.coverage.iloc[-1], rc.within_one.iloc[-1]), xytext=(6, -2),
                textcoords="offset points", fontsize=8, color=INK2)
    ax.invert_xaxis()
    ax.set_xlabel("Share of questions labelled automatically (rest go to the teacher)", color=INK2, fontsize=9)
    ax.set_ylabel("Accuracy on those questions", color=INK2, fontsize=9)
    ax.set_title("Abstaining on low confidence helps only a little across sources", loc="left", color=INK, fontsize=10)
    fig.tight_layout()
    fig.savefig(out / "fig_risk_coverage.png", facecolor=SURFACE)
    plt.close(fig)


def make_all(root: Path, results: Path, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    fig_drop(results, out)
    if (results / "tables" / "adapt.csv").exists():
        fig_adapt(results, out)
    fig_risk(results, out)
    ct = fig_concept_shift(root, out)
    ct.to_csv(results / "tables" / "concept_shift.csv")
