#!/usr/bin/env python3
"""
plot_eval.py - turn the captioning eval JSON into comparison figures.

    python plot_eval.py eval_CLEAN.json --out figures

Expected input (what the eval step writes):
    {"n_images": N,
     "models": {<key>: {"corpus": {...}, "diagnostics": {...},
                        "clip_mean": {...}, "chair": {...}}}}

Figures written to --out:
    01_scorecard.png    every metric x model, coloured relative to the best model
    02_reference.png    BLEU-n decay, METEOR / ROUGE-L, CIDEr
    03_grounding.png    hallucination (CHAIR) and image-text agreement (CLIPScore)
    04_diversity.png    distinct-n vs absolute unique n-grams, duplicate captions
    05_length.png       caption-length spread vs the reference length
    06_tradeoff.png     CIDEr vs CHAIR-s (swap the axes for VRAM / latency later)
    metrics_long.csv    the tidy table every figure is drawn from

Example command:
    $ python evaluation/plot_eval.py evaluation/results/eval_CLEAN.json --out figures
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Patch

# --------------------------------------------------------------------------
# Fixed identity: display name, order and colour per model. Colour follows the
# model in every figure, so "BLIP is orange" is learned once.
# --------------------------------------------------------------------------
MODELS = {"vit_gpt2": "ViT-GPT2", "blip": "BLIP-base", "git": "GIT-base"}
PALETTE = {"ViT-GPT2": "#2a78d6", "BLIP-base": "#eb6834", "GIT-base": "#1baf7a"}
ORDER = list(PALETTE)

SURFACE, INK, INK_2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
BLUES = ["#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"]  # sequential ramp

# metric key -> (label, group, higher_is_better). Order here = order on the scorecard.
SCORECARD = {
    "Bleu_1": ("BLEU-1", "n-gram overlap\nwith references", True),
    "Bleu_2": ("BLEU-2", "n-gram overlap\nwith references", True),
    "Bleu_3": ("BLEU-3", "n-gram overlap\nwith references", True),
    "Bleu_4": ("BLEU-4", "n-gram overlap\nwith references", True),
    "METEOR": ("METEOR", "n-gram overlap\nwith references", True),
    "ROUGE_L": ("ROUGE-L", "n-gram overlap\nwith references", True),
    "CIDEr": ("CIDEr", "n-gram overlap\nwith references", True),
    "CLIPScore": ("CLIPScore", "image-text\nembedding", True),
    "RefCLIPScore": ("RefCLIPScore", "image-text\nembedding", True),
    "CHAIR_i": ("CHAIR-i", "hallucination", False),
    "CHAIR_s": ("CHAIR-s", "hallucination", False),
    "vocab_size": ("Vocab size", "diversity", True),
    "distinct_1": ("distinct-1", "diversity", True),
    "distinct_2": ("distinct-2", "diversity", True),
    "dup_caption_rate": ("Duplicate captions", "diversity", False),
}


def set_theme() -> None:
    sns.set_theme(
        context="notebook",
        style="whitegrid",
        rc={
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "axes.edgecolor": GRID,
            "grid.color": GRID,
            "grid.linewidth": 0.8,
            "axes.labelcolor": INK_2,
            "xtick.color": INK_2,
            "ytick.color": INK_2,
            "text.color": INK,
            "axes.titlesize": 12,
            "axes.titleweight": "bold",
            "axes.titlelocation": "left",
            "axes.titlepad": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
        },
    )


# --------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------
def load(path: Path) -> tuple[pd.DataFrame, pd.DataFrame, int]:
    """Return (long, wide, n_images).

    long: one row per (model, section, metric) - the "tidy" shape seaborn wants.
    wide: one row per model, one column per metric - handy for arithmetic.
    """
    raw = json.loads(path.read_text())
    n = int(raw["n_images"])
    rows = []
    for key, sections in raw["models"].items():
        model = MODELS.get(key, key)
        for section, metrics in sections.items():
            for metric, value in metrics.items():
                rows.append({"model": model, "section": section, "metric": metric, "value": float(value)})
    long = pd.DataFrame(rows)
    missing = set(long["model"]) - set(ORDER)
    if missing:
        raise SystemExit(f"Add a display name and colour for: {sorted(missing)}")
    wide = long.pivot(index="model", columns="metric", values="value").reindex(ORDER)
    return long, wide, n


def wilson(k: np.ndarray, n: int, z: float = 1.96) -> tuple[np.ndarray, np.ndarray]:
    """95% Wilson score interval for a proportion k/n (better than p +- 1.96*SE near 0)."""
    p = k / n
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return centre - half, centre + half


def add_derived(wide: pd.DataFrame, n: int) -> pd.DataFrame:
    """Recover counts hidden inside the ratios.

    distinct-1 = unique unigrams / total tokens, so total tokens = len_mean * n.
    distinct-2 counts bigrams inside each caption, so total bigrams = tokens - n.
    CHAIR-s and the two *_rate fields are fractions of the n captions, so they
    are binomial proportions and get a Wilson CI straight from the summary.
    """
    d = wide.copy()
    d["total_tokens"] = (d["len_mean"] * n).round()
    d["unique_unigrams"] = d["vocab_size"]
    d["unique_bigrams"] = (d["distinct_2"] * (d["total_tokens"] - n)).round()

    check = (d["vocab_size"] / d["total_tokens"] - d["distinct_1"]).abs().max()
    if check > 1e-9:
        print(f"[warn] distinct-1 != vocab/tokens (max gap {check:.2e}); token counts are approximate")

    for col in ["CHAIR_s", "dup_caption_rate", "exact_ref_match_rate"]:
        k = (d[col] * n).round()
        d[f"{col}_k"] = k
        d[f"{col}_lo"], d[f"{col}_hi"] = wilson(k.to_numpy(), n)
    return d


def melt(wide: pd.DataFrame, cols: list[str], labels: dict[str, str] | None = None) -> pd.DataFrame:
    out = wide[cols].rename(columns=labels or {}).reset_index()
    return out.melt(id_vars="model", var_name="metric", value_name="value")


# --------------------------------------------------------------------------
# Small helpers
# --------------------------------------------------------------------------
def model_legend(fig: plt.Figure, y: float = 1.02) -> None:
    handles = [Patch(facecolor=PALETTE[m], label=m) for m in ORDER]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, y),
               ncol=len(ORDER), frameon=False, handlelength=1.2, columnspacing=1.6)


def single_metric_bars(ax, d: pd.DataFrame, col: str, title: str, fmt: str,
                       ci: bool = False, scale: float = 1.0, ylabel: str = "") -> None:
    """One bar per model, value written above the bar (or above its CI)."""
    vals = d[col].to_numpy() * scale
    sns.barplot(x=ORDER, y=vals, hue=ORDER, palette=PALETTE, order=ORDER,
                legend=False, width=0.62, saturation=1, ax=ax)
    top = vals.copy()
    if ci:
        lo, hi = d[f"{col}_lo"].to_numpy() * scale, d[f"{col}_hi"].to_numpy() * scale
        ax.errorbar(np.arange(len(vals)), vals, yerr=[vals - lo, hi - vals],
                    fmt="none", ecolor=INK_2, elinewidth=1.2, capsize=4)
        top = hi
    pad = 0.02 * top.max()
    for i, (v, t) in enumerate(zip(vals, top)):
        ax.text(i, t + pad, format(v, fmt), ha="center", va="bottom", fontsize=10, color=INK)
    ax.set_ylim(0, top.max() * 1.18)
    ax.set(title=title, xlabel="", ylabel=ylabel)


def save(fig: plt.Figure, out: Path, name: str) -> None:
    path = out / name
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {path}")


# --------------------------------------------------------------------------
# Figures
# --------------------------------------------------------------------------
def fig_scorecard(d: pd.DataFrame, out: Path) -> None:
    """Every metric on one grid. Colour = score / best score on that row
    (inverted for lower-is-better rows), so 1.0 is always "best" and a
    0.4% difference stays a 0.4% difference - unlike min-max scaling."""
    keys = [k for k in SCORECARD if k in d.columns]
    raw = d[keys].T  # rows = metrics, cols = models
    rel = raw.copy()
    for k in keys:
        hib = SCORECARD[k][2]
        rel.loc[k] = raw.loc[k] / raw.loc[k].max() if hib else raw.loc[k].min() / raw.loc[k]

    annot = raw.map(lambda v: f"{v:.0f}" if v >= 100 else f"{v:.3f}")
    rel.index = [f"{SCORECARD[k][0]}  {'↑' if SCORECARD[k][2] else '↓'}" for k in keys]

    cmap = LinearSegmentedColormap.from_list("blues", BLUES)
    fig, ax = plt.subplots(figsize=(7.2, 8.6))
    sns.heatmap(rel, annot=annot, fmt="", cmap=cmap, vmin=0.5, vmax=1.0,
                linewidths=2, linecolor=SURFACE, annot_kws={"fontsize": 10},
                cbar_kws={"orientation": "horizontal", "shrink": 0.55, "pad": 0.03,
                          "label": "score relative to the best model on that row (1.0 = best)"},
                ax=ax)
    ax.xaxis.tick_top()
    ax.tick_params(axis="x", length=0, labelsize=11)
    ax.tick_params(axis="y", length=0, labelsize=10)
    ax.set(xlabel="", ylabel="")

    groups = [SCORECARD[k][1] for k in keys]
    start = 0
    for i in range(1, len(groups) + 1):
        if i == len(groups) or groups[i] != groups[start]:
            if i < len(groups):
                ax.axhline(i, color=SURFACE, lw=7)
            ax.text(len(raw.columns) + 0.12, (start + i) / 2, groups[start],
                    va="center", ha="left", fontsize=9, color=INK_2, linespacing=1.2)
            start = i
    ax.set_title("Scorecard  ·  cell text = raw value, colour = relative to best",
                 pad=34, fontsize=12)
    save(fig, out, "01_scorecard.png")


def fig_reference(d: pd.DataFrame, out: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.4), gridspec_kw={"width_ratios": [1.35, 1.1, 0.9]})

    bleu = melt(d, [f"Bleu_{i}" for i in range(1, 5)])
    bleu["n"] = bleu["metric"].str[-1].astype(int)
    sns.lineplot(data=bleu, x="n", y="value", hue="model", hue_order=ORDER, palette=PALETTE,
                 marker="o", markersize=8, linewidth=2, legend=False, ax=axes[0])
    axes[0].set(title="BLEU-n: precision falls as n-grams get longer", xlabel="",
                ylabel="score", ylim=(0, 0.9), xticks=[1, 2, 3, 4])
    axes[0].set_xticklabels([f"BLEU-{i}" for i in range(1, 5)])

    mr = melt(d, ["METEOR", "ROUGE_L"], {"ROUGE_L": "ROUGE-L"})
    sns.barplot(data=mr, x="metric", y="value", hue="model", hue_order=ORDER, palette=PALETTE,
                legend=False, width=0.7, gap=0.12, saturation=1, ax=axes[1])
    axes[1].set(title="METEOR and ROUGE-L", xlabel="", ylabel="score", ylim=(0, 0.75))

    single_metric_bars(axes[2], d, "CIDEr", "CIDEr (papers report ×100)", ".3f", ylabel="score")
    axes[2].tick_params(axis="x", labelsize=9)

    fig.tight_layout(w_pad=2.5)
    model_legend(fig)
    save(fig, out, "02_reference.png")


def fig_grounding(d: pd.DataFrame, out: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.4), gridspec_kw={"width_ratios": [1, 1, 1.25]})

    single_metric_bars(axes[0], d, "CHAIR_s", "CHAIR-s  ↓\ncaptions with ≥1 hallucinated object",
                       ".1f", ci=True, scale=100, ylabel="% of captions (95% CI)")
    single_metric_bars(axes[1], d, "CHAIR_i", "CHAIR-i  ↓\nhallucinated / mentioned objects",
                       ".1f", scale=100, ylabel="% of object mentions")

    clip = melt(d, ["CLIPScore", "RefCLIPScore"])
    sns.stripplot(data=clip, x="metric", y="value", hue="model", hue_order=ORDER, palette=PALETTE,
                  dodge=True, jitter=False, size=10, linewidth=0, legend=False, ax=axes[2])
    axes[2].set(title="CLIPScore  ↑\n(axis does not start at 0)", xlabel="", ylabel="score",
                ylim=(0.55, 1.0))
    for ax in axes[:2]:
        ax.tick_params(axis="x", labelsize=9)

    fig.tight_layout(w_pad=2.5)
    model_legend(fig)
    save(fig, out, "03_grounding.png")


def fig_diversity(d: pd.DataFrame, out: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.4), gridspec_kw={"width_ratios": [1.1, 1.1, 1]})

    ratio = melt(d, ["distinct_1", "distinct_2"], {"distinct_1": "distinct-1", "distinct_2": "distinct-2"})
    sns.barplot(data=ratio, x="metric", y="value", hue="model", hue_order=ORDER, palette=PALETTE,
                legend=False, width=0.7, gap=0.12, saturation=1, ax=axes[0])
    axes[0].set(title="distinct-n\n= unique / total n-grams", xlabel="", ylabel="ratio")

    counts = melt(d, ["unique_unigrams", "unique_bigrams"],
                  {"unique_unigrams": "unique unigrams\n(= vocab size)", "unique_bigrams": "unique bigrams"})
    sns.barplot(data=counts, x="metric", y="value", hue="model", hue_order=ORDER, palette=PALETTE,
                legend=False, width=0.7, gap=0.12, saturation=1, ax=axes[1])
    axes[1].set(title="Unique n-grams\n(the numerators, as raw counts)", xlabel="", ylabel="count")

    single_metric_bars(axes[2], d, "dup_caption_rate", "Duplicate-caption rate  ↓\n(dup_caption_rate)",
                       ".1f", ci=True, scale=100, ylabel="% of captions (95% CI)")
    axes[2].tick_params(axis="x", labelsize=9)

    fig.tight_layout(w_pad=2.5)
    model_legend(fig)
    save(fig, out, "04_diversity.png")


def fig_length(d: pd.DataFrame, out: Path) -> None:
    """Only summary stats exist, so draw them directly: thin line = p5-p95,
    thick line = mean +- 1 sd, dot = mean."""
    fig, ax = plt.subplots(figsize=(8.5, 3.4))
    for i, m in enumerate(ORDER):
        r, c = d.loc[m], PALETTE[m]
        ax.hlines(i, r["len_p5"], r["len_p95"], color=c, lw=1.5)
        ax.hlines(i, r["len_mean"] - r["len_std"], r["len_mean"] + r["len_std"], color=c, lw=6, alpha=0.45)
        ax.plot(r["len_mean"], i, "o", color=c, ms=10, mec=SURFACE, mew=2)
        ax.text(r["len_p95"] + 0.25, i, f"mean {r['len_mean']:.1f}  ·  sd {r['len_std']:.1f}",
                va="center", fontsize=9, color=INK_2)

    ref = d["ref_len_mean"].iloc[0]
    ax.axvline(ref, color=INK_2, lw=1.2)
    ax.text(ref + 0.12, -0.72, f"reference mean {ref:.1f}", ha="left", va="center", fontsize=9, color=INK_2)
    ax.set_yticks(range(len(ORDER)), ORDER)
    ax.set_ylim(len(ORDER) - 0.5, -0.9)
    ax.set_xlim(5, 15.5)
    ax.set(xlabel="caption length (tokens)",
           title="Caption length  ·  thin = p5–p95, thick = mean ± 1 sd")
    ax.grid(axis="y", visible=False)
    ax.tick_params(axis="y", length=0)
    save(fig, out, "05_length.png")


def fig_tradeoff(d: pd.DataFrame, out: Path, x: str = "CIDEr", y: str = "CHAIR_s") -> None:
    """Quality vs hallucination. Re-use with x='peak_vram_gb' etc. once you log it."""
    fig, ax = plt.subplots(figsize=(6.8, 5))
    yv = d[y] * 100
    has_ci = f"{y}_lo" in d
    for m in ORDER:
        c = PALETTE[m]
        if has_ci:
            ax.errorbar(d.loc[m, x], yv[m],
                        yerr=[[yv[m] - d.loc[m, f"{y}_lo"] * 100], [d.loc[m, f"{y}_hi"] * 100 - yv[m]]],
                        fmt="none", ecolor=c, elinewidth=1.5, capsize=4, alpha=0.7)
    sns.scatterplot(x=d[x], y=yv, hue=d.index, hue_order=ORDER, palette=PALETTE, s=140,
                    edgecolor=SURFACE, linewidth=2, legend=False, ax=ax, zorder=3)
    for m in ORDER:
        ax.annotate(m, (d.loc[m, x], yv[m]), xytext=(10, 6), textcoords="offset points",
                    fontsize=10, color=INK)

    ax.set(xlabel=f"{x}  ↑", ylabel=f"{y.replace('_', '-')} %  ↓  (95% CI)",
           title="Quality vs hallucination  ·  best corner is bottom-right")
    xr, yr = d[x].max() - d[x].min(), yv.max() - yv.min()
    ax.set_xlim(d[x].min() - 0.25 * xr, d[x].max() + 0.3 * xr)
    ax.set_ylim(max(0, yv.min() - 0.6 * yr), yv.max() + 0.5 * yr)
    save(fig, out, "06_tradeoff.png")


# --------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("eval_json", type=Path)
    ap.add_argument("--out", type=Path, default=Path("figures"))
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    set_theme()
    long, wide, n = load(args.eval_json)
    d = add_derived(wide, n)
    long.to_csv(args.out / "metrics_long.csv", index=False)

    cols = ["CIDEr", "Bleu_4", "CLIPScore", "CHAIR_s", "CHAIR_s_lo", "CHAIR_s_hi",
            "total_tokens", "vocab_size", "unique_bigrams", "dup_caption_rate_k"]
    with pd.option_context("display.float_format", "{:.4f}".format, "display.width", 140):
        print(d[cols].T, "\n")

    fig_scorecard(d, args.out)
    fig_reference(d, args.out)
    fig_grounding(d, args.out)
    fig_diversity(d, args.out)
    fig_length(d, args.out)
    fig_tradeoff(d, args.out)


if __name__ == "__main__":
    main()