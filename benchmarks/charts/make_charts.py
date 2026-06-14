#!/usr/bin/env python3
"""make_charts.py — render the SAM/IA benchmark compendium figures.

What: hard-codes the five verified chart datasets drawn directly from the
  compendium's verified tables (no recomputation, no new numbers) and writes
  five PNGs to this directory using matplotlib's non-interactive Agg backend.
Why: the compendium is a static markdown record; embedding rendered figures
  makes the equalized 2x2, the temporal collapse-vs-hold, the recall@k envelope
  and the embedder-slot study legible at a glance. Every value below is copied
  verbatim from SAMIA_BENCHMARK_COMPENDIUM_2026-06-13.md — this script renders,
  it does not compute.

Graceful degradation: if matplotlib is not importable, this script writes the
  same data to chart_data.csv and exits 0 with a clear message, so the wider
  documentation task does not fail on a missing optional dependency. Re-run
  after `pip install matplotlib` to produce the PNGs.

Usage:
  python3 make_charts.py
Outputs (this directory):
  equalized_2x2.png, category_frontier.png, temporal_degradation.png,
  temporal_recall_at_k.png, embedder_sweep.png
  (or chart_data.csv if matplotlib is unavailable)
"""
import csv
import os
import sys

OUT_DIR = os.path.dirname(os.path.abspath(__file__))

# --- VERIFIED chart data (verbatim from the compendium; do NOT alter) --------

# 1. Equalized 2x2 — overall accuracy by extractor tier (LoCoMo, n=100).
EQ_GROUPS = ["4B extractor", "frontier extractor"]
EQ_SAMIA = [0.47, 0.59]
EQ_MEM0 = [0.37, 0.48]

# 2. Per-category accuracy at the frontier extractor (n=20/category).
CAT_LABELS = ["multihop", "temporal", "open-domain", "single-hop", "adversarial"]
CAT_SAMIA_FRONTIER = [0.75, 0.70, 0.45, 0.90, 0.15]
CAT_MEM0_FRONTIER = [0.55, 0.55, 0.35, 0.80, 0.15]

# 3. Temporal accuracy by extractor — the collapse-vs-hold.
TD_GROUPS = ["4B extractor", "frontier extractor"]
TD_SAMIA = [0.50, 0.70]
TD_MEM0 = [0.10, 0.55]

# 4. Temporal recall@k (calibration corpus v3_balanced; intrinsic recall, no LLM).
RK_K = [1, 5, 10]
RK_FLAG_OFF = [0.20, 0.84, 1.00]
RK_BEST_ENVELOPE = [0.60, 0.90, 1.00]

# 5. Embedder sweep — overall LoCoMo accuracy by embedder slot.
EMB_LABELS = ["MiniLM-L6", "bge-small", "mpnet-base", "bge-large", "qwen3-4b-gguf"]
EMB_OVERALL = [0.47, 0.48, 0.47, 0.49, 0.10]

# Colorblind-safe palette (Wong / Okabe-Ito derived).
C_SAMIA = "#0072B2"   # blue
C_MEM0 = "#D55E00"    # vermillion
C_OFF = "#999999"     # neutral grey
C_BEST = "#009E73"    # bluish green
C_NEUTRAL = "#56B4E9" # sky blue


def _csv_fallback():
    """Write all five datasets to chart_data.csv when matplotlib is absent."""
    path = os.path.join(OUT_DIR, "chart_data.csv")
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["chart", "series", "x", "value"])
        for g, s, m in zip(EQ_GROUPS, EQ_SAMIA, EQ_MEM0):
            w.writerow(["equalized_2x2", "SAM/IA", g, s])
            w.writerow(["equalized_2x2", "mem0", g, m])
        for c, s, m in zip(CAT_LABELS, CAT_SAMIA_FRONTIER, CAT_MEM0_FRONTIER):
            w.writerow(["category_frontier", "SAM/IA-frontier", c, s])
            w.writerow(["category_frontier", "mem0-frontier", c, m])
        for g, s, m in zip(TD_GROUPS, TD_SAMIA, TD_MEM0):
            w.writerow(["temporal_degradation", "SAM/IA", g, s])
            w.writerow(["temporal_degradation", "mem0", g, m])
        for k, off, best in zip(RK_K, RK_FLAG_OFF, RK_BEST_ENVELOPE):
            w.writerow(["temporal_recall_at_k", "flag-off", k, off])
            w.writerow(["temporal_recall_at_k", "best-envelope", k, best])
        for lab, v in zip(EMB_LABELS, EMB_OVERALL):
            w.writerow(["embedder_sweep", "overall", lab, v])
    return path


def _label_bars(ax, bars, fmt="{:.2f}"):
    for b in bars:
        h = b.get_height()
        ax.annotate(fmt.format(h),
                    xy=(b.get_x() + b.get_width() / 2, h),
                    xytext=(0, 3), textcoords="offset points",
                    ha="center", va="bottom", fontsize=9)


def _save(fig, name):
    path = os.path.join(OUT_DIR, name)
    fig.savefig(path, dpi=130, bbox_inches="tight")
    return path


def render_all(plt, np):
    import numpy as _np  # noqa: F401  (np passed in already)
    written = []

    # 1. Equalized 2x2 grouped bars.
    fig, ax = plt.subplots(figsize=(7, 4.5))
    x = np.arange(len(EQ_GROUPS))
    w = 0.36
    b1 = ax.bar(x - w / 2, EQ_SAMIA, w, label="SAM/IA", color=C_SAMIA)
    b2 = ax.bar(x + w / 2, EQ_MEM0, w, label="mem0", color=C_MEM0)
    _label_bars(ax, b1)
    _label_bars(ax, b2)
    ax.set_title("Equalized 2x2: SAM/IA vs mem0 by extractor tier\n"
                 "LoCoMo, n=100, llama-3.3-70b reader+judge")
    ax.set_ylabel("Overall accuracy")
    ax.set_xlabel("Extraction model (same model for both systems within a tier)")
    ax.set_xticks(x)
    ax.set_xticklabels(EQ_GROUPS)
    ax.set_ylim(0, 0.75)
    ax.legend(loc="upper left")
    ax.grid(axis="y", alpha=0.3)
    written.append(_save(fig, "equalized_2x2.png"))
    plt.close(fig)

    # 2. Per-category at the frontier extractor.
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    x = np.arange(len(CAT_LABELS))
    w = 0.38
    b1 = ax.bar(x - w / 2, CAT_SAMIA_FRONTIER, w, label="SAM/IA (frontier)", color=C_SAMIA)
    b2 = ax.bar(x + w / 2, CAT_MEM0_FRONTIER, w, label="mem0 (frontier)", color=C_MEM0)
    _label_bars(ax, b1)
    _label_bars(ax, b2)
    ax.set_title("Per-category accuracy at the frontier extractor\n"
                 "LoCoMo, n=20 per category")
    ax.set_ylabel("Accuracy")
    ax.set_xlabel("Question category")
    ax.set_xticks(x)
    ax.set_xticklabels(CAT_LABELS)
    ax.set_ylim(0, 1.05)
    ax.legend(loc="upper right")
    ax.grid(axis="y", alpha=0.3)
    written.append(_save(fig, "category_frontier.png"))
    plt.close(fig)

    # 3. Temporal degradation (collapse vs hold).
    fig, ax = plt.subplots(figsize=(7, 4.5))
    x = np.arange(len(TD_GROUPS))
    w = 0.36
    b1 = ax.bar(x - w / 2, TD_SAMIA, w, label="SAM/IA", color=C_SAMIA)
    b2 = ax.bar(x + w / 2, TD_MEM0, w, label="mem0", color=C_MEM0)
    _label_bars(ax, b1)
    _label_bars(ax, b2)
    ax.set_title("Temporal accuracy by extractor: graceful hold vs collapse\n"
                 "SAM/IA holds 0.70->0.50 (-20); mem0 collapses 0.55->0.10 (-45)")
    ax.set_ylabel("Temporal-category accuracy")
    ax.set_xlabel("Extraction model")
    ax.set_xticks(x)
    ax.set_xticklabels(TD_GROUPS)
    ax.set_ylim(0, 0.85)
    ax.legend(loc="upper left")
    ax.grid(axis="y", alpha=0.3)
    written.append(_save(fig, "temporal_degradation.png"))
    plt.close(fig)

    # 4. Temporal recall@k line chart.
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(RK_K, RK_FLAG_OFF, marker="o", color=C_OFF, label="flag-off (baseline)")
    ax.plot(RK_K, RK_BEST_ENVELOPE, marker="s", color=C_BEST, label="best envelope")
    for k, v in zip(RK_K, RK_FLAG_OFF):
        ax.annotate(f"{v:.2f}", (k, v), textcoords="offset points",
                    xytext=(0, -14), ha="center", fontsize=9, color=C_OFF)
    for k, v in zip(RK_K, RK_BEST_ENVELOPE):
        ax.annotate(f"{v:.2f}", (k, v), textcoords="offset points",
                    xytext=(0, 7), ha="center", fontsize=9, color=C_BEST)
    ax.set_title("Temporal recall@k: multiplicative need/STC/distinctiveness envelope\n"
                 "calibration corpus v3_balanced (intrinsic recall, no LLM judge)")
    ax.set_ylabel("recall@k")
    ax.set_xlabel("k")
    ax.set_xticks(RK_K)
    ax.set_ylim(0, 1.08)
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)
    written.append(_save(fig, "temporal_recall_at_k.png"))
    plt.close(fig)

    # 5. Embedder sweep bar chart (highlight the non-viable outlier).
    fig, ax = plt.subplots(figsize=(8, 4.6))
    x = np.arange(len(EMB_LABELS))
    colors = [C_NEUTRAL] * (len(EMB_LABELS) - 1) + [C_MEM0]
    bars = ax.bar(x, EMB_OVERALL, 0.6, color=colors)
    _label_bars(ax, bars)
    ax.set_title("Embedder-slot sweep: overall LoCoMo accuracy\n"
                 "slot is ~flat (<=+0.02); qwen3-4b mean-pool is non-viable (anisotropic)")
    ax.set_ylabel("Overall accuracy")
    ax.set_xlabel("Embedder")
    ax.set_xticks(x)
    ax.set_xticklabels(EMB_LABELS, rotation=15, ha="right")
    ax.set_ylim(0, 0.62)
    ax.grid(axis="y", alpha=0.3)
    written.append(_save(fig, "embedder_sweep.png"))
    plt.close(fig)

    return written


def main():
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except Exception as e:  # noqa: BLE001
        path = _csv_fallback()
        print(f"[make_charts] matplotlib unavailable ({e!r}).")
        print(f"[make_charts] wrote data table instead: {path}")
        print("[make_charts] install matplotlib and re-run to produce the PNGs.")
        return 0

    written = render_all(plt, np)
    print("[make_charts] wrote:")
    for p in written:
        print("  " + p)
    return 0


if __name__ == "__main__":
    sys.exit(main())

# ---
# File: charts/make_charts.py
# Purpose: render the 5 compendium figures (Agg backend) or fall back to CSV.
# Inputs: hard-coded verified chart data (verbatim from the compendium tables).
# Outputs: 5 PNGs in this directory, or chart_data.csv if matplotlib is absent.
