#!/usr/bin/env python
"""
make_fig2_histogram.py  -- standalone recreation of paper Figure 2.

Fig2 = histogram of "years from eye measurement to AD diagnosis".
Originally this lived inline in main.ipynb (code cell #13). Extracted here so it
is easy to find and re-run on its own.

  Input : ad_years.csv        (column "ad_after0"; produced by the cohort/cox prep)
  Output: ad_hist.pdf + ad_hist.png   (ad_hist.pdf is copied to paper Fig2.pdf)

Run:  python make_fig2_histogram.py        (from the "Fix Imbalance" folder)

Style is Springer/GeroScience-compliant: Arial embedded as Type-42 TrueType,
no Type-3, NORMAL-weight axis labels.
"""

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")  # headless; remove if running interactively
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

# --- Publication figure style (self-contained; no functions.py needed) ---
plt.rcParams.update(
    {
        "text.usetex": False,
        "pdf.fonttype": 42,  # embed text as Type-42 TrueType (no Type-3)
        "ps.fonttype": 42,
        "svg.fonttype": "none",
        "font.family": "sans-serif",
        # Arial first; Arial-metric / sans-serif fallbacks so it stays compliant
        # even without mscorefonts installed.
        "font.sans-serif": [
            "Arial",
            "Liberation Sans",
            "Helvetica",
            "Nimbus Sans",
            "DejaVu Sans",
        ],
        "font.size": 13,
        "axes.linewidth": 0.8,
        "axes.edgecolor": "#444444",
        "axes.labelweight": "normal",  # axis labels NOT bold
        "axes.labelsize": 19,  # larger, readable axis labels
        "xtick.labelsize": 18,  # larger tick labels (x-interval labels below are set to 11 explicitly)
        "ytick.labelsize": 18,
    }
)


ad_years = pd.read_csv("ad_years.csv")
dd = ad_years["ad_after0"]


BAR_COLOR = "#2C6E8F"
EDGE_COLOR = "#1B4A61"
MEAN_COLOR = "#B5452F"

# ----------------------------------------------------------------------
# BINS  (left-closed: [0,2), [2,4), ...)
# ----------------------------------------------------------------------
min_val = int(np.floor(dd.min()))
max_val = int(np.ceil(dd.max()))
# round the lower edge down to an even number so bins align on even years
min_edge = min_val - (min_val % 2)
bins = np.arange(min_edge, max_val + 2, 2)

counts, edges = np.histogram(dd, bins=bins)
centers = edges[:-1] + np.diff(edges) / 2
mean_val = dd.mean()
std_val = dd.std()

# ----------------------------------------------------------------------
# PLOT
# ----------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(8, 5), dpi=150)

bars = ax.bar(
    centers,
    counts,
    width=np.diff(edges) * 0.92,
    color=BAR_COLOR,
    edgecolor=EDGE_COLOR,
    linewidth=0.8,
    zorder=3,
)

# value labels on top of each bar
for c, n in zip(centers, counts):
    if n > 0:
        ax.text(
            c,
            n + max(counts) * 0.015,
            str(int(n)),
            ha="center",
            va="bottom",
            fontsize=11,
            color="#333333",
            zorder=4,
        )

# mean line + label
ax.axvline(mean_val, color=MEAN_COLOR, linestyle=(0, (5, 3)), linewidth=1.6, zorder=5)
ax.text(
    mean_val,
    max(counts) * 1.2,
    f"  mean = {mean_val:.2f} yr\n  (SD = {std_val:.2f})",
    ha="left",
    va="top",
    fontsize=11,
    color=MEAN_COLOR,
    zorder=6,
)

# axis labels (real meaning, not the variable name)
ax.set_xlabel("Years from eye measurement to AD diagnosis", labelpad=8)
ax.set_ylabel("Number of AD cases", labelpad=8)

# x ticks: bin-edge interval labels, centered under each bar, horizontal
edge_labels = [f"[{int(edges[i])}, {int(edges[i + 1])})" for i in range(len(edges) - 1)]
ax.set_xticks(centers)
ax.set_xticklabels(edge_labels, fontsize=11)

# y ticks every 20, light minor ticks every 10
ax.yaxis.set_major_locator(MultipleLocator(20))
ax.yaxis.set_minor_locator(MultipleLocator(10))
ax.set_ylim(0, max(counts) * 1.18)
ax.set_xlim(edges[0] - 1, edges[-1] + 1)

# grid behind bars only on y
ax.yaxis.grid(True, which="major", color="#000000", alpha=0.10, linewidth=0.7, zorder=0)
ax.yaxis.grid(True, which="minor", color="#000000", alpha=0.05, linewidth=0.5, zorder=0)
ax.set_axisbelow(True)

# despine top/right
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.tick_params(axis="both", color="#444444", length=4, width=0.8)
ax.tick_params(axis="x", length=0)  # no x tick marks; labels are intervals

fig.tight_layout()
fig.savefig("ad_hist.pdf", bbox_inches="tight")
fig.savefig("ad_hist.png", bbox_inches="tight", dpi=200)
print(f"counts = {counts.tolist()}")
print(f"mean = {mean_val:.3f}, sd = {std_val:.3f}, n = {len(dd)}")
