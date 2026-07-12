"""Render the hidden-positive injection backtest as a presentation-quality chart.

    PYTHONPATH=. python3 scripts/plot_injection.py

Writes out/injection.png — the headline slide: naive hard-negative mining picks
~3 of 4 hidden real interactions as "negatives"; negaverse's default catches ~100%.
Numbers are the 3-seed means from scripts/bench_corrected.py --injection-test.
"""
from __future__ import annotations
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager  # noqa: F401

# 3-seed means, K=1000 injected hidden positives (bench_corrected.py --injection-test)
DATA = {
    "HuRI (human, dense)": {
        "topology_hard\n(naive hard mining)": 74.6,
        "random_veto\n(screened random)": 7.6,
        "stacked\n(negaverse default)": 0.6,
    },
    "DRYAD (sparse)": {
        "topology_hard\n(naive hard mining)": 64.3,
        "random_veto\n(screened random)": 0.8,
        "stacked\n(negaverse default)": 0.0,
    },
}
COLORS = {"topology_hard\n(naive hard mining)": "#e2483d",   # danger red
          "random_veto\n(screened random)": "#b7b0a8",       # muted grey
          "stacked\n(negaverse default)": "#1f9d6b"}         # go green

plt.rcParams.update({"font.size": 13, "font.family": "DejaVu Sans",
                     "axes.edgecolor": "#cccccc", "svg.fonttype": "none"})
fig, axes = plt.subplots(1, 2, figsize=(13, 5.6), sharey=True)
fig.suptitle("Hidden-positive injection backtest — what % of 1,000 real interactions\n"
             "does each strategy wrongly label “negative”?",
             fontsize=16, fontweight="bold", y=1.02)

for ax, (title, arms) in zip(axes, DATA.items()):
    labels = list(arms)
    vals = [arms[k] for k in labels]
    bars = ax.bar(range(len(labels)), vals, width=0.62,
                  color=[COLORS[k] for k in labels], zorder=3)
    for i, (b, v) in enumerate(zip(bars, vals)):
        ax.text(b.get_x() + b.get_width() / 2, v + 1.6, f"{v:.1f}%",
                ha="center", va="bottom", fontsize=14, fontweight="bold",
                color=COLORS[labels[i]])
    ax.set_title(title, fontsize=14, fontweight="bold", pad=10)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=10.5)
    ax.set_ylim(0, 85)
    ax.grid(axis="y", color="#eeeeee", zorder=0)
    ax.spines[["top", "right"]].set_visible(False)

axes[0].set_ylabel("hidden interactions mislabeled as negative  (lower = better)", fontsize=11.5)
fig.text(0.5, -0.05,
         "Lower is better ·  negaverse's default (stacked) lets ~0 through; naive hard mining poisons "
         "training data with 3 of every 4 hidden positives.\n"
         "Selection is model-independent (it precedes any learner). 3-seed means.",
         ha="center", fontsize=10.5, color="#555555")

out = Path("out"); out.mkdir(exist_ok=True)
fig.savefig(out / "injection.png", dpi=200, bbox_inches="tight", facecolor="white")
print("wrote", out / "injection.png")


# The interactive, drag-to-rotate 3D version of this chart lives in the showcase
# page — build it with scripts/build_showcase.py (Plotly). See negaverse/viz/showcase.py.
