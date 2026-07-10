"""Phase 1 visualizations (demo-focused, matplotlib only).

Two stories the demo needs to tell:
  * separability — negaverse's hard negatives sit *closer to the positives* than
    random negatives do, on structural axes (common neighbours, shortest path).
    That's the "harder than random" claim, shown not asserted.
  * transparency — the hourglass funnel: how many candidates each stage drops /
    keeps, so a reviewer can see where negatives come from.

ESM2/MolFormer UMAP panels are intentionally deferred (embeddings are a separate,
heavier dependency; see docs/IMPLEMENTATION-PLAN.md).
"""
from .plots import (plot_separability, plot_funnel, plot_manifold,
                    plot_confidence_hardness, plot_flag_breakdown, render_all)
from .report import build_report

__all__ = ["plot_separability", "plot_funnel", "plot_manifold",
           "plot_confidence_hardness", "plot_flag_breakdown",
           "render_all", "build_report"]
