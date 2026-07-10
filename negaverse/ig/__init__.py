"""Information-geometry (IG) prototypes ported from sinain-hud.

Each module here is a small, self-contained mechanism transferred from the
"Memory as Geometry" lecture into the negaverse negative-selection setting.
They are *prototypes*: pure functions with an evaluation harness
(`scripts/eval_ig_features.py`), gated into the pipeline only where the change
is non-breaking (entropy-weighted fusion). See docs/IG-FEATURES.md.

  entropy_fusion — Ch4: trust each stream in proportion to how decisive it is.
  dpp            — Ch5: pick a negative set that *spans* the space, not clones.
  surprisal      — Ch1: score a pair by resemblance to a frozen background cloud
                   (gold negatives = safe; positive manifold = suspected FN).
  margin         — Ch7: gate on the relative margin between two exemplar clouds,
                   not an absolute threshold.
"""
from __future__ import annotations

from .entropy_fusion import (
    binary_entropy,
    decisiveness,
    entropy_weighted_fuse,
    stream_disagreement,
)
from .dpp import greedy_map_dpp
from .surprisal import background_similarity, normalize_rows, topk_mean_sim
from .margin import margin_gate, margin_score

__all__ = [
    "binary_entropy",
    "decisiveness",
    "entropy_weighted_fuse",
    "stream_disagreement",
    "greedy_map_dpp",
    "background_similarity",
    "normalize_rows",
    "topk_mean_sim",
    "margin_gate",
    "margin_score",
]
