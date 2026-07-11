"""Layer 6 — fusion of the scoring streams into one heuristic score
(ARCHITECTURE.md §5 Fusion).

⚠️ NOT a calibrated probability. `confidence` here is an UNCALIBRATED weighted MEAN
of the non-abstaining streams' heuristic sub-scores. It is a monotone ranking
signal, not P(true negative). Two known consequences, both measured
(docs/FILTER-EFFECTIVENESS.md): a mean *dilutes* agreement (two streams that both
say "safe", e.g. 1.0 and 0.75, fuse to 0.875 < 1.0 instead of reinforcing), and the
value cannot be read as a probability or thresholded across datasets. Proper
cross-fitted calibration + a lower-confidence bound (the quantity that should
*authorise* a negative label) is the calibration arm; until then treat this as a
score, not a confidence.

Weighted combine of the streams that did *not* abstain; a hard veto from any
stream drops the pair entirely (never emitted as a negative). Each fused output
keeps the raw sub-scores so a user can see *why* a pair is a negative.
"""
from __future__ import annotations

from dataclasses import dataclass

from .schema import StreamScore

DEFAULT_WEIGHTS = {"structured": 1.0, "topology": 1.0, "literature": 1.0}


@dataclass
class Fused:
    vetoed: bool
    confidence: float | None                 # None when vetoed
    sub_scores: dict[str, float | None]
    contributing: list[str]


def fuse(scores: list[StreamScore], weights: dict[str, float] | None = None) -> Fused:
    w = weights or DEFAULT_WEIGHTS
    sub = {s.stream: s.value for s in scores}
    if any(s.veto for s in scores):
        return Fused(vetoed=True, confidence=None, sub_scores=sub, contributing=[])
    num = den = 0.0
    contributing: list[str] = []
    for s in scores:
        if s.value is None:
            continue
        wt = w.get(s.stream, 1.0)
        num += wt * s.value
        den += wt
        contributing.append(s.stream)
    conf = num / den if den > 0 else 0.5   # all abstained -> maximally uncertain
    return Fused(vetoed=False, confidence=round(conf, 4),
                 sub_scores=sub, contributing=contributing)
