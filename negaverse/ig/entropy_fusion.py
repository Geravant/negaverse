"""Ch4 — Entropy-weighted fusion.

Sinain's read-side idea (Cronen-Townsend 2002 query-clarity, applied to fusion):
trust each retrieval channel in proportion to how *sharply* its scores spike.
A confident witness piles weight on one answer (low entropy); a guesser spreads
flat (high entropy). Weight ∝ 1 + λ·(1 − H/H_max).

In negaverse each stream returns a single scalar `value ∈ [0,1]` = confidence
the pair is a true non-interaction. The distributional analog of "peakedness"
for a scalar Bernoulli-style score is the *binary* entropy H(value): maximal
(1 bit) at value=0.5 (the stream is maximally undecided) and 0 at the extremes
(the stream commits). So a stream sitting at 0.5 is a guesser; one near 0 or 1
knows its mind. Decisiveness = 1 − H_bin(value).

A stream may also publish its own peakedness in `evidence["confidence"] ∈ [0,1]`
(e.g. topology from its L3/RA spread, the LLM from its answer distribution);
when present that overrides the scalar proxy.

This is a faithful, drop-in variant of `negaverse.fusion.fuse`: same veto
semantics, same abstention handling, reduces *exactly* to the fixed-weight mean
when λ=0. Geometry suggests; evidence decides — so it is opt-in.
"""
from __future__ import annotations

import math

from ..fusion import Fused
from ..schema import StreamScore

_EPS = 1e-9


def binary_entropy(p: float) -> float:
    """H(p) in bits for a Bernoulli(p); H(0.5)=1, H(0)=H(1)=0."""
    p = min(1.0 - _EPS, max(_EPS, float(p)))
    return -(p * math.log2(p) + (1.0 - p) * math.log2(1.0 - p))


def decisiveness(score: StreamScore) -> float:
    """How committed this stream is, in [0,1]. Prefers an explicit
    evidence["confidence"]; else 1 − binary-entropy of the scalar value."""
    if score.value is None:
        return 0.0
    ev = score.evidence or {}
    if "confidence" in ev and ev["confidence"] is not None:
        return max(0.0, min(1.0, float(ev["confidence"])))
    return 1.0 - binary_entropy(score.value)


def entropy_weighted_fuse(
    scores: list[StreamScore],
    base_weights: dict[str, float] | None = None,
    lam: float = 1.0,
) -> Fused:
    """Fuse streams with entropy-adaptive weights.

    w_stream = base · (1 + λ · decisiveness(stream)).  λ=0 recovers the plain
    weighted mean. A single veto drops the pair (unchanged from fuse())."""
    w = base_weights or {}
    sub = {s.stream: s.value for s in scores}
    if any(s.veto for s in scores):
        return Fused(vetoed=True, confidence=None, sub_scores=sub, contributing=[])
    num = den = 0.0
    contributing: list[str] = []
    for s in scores:
        if s.value is None:
            continue
        base = w.get(s.stream, 1.0)
        wt = base * (1.0 + lam * decisiveness(s))
        num += wt * s.value
        den += wt
        contributing.append(s.stream)
    conf = num / den if den > 0 else 0.5
    return Fused(vetoed=False, confidence=round(conf, 4),
                 sub_scores=sub, contributing=contributing)


def stream_disagreement(scores: list[StreamScore]) -> float:
    """Spread of the non-abstaining streams' values, in [0,1].

    Routing signal for the GATED stage (Ch4 applied to *which pairs* deserve the
    expensive LLM): a pair where independent streams disagree is genuinely
    contested even if its fused mean looks unremarkable. Returns the population
    standard deviation of the stream values (0 = unanimous)."""
    vals = [s.value for s in scores if s.value is not None]
    if len(vals) < 2:
        return 0.0
    mean = sum(vals) / len(vals)
    var = sum((v - mean) ** 2 for v in vals) / len(vals)
    return math.sqrt(var)
