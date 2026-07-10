"""Ch7 — Relative-margin gating (geometric intent routing).

Sinain classifies a question by *which exemplar cloud it is nearer to*, and —
because the clouds overlap — judges nearness relatively, not with an absolute
threshold:

    is_A(q) ⟺ mean-cos(q, M_A) − mean-cos(q, M_B) > margin  AND  mean-cos(q, M_A) > floor

Negaverse's topology gate floors no-overlap pairs to a fixed 0.98 and the rule
graded map is a fixed 0.5 ± 0.5·weight — both absolute cutoffs. When the
positive-like and negative-like clouds overlap (they do), the honest test is the
*margin* between "looks like the negative cloud" and "looks like the positive
cloud", with a floor so a pair far from both abstains rather than guessing.
"""
from __future__ import annotations

import numpy as np


def margin_gate(sim_neg, sim_pos, margin: float = 0.05, floor: float = 0.0):
    """Boolean gate: is each pair confidently negative-like?
    True where (sim_neg − sim_pos) > margin AND sim_neg > floor."""
    sn = np.asarray(sim_neg, dtype=float)
    sp = np.asarray(sim_pos, dtype=float)
    return (sn - sp > margin) & (sn > floor)


def margin_score(sim_neg, sim_pos) -> np.ndarray:
    """Graded confidence-that-it's-a-negative in [0,1] from the relative margin.
    Squashes (sim_neg − sim_pos) through a logistic centred at 0, so a pair
    equidistant from both clouds sits at 0.5 (maximally uncertain — abstain-like)
    rather than being forced to a hard label."""
    sn = np.asarray(sim_neg, dtype=float)
    sp = np.asarray(sim_pos, dtype=float)
    return 1.0 / (1.0 + np.exp(-(sn - sp) / 0.1))
