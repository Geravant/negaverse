"""Ch5 — Determinantal (DPP) selection for a diverse negative set.

Sinain's counting problem: don't fill the budget with six copies of one item;
fill it with items pointing in *different directions*. "Span a big chunk of
space" = the determinant of the Gram matrix, so a Determinantal Point Process
picks quality-weighted, mutually-diverse subsets:

    P(S) ∝ det(L_S),   L_ij = q_i · q_j · cos(x_i, x_j)

Two near-duplicates → two nearly-parallel rows → det collapses → the DPP refuses
to take both. Quality (q) and diversity (cosine) in one quantity.

In negaverse this curates the emitted eval/train negatives: q = confidence (for
eval) or hardness (for train); the cosine is over a pair-embedding, so the set
*spans* the interactome instead of clustering in one dense neighbourhood.

`greedy_map_dpp` is the standard fast greedy MAP inference (Chen et al. 2018),
O(k²n): incremental Cholesky of L, each step adds the item that most increases
log det.
"""
from __future__ import annotations

import numpy as np

_EPS = 1e-10


def greedy_map_dpp(quality, similarity, k: int) -> list[int]:
    """Greedy MAP-DPP: return indices of a high-quality, diverse size-≤k subset.

    quality:    (n,) non-negative relevance scores q_i.
    similarity: (n,n) symmetric similarity with 1s on the diagonal (e.g. cosine).
    Returns selected indices in the order chosen (log-det-greedy)."""
    q = np.asarray(quality, dtype=float).clip(min=0.0)
    S = np.asarray(similarity, dtype=float)
    n = len(q)
    k = min(int(k), n)
    if k <= 0 or n == 0:
        return []
    L = (q[:, None] * q[None, :]) * S            # DPP kernel

    cis = np.zeros((k, n))                        # incremental Cholesky factor rows
    di2 = np.copy(np.diag(L)).astype(float)       # remaining marginal gains
    selected: list[int] = []

    j = int(np.argmax(di2))
    if di2[j] <= _EPS:
        return []
    d_j = float(np.sqrt(di2[j]))                  # sqrt marginal gain of j at selection
    selected.append(j)
    di2[j] = -np.inf

    for it in range(1, k):
        prev = cis[:it - 1, :]                    # factors of earlier-selected items
        ci_j = cis[:it - 1, j]                    # just-selected j's own factor column
        eis = (L[j, :] - ci_j @ prev) / d_j       # e_i = (L_ji - <c_j,c_i>) / d_j
        cis[it - 1, :] = eis
        di2 = di2 - eis ** 2
        di2[selected] = -np.inf
        j = int(np.argmax(di2))
        if di2[j] <= _EPS:
            break
        d_j = float(np.sqrt(di2[j]))
        selected.append(j)
        di2[j] = -np.inf
    return selected
