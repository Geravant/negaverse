"""Fisher-information (Bayesian D-optimal) batch selection.

The DPP (`ig/dpp.py`) picks a set that spans distinct *embedding* directions.
This picks a set that constrains distinct *downstream-model parameter*
directions — it chooses candidates that, if labelled, would most sharpen the
model, while refusing 500 near-duplicates that all teach the same parameter.

For a candidate with feature x_i the (per-candidate) Fisher information of a
logistic model is F_i = w_i · x_i x_iᵀ, with w_i = p_i(1−p_i) — maximal at the
decision boundary (p≈0.5), i.e. the *hard* candidates. After a safety gate,
pick the batch S maximising the log-det of the accumulated information:

    argmax_S  log det( F_0 + Σ_{i∈S} w_i x_i x_iᵀ )

`F_0 = ridge·I` is the prior precision (a Bayesian D-optimal prior; also keeps the
matrix invertible). Greedy adds the candidate with the largest marginal
log-det gain log(1 + w_i x_iᵀ M⁻¹ x_i) and updates M⁻¹ by Sherman–Morrison —
O(k·n·d²), and unlike a low-rank cosine DPP it does *not* saturate at the
embedding rank. Same "greedy log-det" lineage as `dpp.py`, in parameter space.

RandomForest / LightGBM (the downstream learners in the bench) have no usable
θ-gradient, so the Fisher weights come from a **differentiable surrogate** — a
logistic regression on the same features — and the selected batch is transferred
to the tree model for evaluation. Sourati et al. (JMLR 2017) analyse the
active-selection interpretation.
"""
from __future__ import annotations

import numpy as np


def logistic_fisher_weights(X_fit, y_fit, X_cand, l2: float = 1.0):
    """Fit a logistic surrogate on (X_fit, y_fit) and return, for each candidate
    row of X_cand, the Fisher weight w = p(1−p) and the predicted p (link prob).
    Boundary candidates (p≈0.5) get the largest weight."""
    from sklearn.linear_model import LogisticRegression
    clf = LogisticRegression(C=1.0 / l2, solver="lbfgs", max_iter=2000)
    clf.fit(np.asarray(X_fit, dtype=float), np.asarray(y_fit, dtype=int))
    p = clf.predict_proba(np.asarray(X_cand, dtype=float))[:, 1]
    return p * (1.0 - p), p


def greedy_fisher_design(X, weights=None, k: int = 0, ridge: float = 1.0) -> list[int]:
    """Greedy Bayesian D-optimal selection: indices of a size-≤k batch that
    maximises log det(ridge·I + Σ w_i x_i x_iᵀ).

    X:       (n, d) candidate feature vectors.
    weights: (n,) non-negative Fisher weights w_i (default all 1 = pure D-optimal,
             i.e. span the feature directions / de-duplicate). Pass p(1−p) to also
             pull the batch toward the model's decision boundary.
    Returns selected indices in the order chosen."""
    X = np.asarray(X, dtype=float)
    n, d = X.shape
    k = min(int(k), n)
    if k <= 0 or n == 0:
        return []
    w = np.ones(n) if weights is None else np.asarray(weights, dtype=float).clip(min=0.0)
    Minv = np.eye(d) / float(ridge)              # (ridge·I)⁻¹
    selected: list[int] = []
    remaining = np.ones(n, dtype=bool)

    for _ in range(k):
        XM = X @ Minv                            # (n, d)
        quad = np.einsum("ij,ij->i", XM, X)      # x_iᵀ M⁻¹ x_i
        gain = w * quad                          # marginal log-det gain = log(1+gain)
        gain[~remaining] = -np.inf
        j = int(np.argmax(gain))
        if not np.isfinite(gain[j]) or gain[j] <= 1e-12:
            break                                # no further information to gain
        selected.append(j)
        remaining[j] = False
        # Sherman–Morrison rank-1 update of M⁻¹ for M += w_j x_j x_jᵀ
        xj = X[j]
        Mx = Minv @ xj
        Minv = Minv - np.outer(Mx, Mx) * (w[j] / (1.0 + w[j] * (xj @ Mx)))
    return selected
