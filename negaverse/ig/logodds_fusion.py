"""Log-odds (log-opinion-pool) fusion + a calibrated stack with a lower bound.

The shipped fusion (`negaverse.fusion.fuse`) is a weighted arithmetic *mean* of
the streams' [0,1] sub-scores. Its own docstring flags the defect: a mean
*dilutes* agreement — two streams that both say "safe" (0.75 and 1.0) fuse to
0.875 < either — and the output is not a probability.

Fuse in the Bernoulli family's natural coordinate instead. Read each stream's
value v_j as its own posterior P(N | s_j) and combine as an independent-evidence
(naive-Bayes / log-opinion) pool:

    L(x) = logit(pi0) + Σ_j w_j (logit(v_j) - logit(pi0)),   P_N(x) = σ(L(x))

Then agreeing evidence *adds* (reinforces, never dilutes), contradicting evidence
cancels, an abstaining stream contributes exactly 0, and a veto is still a hard
drop. `logodds_fuse` is a drop-in for `fuse` with identical veto/abstention
semantics (opt-in, like `entropy_weighted_fuse`).

The pool with fixed weights and pi0 gives the right *coordinate* but not a
calibrated scale, and it double-counts correlated streams (the rules stream is
itself a mean of correlated per-rule votes — the "mean of mean" dilution). The
fix for both is `LogOddsStack`: a cross-fittable L2-regularised logistic
regression over the per-stream (and per-rule) logits. The L2 penalty shrinks
redundant/correlated features, so it *decorrelates*; the fit yields a calibrated
P_N and, via the delta method on the coefficient covariance, a lower-confidence
bound σ(L − z·sd) — the quantity that should *authorise* a negative label
(docs/FILTER-EFFECTIVENESS.md calibration arm; the LCB the schema still lacks).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from ..fusion import Fused
from ..schema import StreamScore

_EPS = 1e-6


def logit(p: float, eps: float = _EPS) -> float:
    p = min(1.0 - eps, max(eps, float(p)))
    return math.log(p / (1.0 - p))


def sigmoid(x: float) -> float:
    # overflow-safe
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def logodds_pool(values, weights=None, prior: float = 0.5) -> float:
    """Accumulated log-odds L(x) of the non-abstaining stream values.

    `values` is a mapping {stream: value|None} or an iterable of (name, value).
    Abstentions (None) are skipped and contribute 0 (the pool's neutral element).
    Empty evidence returns logit(prior)."""
    if hasattr(values, "items"):
        items = list(values.items())
    else:
        items = list(values)
    w = weights or {}
    l0 = logit(prior)
    total = l0
    for name, v in items:
        if v is None:
            continue
        total += w.get(name, 1.0) * (logit(v) - l0)
    return total


def logodds_fuse(scores: list[StreamScore],
                 weights: dict[str, float] | None = None,
                 prior: float = 0.5) -> Fused:
    """Fuse streams in log-odds coordinates. Drop-in for `fusion.fuse`: a single
    veto drops the pair; abstaining streams are skipped; confidence = σ(L)."""
    sub = {s.stream: s.value for s in scores}
    if any(s.veto for s in scores):
        return Fused(vetoed=True, confidence=None, sub_scores=sub, contributing=[])
    contributing = [s.stream for s in scores if s.value is not None]
    L = logodds_pool(((s.stream, s.value) for s in scores), weights, prior)
    conf = sigmoid(L) if contributing else prior
    return Fused(vetoed=False, confidence=round(conf, 4),
                 sub_scores=sub, contributing=contributing)


@dataclass
class _Fit:
    coef: np.ndarray            # (d+1,) includes intercept at index 0
    cov: np.ndarray             # (d+1, d+1) coefficient covariance (delta method)
    names: list[str]


class LogOddsStack:
    """Calibrated log-odds stack over per-stream logits, with a lower bound.

    Features are logits of stream/rule values with abstentions imputed to 0
    (= logit(prior=0.5), the pool's neutral contribution). Labels y: 1 = true
    non-interaction (N), 0 = interaction (P). L2 regularisation shrinks correlated
    features (decorrelation). `lcb` returns σ(L − z·sd) from the coefficient
    covariance — the calibrated floor that should authorise a label."""

    def __init__(self, l2: float = 1.0, alpha: float = 0.05):
        self.l2 = l2
        self.alpha = alpha
        self._fit: _Fit | None = None

    def fit(self, Z: np.ndarray, y: np.ndarray, names: list[str] | None = None) -> "LogOddsStack":
        from sklearn.linear_model import LogisticRegression
        Z = np.asarray(Z, dtype=float)
        y = np.asarray(y, dtype=int)
        clf = LogisticRegression(C=1.0 / self.l2, solver="lbfgs", max_iter=2000)
        clf.fit(Z, y)
        coef = np.concatenate([clf.intercept_, clf.coef_.ravel()])
        # delta-method covariance: (Xᵀ W X + λ I)⁻¹, W = p(1−p); λ on non-intercept
        D = np.column_stack([np.ones(len(Z)), Z])
        p = clf.predict_proba(Z)[:, 1]
        Wd = p * (1.0 - p)
        H = D.T @ (Wd[:, None] * D)
        lam = 1.0 / clf.C
        reg = np.eye(D.shape[1]) * lam
        reg[0, 0] = 0.0
        cov = np.linalg.pinv(H + reg)
        self._fit = _Fit(coef=coef, cov=cov, names=names or [f"f{i}" for i in range(Z.shape[1])])
        self._clf = clf
        return self

    def _design(self, Z: np.ndarray) -> np.ndarray:
        Z = np.asarray(Z, dtype=float)
        return np.column_stack([np.ones(len(Z)), Z])

    def logodds(self, Z: np.ndarray) -> np.ndarray:
        return self._design(Z) @ self._fit.coef

    def proba(self, Z: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-self.logodds(Z)))

    def lcb(self, Z: np.ndarray, alpha: float | None = None) -> np.ndarray:
        """Lower-confidence bound on P_N: σ(L − z·sd(L)). Higher = safer even after
        accounting for how little the fit knows about this region."""
        from scipy.stats import norm
        a = self.alpha if alpha is None else alpha
        D = self._design(Z)
        L = D @ self._fit.coef
        var = np.einsum("ij,jk,ik->i", D, self._fit.cov, D)
        z = norm.ppf(1.0 - a)
        return 1.0 / (1.0 + np.exp(-(L - z * np.sqrt(np.maximum(var, 0.0)))))
