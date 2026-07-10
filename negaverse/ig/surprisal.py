"""Ch1 — Surprisal / typicality against a frozen background cloud.

Sinain freezes a cloud of *generic* sentences and scores a new fact by its
top-k-mean cosine to that cloud: land inside the cloud → unsurprising (prune);
land far outside → informative (keep).

Negaverse has two backgrounds worth freezing, and the primitive is the same:

  * gold negatives (Negatome) — a candidate that *resembles* validated
    non-interactions is a confident safe negative (typicality here is GOOD).
  * the positive manifold — a candidate that lands *inside* the cloud of known
    interactions is a suspected false negative (typicality here is BAD → flag).

`background_similarity(X, B, k)` is the shared kernel: top-k-mean cosine of each
row of X to the background set B. Turn it into a confidence (gold background) or
a risk (positive background) at the call site.
"""
from __future__ import annotations

import numpy as np

_EPS = 1e-12


def normalize_rows(X) -> np.ndarray:
    """L2-normalize each row so dot product = cosine (points on a sphere)."""
    X = np.asarray(X, dtype=float)
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    return X / np.maximum(norms, _EPS)


def topk_mean_sim(X, B, k: int = 10) -> np.ndarray:
    """Top-k-mean cosine of each row of X (n,d) to background B (m,d).

    Robust to k > m (uses all of B) and to an empty background (returns zeros)."""
    X = np.asarray(X, dtype=float)
    B = np.asarray(B, dtype=float)
    if X.ndim == 1:
        X = X[None, :]
    if B.size == 0:
        return np.zeros(len(X))
    Xn, Bn = normalize_rows(X), normalize_rows(B)
    sims = Xn @ Bn.T                              # (n, m) cosines
    k = max(1, min(int(k), sims.shape[1]))
    # top-k per row without a full sort
    part = np.partition(sims, -k, axis=1)[:, -k:]
    return part.mean(axis=1)


def background_similarity(X, B, k: int = 10) -> np.ndarray:
    """Alias with intent: 'how much does each pair resemble the background set'.
    High vs the gold-negative cloud = confident safe negative; high vs the
    positive manifold = suspected false negative."""
    return topk_mean_sim(X, B, k)
