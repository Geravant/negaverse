"""Layer 5 — matching & balancing, and the train/eval split (ARCHITECTURE.md §4 L5, P1).

Two products from the same pool, never merged (Park & Marcotte 2011):

  * eval  — *representative*: degree-matched to the positives so the benchmark
    can't be gamed by hubbiness. Sampled so a host's frequency among negatives
    tracks its positive degree.
  * train — *informative*: deliberately hard. Sampled toward near-boundary pairs
    (high topology risk / low confidence) — the negatives that sit near the
    decision boundary (Koyama 2023).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class Scored:
    u: str
    v: str
    confidence: float
    hardness: float          # topo percentile in [0,1]; higher = nearer positives
    sub_scores: dict
    conf_evidence: dict = field(default_factory=dict)   # per-stream reported confidence


def degree_matched_eval(
    scored: list[Scored],
    weights: "np.ndarray",
    n: int,
    seed: int = 0,
) -> list[Scored]:
    """Sample n negatives with the given per-candidate weights so the endpoint
    (host) degree distribution tracks the positives — defuses the 'this node is
    a hub' shortcut. Weights are precomputed by the pipeline over the confounder
    node type (see PipelineConfig.match_on_type)."""
    rng = np.random.default_rng(seed)
    if not scored:
        return []
    w = np.asarray(weights, dtype=float)
    if w.sum() == 0:
        w = np.ones(len(scored))
    w = w / w.sum()
    k = min(n, len(scored))
    idx = rng.choice(len(scored), size=k, replace=False, p=w)
    return [scored[i] for i in idx]


def hard_train(scored: list[Scored], n: int, exclude: set[tuple[str, str]]) -> list[Scored]:
    """The n hardest available negatives (nearest the positive manifold),
    disjoint from the eval set."""
    pool = [s for s in scored if (s.u, s.v) not in exclude]
    pool.sort(key=lambda s: s.hardness, reverse=True)
    return pool[:min(n, len(pool))]


def select_train(scored: list[Scored], n: int, exclude: set[tuple[str, str]],
                 mode: str = "stacked") -> list[Scored]:
    """Pick the n emitted training negatives from the scored pool. See
    PipelineConfig.train_selection for the modes and the evidence (FILTER-EFFECTIVENESS §11):

      * "stacked" (default) — hard tail (top 4n by hardness) RE-RANKED by fused
        confidence; keeps the pairs every signal agrees are true negatives. Best arm.
      * "safe" — the n highest-confidence negatives across the whole pool.
      * "hard" — the n hardest by topology alone (the historical default; loses to random).
    """
    pool = [s for s in scored if (s.u, s.v) not in exclude]
    if mode == "hard":
        pool.sort(key=lambda s: s.hardness, reverse=True)
        return pool[:min(n, len(pool))]
    if mode == "safe":
        pool.sort(key=lambda s: s.confidence, reverse=True)
        return pool[:min(n, len(pool))]
    if mode == "stacked":
        hard = sorted(pool, key=lambda s: s.hardness, reverse=True)[:max(4 * n, n)]
        hard.sort(key=lambda s: s.confidence, reverse=True)
        return hard[:min(n, len(hard))]
    raise ValueError(f"unknown train_selection mode: {mode!r} (want hard|safe|stacked)")
