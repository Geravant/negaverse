"""Validation harness (ARCHITECTURE.md §8 validation).

What we can measure on the SARS-CoV-2 demo graph *without* an in-space gold
negative set:

  * leakage        — no emitted negative may be a known positive (must be 0).
  * degree-match   — KS distance between the host-degree distribution of the
    positives and of the eval negatives, vs a random-negative baseline. Lower =
    better matched = the hubbiness shortcut is defused (Park & Marcotte).
  * hardness split — train negatives should sit nearer the positive manifold
    than eval negatives (Koyama).

The gold-negative checks (rank Negatome golds high-confidence; don't emit
held-out positives as confident negatives) require a human-human positive graph
where Negatome is in-space; `gold_recall` implements them for when that graph is
loaded.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import ks_2samp

from .graph import TypedInteractionGraph
from .schema import NegativeRecord


def _typed_degrees(graph: TypedInteractionGraph, pairs, match_type: str):
    degs = []
    for u, v in pairs:
        for n in (u, v):
            if graph.node_type.get(n) == match_type:
                degs.append(graph.degree(n))
    return np.array(degs, dtype=float)


def degree_match(
    graph: TypedInteractionGraph,
    eval_records: list[NegativeRecord],
    match_type: str = "viral",
    seed: int = 0,
) -> dict:
    """KS distance of the confounder node type's degree distribution: negaverse
    eval set vs a random non-edge baseline, both against the positives. Lower KS
    = better matched = the hubbiness shortcut on that side is defused."""
    # positives that live in the candidate (admissible) space
    pos_pairs = [(u, v) for u, v in graph.g.edges() if graph.admissible(u, v)]
    pos_d = _typed_degrees(graph, pos_pairs, match_type)

    neg_pairs = [(r.u, r.v) for r in eval_records]
    neg_d = _typed_degrees(graph, neg_pairs, match_type)

    # random baseline: uniform admissible non-edges, same count
    rng = np.random.default_rng(seed)
    hosts = graph.nodes_of_type("host")
    virals = graph.nodes_of_type("viral")
    rand_pairs = []
    tries = 0
    while len(rand_pairs) < len(neg_pairs) and tries < 50 * len(neg_pairs) + 100:
        tries += 1
        a = virals[rng.integers(len(virals))] if virals else hosts[rng.integers(len(hosts))]
        b = hosts[rng.integers(len(hosts))]
        if not graph.is_positive(a, b) and a != b:
            rand_pairs.append((a, b))
    rand_d = _typed_degrees(graph, rand_pairs, match_type)

    ks_neg = float(ks_2samp(pos_d, neg_d).statistic) if len(neg_d) else float("nan")
    ks_rand = float(ks_2samp(pos_d, rand_d).statistic) if len(rand_d) else float("nan")
    return {
        "match_type": match_type,
        "ks_negaverse_vs_positive": round(ks_neg, 4),
        "ks_random_vs_positive": round(ks_rand, 4),
        "improvement": round(ks_rand - ks_neg, 4),
        f"mean_{match_type}_degree": {
            "positive": round(float(pos_d.mean()), 3) if len(pos_d) else None,
            "negaverse_eval": round(float(neg_d.mean()), 3) if len(neg_d) else None,
            "random": round(float(rand_d.mean()), 3) if len(rand_d) else None,
        },
    }


def leakage(graph: TypedInteractionGraph, records: list[NegativeRecord]) -> int:
    return sum(1 for r in records if graph.is_positive(r.u, r.v))


def hardness_split(records: list[NegativeRecord]) -> dict:
    tr = [r.hardness for r in records if r.mode == "train"]
    ev = [r.hardness for r in records if r.mode == "eval"]
    return {
        "train_mean_hardness": round(float(np.mean(tr)), 4) if tr else None,
        "eval_mean_hardness": round(float(np.mean(ev)), 4) if ev else None,
    }


def gold_recall(
    records: list[NegativeRecord],
    gold_pairs: set[frozenset],
    conf_threshold: float = 0.5,
) -> dict:
    """Of the Negatome golds that appear among emitted negatives, what fraction
    are called high-confidence? Only meaningful when the graph is in-space."""
    emitted = {frozenset((r.u, r.v)): r for r in records}
    hits = [emitted[g] for g in gold_pairs if g in emitted]
    if not hits:
        return {"golds_in_pool": 0, "note": "no gold pairs in emitted set (space mismatch?)"}
    high = sum(1 for r in hits if r.confidence >= conf_threshold)
    return {
        "golds_in_pool": len(hits),
        "high_confidence": high,
        "recall_at_threshold": round(high / len(hits), 4),
        "mean_confidence": round(float(np.mean([r.confidence for r in hits])), 4),
    }
