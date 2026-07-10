"""Unit tests for the information-geometry prototypes (negaverse/ig/).

Proves the mechanisms are correct in isolation: entropy-weighting reduces to the
plain mean at λ=0 and up-weights the decisive stream otherwise; greedy MAP-DPP
refuses near-duplicates; top-k-mean surprisal ranks near-background rows highest;
the relative margin gate behaves symmetrically.

    python -m tests.test_ig_features
"""
from __future__ import annotations

import numpy as np

from negaverse.ig import (
    binary_entropy,
    decisiveness,
    entropy_weighted_fuse,
    stream_disagreement,
    greedy_map_dpp,
    topk_mean_sim,
    margin_gate,
    margin_score,
)
from negaverse.schema import StreamScore


# --- Ch4: entropy-weighted fusion --------------------------------------
def test_binary_entropy_peaks_at_half():
    assert binary_entropy(0.5) > binary_entropy(0.9) > binary_entropy(0.99)
    assert abs(binary_entropy(0.5) - 1.0) < 1e-6          # 1 bit at p=0.5
    assert binary_entropy(0.0) < 1e-6 and binary_entropy(1.0) < 1e-6


def test_lambda_zero_is_plain_mean():
    scores = [StreamScore("topology", 0.9), StreamScore("rules", 0.6)]
    fused = entropy_weighted_fuse(scores, lam=0.0)
    assert abs(fused.confidence - 0.75) < 1e-6           # (0.9+0.6)/2


def test_decisive_stream_is_upweighted():
    # topology commits (0.95, low entropy); rules hedge (0.55, high entropy).
    scores = [StreamScore("topology", 0.95), StreamScore("rules", 0.55)]
    mean = (0.95 + 0.55) / 2                              # 0.75
    fused = entropy_weighted_fuse(scores, lam=1.0)
    # entropy weighting pulls the fused value toward the confident 0.95
    assert fused.confidence > mean


def test_explicit_confidence_overrides_scalar_proxy():
    s = StreamScore("literature", 0.6, evidence={"confidence": 1.0})
    assert abs(decisiveness(s) - 1.0) < 1e-9


def test_veto_short_circuits():
    scores = [StreamScore("topology", 0.9), StreamScore("veto_rule", None, veto=True)]
    fused = entropy_weighted_fuse(scores)
    assert fused.vetoed and fused.confidence is None


def test_abstentions_ignored():
    scores = [StreamScore("topology", 0.8), StreamScore("literature", None)]
    fused = entropy_weighted_fuse(scores, lam=1.0)
    assert fused.confidence == 0.8 and fused.contributing == ["topology"]


def test_disagreement_metric():
    unanimous = [StreamScore("a", 0.8), StreamScore("b", 0.8)]
    split = [StreamScore("a", 0.1), StreamScore("b", 0.9)]
    assert stream_disagreement(unanimous) == 0.0
    assert stream_disagreement(split) > stream_disagreement(unanimous)


# --- Ch5: DPP diversity -------------------------------------------------
def test_dpp_refuses_near_duplicates():
    # three clones pointing one way, one lone vector pointing another
    X = np.array([[1, 0], [1, 0], [1, 0], [0, 1]], dtype=float)
    Xn = X / np.linalg.norm(X, axis=1, keepdims=True)
    S = Xn @ Xn.T
    q = np.ones(4)
    picked = greedy_map_dpp(q, S, k=2)
    # must cover both directions: one of the clones {0,1,2} AND the lone item 3
    assert 3 in picked
    assert len(set(picked) & {0, 1, 2}) == 1


def test_dpp_respects_quality():
    X = np.eye(4)                                         # all orthogonal
    q = np.array([0.1, 0.2, 0.9, 0.3])
    picked = greedy_map_dpp(q, X @ X.T, k=1)
    assert picked == [2]                                 # highest quality first


def test_dpp_k_capped_at_n():
    X = np.eye(3)
    picked = greedy_map_dpp(np.ones(3), X @ X.T, k=10)
    assert sorted(picked) == [0, 1, 2]


# --- Ch1: surprisal against a frozen background -------------------------
def test_topk_mean_sim_ranks_near_background_high():
    B = np.array([[1.0, 0.0], [0.9, 0.1]])               # background cluster near +x
    X = np.array([[1.0, 0.0],                            # on the cluster
                  [0.0, 1.0]])                            # orthogonal to it
    sims = topk_mean_sim(X, B, k=2)
    assert sims[0] > sims[1]


def test_topk_handles_small_background_and_empty():
    B = np.array([[1.0, 0.0]])
    X = np.array([[1.0, 0.0]])
    assert abs(topk_mean_sim(X, B, k=5)[0] - 1.0) < 1e-6  # k clamped to m
    assert topk_mean_sim(X, np.zeros((0, 2)), k=3)[0] == 0.0


# --- Ch7: relative-margin gate -----------------------------------------
def test_margin_gate_and_score():
    sim_neg = np.array([0.8, 0.3])
    sim_pos = np.array([0.2, 0.7])
    gate = margin_gate(sim_neg, sim_pos, margin=0.1, floor=0.0)
    assert bool(gate[0]) and not bool(gate[1])
    sc = margin_score(sim_neg, sim_pos)
    assert sc[0] > 0.5 > sc[1]                            # 0.5 == equidistant


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"PASS  {fn.__name__}")
    print(f"\n{len(fns)} checks passed")
