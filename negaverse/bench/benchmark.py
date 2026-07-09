"""Downstream-model benchmark (Koyama-style hypothesis test).

Design (avoids the obvious leakage/circularity traps):
  * split positives into train/test; build features on the TRAIN graph only, so
    no test edge leaks into a feature;
  * train a RandomForest link-predictor on positives + a training-negative set;
  * evaluate on held-out positives + a FIXED, unbiased random negative set (the
    same test set for every strategy, per Park & Marcotte);
  * compare training with random negatives vs negaverse (hard) negatives.

Caveat (worth stating in results): pair features are topological and negaverse
selects negatives partly by topology + degree, so there is some overlap between
the selection signal and the feature space. The test negatives are held fixed
and unbiased, so the comparison is still meaningful, but node2vec/ESM2 features
would make it more rigorous.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import networkx as nx
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, average_precision_score

from ..graph import TypedInteractionGraph
from ..pipeline import PipelineConfig, run_pipeline

_FEATS = ["deg_u", "deg_v", "common_neighbors", "jaccard",
          "adamic_adar", "resource_alloc", "pref_attach"]


def _features(adj: dict[str, set], pairs) -> np.ndarray:
    X = []
    for u, v in pairs:
        nu, nv = adj.get(u, set()), adj.get(v, set())
        cn = nu & nv
        union = len(nu | nv)
        jacc = len(cn) / union if union else 0.0
        aa = sum(1.0 / math.log(len(adj[w])) for w in cn if len(adj.get(w, ())) > 1)
        ra = sum(1.0 / len(adj[w]) for w in cn if adj.get(w))
        X.append([len(nu), len(nv), len(cn), jacc, aa, ra, len(nu) * len(nv)])
    return np.asarray(X, dtype=float)


def _random_nonedges(nodes, positives: set, n: int, rng, exclude: set | None = None):
    exclude = exclude or set()
    seen, out = set(), []
    N = len(nodes)
    tries, cap = 0, n * 50 + 1000
    while len(out) < n and tries < cap:
        tries += 1
        a, b = nodes[rng.integers(N)], nodes[rng.integers(N)]
        if a == b:
            continue
        k = frozenset((a, b))
        if k in positives or k in exclude or k in seen:
            continue
        seen.add(k)
        out.append((a, b))
    return out


@dataclass
class BenchmarkResult:
    strategies: dict           # {strategy: {auroc, auprc, n_train_neg}}
    n_positives: int
    n_test: int

    def summary(self) -> str:
        lines = [f"positives={self.n_positives}  test_pairs={2 * self.n_test}"]
        for s, m in self.strategies.items():
            lines.append(f"  {s:<10} AUROC={m['auroc']:.4f}  AUPRC={m['auprc']:.4f}  "
                         f"(train_neg={m['n_train_neg']})")
        r, nv = self.strategies.get("random"), self.strategies.get("negaverse")
        if r and nv:
            lines.append(f"  Δ negaverse-random: AUROC {nv['auroc'] - r['auroc']:+.4f}  "
                         f"AUPRC {nv['auprc'] - r['auprc']:+.4f}")
        return "\n".join(lines)


def _negaverse_negatives(graph, train_pos, node_type, n, seed, max_pool):
    tg = TypedInteractionGraph.from_edges(
        train_pos, dict(node_type), admissible_types=[("protein", "protein")],
        name="bench-train")
    cfg = PipelineConfig(modality="ppi", n_eval=0, n_train=n, max_pool=max_pool, seed=seed,
                         filters=["known_positive_veto", "structured", "embedding"])
    res = run_pipeline(tg, cfg)
    return [(r.u, r.v) for r in res.records if r.mode == "train"]


def run_benchmark(graph: TypedInteractionGraph, seed: int = 0, test_frac: float = 0.2,
                  max_positives: int | None = 10_000, max_pool: int = 40_000,
                  strategies=("random", "negaverse")) -> BenchmarkResult:
    rng = np.random.default_rng(seed)
    pos = [tuple(e) for e in graph.g.edges()]
    if max_positives and len(pos) > max_positives:
        idx = rng.choice(len(pos), size=max_positives, replace=False)
        pos = [pos[i] for i in idx]
    perm = rng.permutation(len(pos))
    pos = [pos[i] for i in perm]
    n_test = int(len(pos) * test_frac)
    test_pos, train_pos = pos[:n_test], pos[n_test:]

    node_type = {n: "protein" for n in graph.g.nodes()}
    pos_set = {frozenset(e) for e in pos}                 # never emit any known positive
    nodes = list(graph.g.nodes())

    # fixed unbiased test negatives (same for every strategy)
    test_neg = _random_nonedges(nodes, pos_set, len(test_pos), rng)
    test_excl = {frozenset(p) for p in test_neg}

    # features come from the TRAIN graph only
    train_G = nx.Graph()
    train_G.add_nodes_from(nodes)
    train_G.add_edges_from(train_pos)
    adj = {n: set(train_G[n]) for n in train_G}

    Xte = _features(adj, list(test_pos) + list(test_neg))
    yte = np.r_[np.ones(len(test_pos)), np.zeros(len(test_neg))]

    out: dict[str, dict] = {}
    for strat in strategies:
        if strat == "negaverse":
            train_neg = _negaverse_negatives(graph, train_pos, node_type,
                                             len(train_pos), seed, max_pool)
        else:
            train_neg = _random_nonedges(nodes, pos_set, len(train_pos), rng, exclude=test_excl)
        if not train_neg:
            continue
        Xtr = _features(adj, list(train_pos) + list(train_neg))
        ytr = np.r_[np.ones(len(train_pos)), np.zeros(len(train_neg))]
        clf = RandomForestClassifier(n_estimators=200, random_state=seed, n_jobs=-1)
        clf.fit(Xtr, ytr)
        p = clf.predict_proba(Xte)[:, 1]
        out[strat] = {
            "auroc": round(float(roc_auc_score(yte, p)), 4),
            "auprc": round(float(average_precision_score(yte, p)), 4),
            "n_train_neg": len(train_neg),
        }
    return BenchmarkResult(strategies=out, n_positives=len(pos), n_test=n_test)
