"""Downstream-model benchmark (Koyama-style hypothesis test).

Design (avoids the obvious leakage/circularity traps):
  * split positives into train/test; build features on the TRAIN graph only, so
    no test edge leaks into a feature;
  * train a RandomForest link-predictor on positives + a training-negative set;
  * evaluate on held-out positives + a FIXED, unbiased random negative set (the
    same test set for every strategy, per Park & Marcotte);
  * compare training with random negatives vs negaverse (hard) negatives.

Feature families (choose with `feature_set`):
  * "topological" — hand-crafted local indices (CN, Jaccard, Adamic-Adar, RA,
    preferential attachment). Cheap, but note negaverse selects negatives partly
    by L3/RA topology, so these features share signal with the selection —
    inflating the comparison.
  * "spectral" — truncated-SVD node embeddings of the TRAIN adjacency, combined
    per pair with the Hadamard operator (the standard node2vec link-prediction
    feature). Structurally *independent* of the L3/Jaccard selection indices, so
    a margin that survives here is not an artifact of feature/selection overlap.

The test negatives are held fixed and unbiased for every strategy, so the
comparison is meaningful under either family; running both is the rigour check.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import networkx as nx
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import svds
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


def _spectral_embeddings(train_G: nx.Graph, nodes: list, dim: int, seed: int):
    """Truncated-SVD node embeddings of the TRAIN adjacency (features never see
    a test edge). Independent of the hand-crafted L3/Jaccard indices."""
    idx = {n: i for i, n in enumerate(nodes)}
    n = len(nodes)
    rows, cols = [], []
    for a, b in train_G.edges():
        i, j = idx[a], idx[b]
        rows += [i, j]
        cols += [j, i]
    k = max(1, min(dim, n - 2))
    if not rows:
        return np.zeros((n, k)), idx, k
    A = sp.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(n, n)).asfptype()
    rng = np.random.default_rng(seed)
    v0 = rng.standard_normal(min(A.shape))
    U, S, _ = svds(A, k=k, v0=v0)
    return U * S, idx, k                       # scale components by singular value


def _hadamard_features(emb: np.ndarray, idx: dict, pairs) -> np.ndarray:
    zero = np.zeros(emb.shape[1])
    return np.asarray([
        emb[idx[u]] * emb[idx[v]] if u in idx and v in idx else zero
        for u, v in pairs
    ], dtype=float)


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
    feature_set: str = "topological"
    test_neg_source: str = "random"

    def summary(self) -> str:
        lines = [f"positives={self.n_positives}  test_pairs={2 * self.n_test}  "
                 f"features={self.feature_set}  test_neg={self.test_neg_source}"]
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
                         filters=["known_positive_veto", "structured", "topology"])
    res = run_pipeline(tg, cfg)
    return [(r.u, r.v) for r in res.records if r.mode == "train"]


def run_benchmark(graph: TypedInteractionGraph, seed: int = 0, test_frac: float = 0.2,
                  max_positives: int | None = 10_000, max_pool: int = 40_000,
                  strategies=("random", "negaverse"),
                  feature_set: str = "topological", emb_dim: int = 32,
                  gold_test_neg: set | None = None) -> BenchmarkResult:
    """If `gold_test_neg` (a set of frozenset pairs in the graph's node space,
    e.g. Negatome mapped into HuRI) is given, the test negatives are these
    *hard, biologically-validated* non-interactions instead of easy random pairs
    — the fair test for whether hard training negatives generalize (see
    docs/BENCHMARK-FINDINGS.md)."""
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
    node_set = set(nodes)

    # test negatives: gold (hard) if supplied, else fixed unbiased random
    test_neg_source = "random"
    if gold_test_neg:
        gold = [tuple(p) for p in gold_test_neg
                if p not in pos_set and set(p) <= node_set]
        rng.shuffle(gold)
        if gold:
            test_neg_source = "gold"
            # balance the test set: pos and neg equal-sized
            n_bal = min(len(test_pos), len(gold))
            test_pos = test_pos[:n_bal]
            test_neg = gold[:n_bal]
            n_test = n_bal
    if test_neg_source == "random":
        test_neg = _random_nonedges(nodes, pos_set, len(test_pos), rng)
    test_excl = {frozenset(p) for p in test_neg}

    # features come from the TRAIN graph only (no test edge ever enters a feature)
    train_G = nx.Graph()
    train_G.add_nodes_from(nodes)
    train_G.add_edges_from(train_pos)

    if feature_set == "spectral":
        emb, idx, _ = _spectral_embeddings(train_G, nodes, emb_dim, seed)
        featurize = lambda pairs: _hadamard_features(emb, idx, pairs)
    elif feature_set == "topological":
        adj = {n: set(train_G[n]) for n in train_G}
        featurize = lambda pairs: _features(adj, pairs)
    else:
        raise ValueError(f"unknown feature_set: {feature_set!r}")

    Xte = featurize(list(test_pos) + list(test_neg))
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
        Xtr = featurize(list(train_pos) + list(train_neg))
        ytr = np.r_[np.ones(len(train_pos)), np.zeros(len(train_neg))]
        clf = RandomForestClassifier(n_estimators=200, random_state=seed, n_jobs=-1)
        clf.fit(Xtr, ytr)
        p = clf.predict_proba(Xte)[:, 1]
        out[strat] = {
            "auroc": round(float(roc_auc_score(yte, p)), 4),
            "auprc": round(float(average_precision_score(yte, p)), 4),
            "n_train_neg": len(train_neg),
        }
    return BenchmarkResult(strategies=out, n_positives=len(pos), n_test=n_test,
                           feature_set=feature_set, test_neg_source=test_neg_source)
