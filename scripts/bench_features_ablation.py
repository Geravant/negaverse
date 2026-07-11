"""Does giving the model a REAL (sequence) axis rescue topology-hard negatives?

Test #1 from the discussion, on DRYAD (where ESM2 covers every protein):
train a link-predictor on positives + {random | topology-hard} negatives, test on
held-out positives + DRYAD gold negatives — under two feature sets:

  * graph  — spectral SVD embeddings (Hadamard). Topology-hard negatives look
    positive-like here => expected to be noise (negaverse hurts).
  * esm2   — ESM2 sequence embeddings (Hadamard) — an axis INDEPENDENT of the
    graph selection. If topology-hard negatives are structurally separable, the
    model can now learn them => negaverse should stop hurting / start helping.

Read: does Δ(hard − random) go from negative (graph features) to ≥0 (esm2)?

    PYTHONPATH=. python3 scripts/bench_features_ablation.py
"""
from __future__ import annotations

import numpy as np
import networkx as nx
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

from negaverse.io import load_embeddings_npz
from negaverse.bench.benchmark import _spectral_embeddings

_DRYAD = "local-docs/dryad-ppi"


def _load():
    pos, neg = [], []
    with open(f"{_DRYAD}/benchmarks/benchmarks/positives_and_negatives.tsv") as fh:
        next(fh)
        for line in fh:
            pair, cat = line.rstrip("\n").split("\t")
            a, b = pair.split("_")
            (pos if cat == "positive" else neg).append((a, b))
    emb = load_embeddings_npz(f"{_DRYAD}/esm2_t6_emb.npz")
    keep = lambda ps: [(a, b) for a, b in ps if a in emb and b in emb]
    return keep(pos), keep(neg), emb


def _random_nonedges(nodes, pos_set, n, rng):
    out, seen, t = [], set(), 0
    while len(out) < n and t < n * 80 + 1000:
        t += 1
        a, b = nodes[rng.integers(len(nodes))], nodes[rng.integers(len(nodes))]
        k = frozenset((a, b))
        if a != b and k not in pos_set and k not in seen:
            seen.add(k); out.append((a, b))
    return out


def _hard_nonedges(nodes, adj, pos_set, n, rng):
    """Topology-hard = non-edges with the most shared neighbours (positive-like),
    i.e. what negaverse's topology filter selects."""
    cand, t = [], 0
    while len(cand) < n * 40 and t < n * 400 + 5000:
        t += 1
        a, b = nodes[rng.integers(len(nodes))], nodes[rng.integers(len(nodes))]
        if a == b or frozenset((a, b)) in pos_set:
            continue
        cn = len(adj.get(a, set()) & adj.get(b, set()))
        if cn > 0:
            cand.append((cn, (a, b)))
    cand.sort(key=lambda x: -x[0])
    return [p for _, p in cand[:n]]


def main(seed=0):
    rng = np.random.default_rng(seed)
    pos, neg, emb = _load()
    rng.shuffle(pos)
    n_test = int(len(pos) * 0.2)
    test_pos, train_pos = pos[:n_test], pos[n_test:]
    nodes = sorted({p for pr in pos + neg for p in pr})
    pos_set = {frozenset(e) for e in pos} | {frozenset(e) for e in neg}

    G = nx.Graph(); G.add_nodes_from(nodes); G.add_edges_from(train_pos)
    adj = {n_: set(G[n_]) for n_ in G}
    sp, sidx, _ = _spectral_embeddings(G, nodes, 32, seed)
    spec = {n_: sp[sidx[n_]] for n_ in nodes}

    m = min(len(train_pos), 1500)
    train_pos_s = train_pos[:m]
    rand_neg = _random_nonedges(nodes, pos_set, m, rng)
    hard_neg = _hard_nonedges(nodes, adj, pos_set, m, rng)

    n_te = min(len(test_pos), len(neg))
    test_pairs = list(test_pos[:n_te]) + list(neg[:n_te])
    yte = np.r_[np.ones(n_te), np.zeros(n_te)]

    def feats(table, pairs):
        d = len(next(iter(table.values())))
        z = np.zeros(d)
        return np.asarray([(table.get(u, z) * table.get(v, z)) for u, v in pairs])

    results = {}
    for fname, table in [("graph(spectral)", spec), ("esm2(sequence)", emb)]:
        Xte = feats(table, test_pairs)
        for sname, tneg in [("random", rand_neg), ("topology-hard", hard_neg)]:
            Xtr = feats(table, list(train_pos_s) + list(tneg))
            ytr = np.r_[np.ones(len(train_pos_s)), np.zeros(len(tneg))]
            clf = RandomForestClassifier(n_estimators=200, random_state=seed, n_jobs=-1)
            clf.fit(Xtr, ytr)
            results[(fname, sname)] = round(float(roc_auc_score(
                yte, clf.predict_proba(Xte)[:, 1])), 3)

    print("=" * 60)
    print("DRYAD: do ESM2 features rescue topology-hard negatives?")
    print(f"  (train {len(train_pos_s)} pos + neg; test {n_te}/class vs gold)")
    print("=" * 60)
    print(f"{'features':<18}{'random':>9}{'topo-hard':>11}{'Δ':>8}")
    print("-" * 46)
    for f in ("graph(spectral)", "esm2(sequence)"):
        r, h = results[(f, "random")], results[(f, "topology-hard")]
        print(f"{f:<18}{r:>9.3f}{h:>11.3f}{h-r:>+8.3f}")
    print("-" * 46)
    print("\n  Δ<0: topology-hard negatives hurt (noise under these features).")
    print("  Δ≥0 under esm2 but <0 under graph => the real axis rescues them.")


if __name__ == "__main__":
    main()
