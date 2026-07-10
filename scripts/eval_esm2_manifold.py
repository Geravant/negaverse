"""ESM2 (sequence) manifold vs spectral (graph) manifold, and their combination.

Lucy's variant: give each protein a location from its *sequence* (ESM2) rather
than from *who it interacts with* (graph SVD). Same surprisal recipe — resemblance
to the frozen positive manifold — over a different feature space. This script
runs both on the DRYAD PPI benchmark (which ships sequences + precomputed ESM2
embeddings + matched positive/negative controls), leakage-free, and asks:

  1. How does ESM2-manifold surprisal separate positives from negatives (AUROC)?
  2. How does it compare to the spectral (graph) manifold and negaverse topology?
  3. Are the axes independent (low correlation → complementary)?
  4. Does fusing them beat the best single axis?

Protocol (mirrors scripts/eval_ig_features.py experiment C2): split positives
into train/test; build the graph + spectral embeddings on TRAIN positives only;
freeze the positive manifold from TRAIN positive pairs; score held-out positives
vs negatives. ESM2 embeddings are pretrained, so they carry no label leakage.

    PYTHONPATH=. python3 scripts/eval_esm2_manifold.py
"""
from __future__ import annotations

import json
import os

import numpy as np
import networkx as nx
from sklearn.metrics import roc_auc_score

from negaverse.ig import background_similarity, normalize_rows
from negaverse.bench.benchmark import _spectral_embeddings
from negaverse.graph import TypedInteractionGraph
from negaverse.streams.topology import TopologyFilter

_DIR = "local-docs/dryad-ppi"
_PAIRS = f"{_DIR}/benchmarks/benchmarks/positives_and_negatives.tsv"
_EMB = f"{_DIR}/esm2_t6_emb.npz"


def _load_pairs():
    pos, neg = [], []
    with open(_PAIRS) as fh:
        next(fh)                                     # header
        for line in fh:
            pair, cat = line.rstrip("\n").split("\t")
            a, b = pair.split("_")
            (pos if cat == "positive" else neg).append((a, b))
    return pos, neg


def _pair_rep(node_emb, pairs, op="hadamard"):
    d = next(iter(node_emb.values())).shape[0]
    zero = np.zeros(d)
    out = []
    for u, v in pairs:
        eu, ev = node_emb.get(u), node_emb.get(v)
        if eu is None or ev is None:
            out.append(zero if op != "concat" else np.zeros(2 * d))
            continue
        if op == "hadamard":
            out.append(eu * ev)
        elif op == "avg":
            out.append(0.5 * (eu + ev))
        elif op == "l1":
            out.append(np.abs(eu - ev))
        elif op == "concat":
            out.append(np.concatenate([np.minimum(eu, ev), np.maximum(eu, ev)]))
        else:
            raise ValueError(op)
    return np.asarray(out, dtype=float)


def _surprisal_auroc(node_emb, test_pairs, bg_pairs, y, ops, k_list):
    grid, best, best_scores = {}, ("", -1.0), None
    for op in ops:
        Xt = normalize_rows(_pair_rep(node_emb, test_pairs, op))
        Xb = normalize_rows(_pair_rep(node_emb, bg_pairs, op))
        for k in k_list:
            s = background_similarity(Xt, Xb, k=k)
            a = round(float(roc_auc_score(y, s)), 4)
            grid[f"{op}_k{k}"] = a
            if a > best[1]:
                best, best_scores = (f"{op}_k{k}", a), s
    return grid, best, best_scores


def main(seed: int = 0, test_frac: float = 0.2, n_neg: int = 3000,
         max_bg: int = 2500, k_list=(10, 25)):
    rng = np.random.default_rng(seed)
    pos_all, neg_all = _load_pairs()
    npz = np.load(_EMB)
    ids, emb = npz["ids"], npz["emb"]
    esm = {str(i): emb[j].astype(float) for j, i in enumerate(ids)}
    have = set(esm)

    # keep only pairs both of whose proteins have an ESM2 embedding (fair to both arms)
    pos_all = [p for p in pos_all if p[0] in have and p[1] in have]
    neg_all = [p for p in neg_all if p[0] in have and p[1] in have]
    rng.shuffle(pos_all)
    rng.shuffle(neg_all)

    n_test = int(len(pos_all) * test_frac)
    test_pos, train_pos = pos_all[:n_test], pos_all[n_test:]
    neg = neg_all[:n_neg]

    nodes = sorted({p for pr in pos_all + neg_all for p in pr})
    # --- graph arm: spectral embeddings on TRAIN positives only (leakage-free) ---
    G = nx.Graph(); G.add_nodes_from(nodes); G.add_edges_from(train_pos)
    semb, sidx, _ = _spectral_embeddings(G, nodes, 32, seed)
    spectral = {n: semb[sidx[n]].astype(float) for n in nodes}

    test_pairs = list(test_pos) + list(neg)
    y = np.r_[np.ones(len(test_pos)), np.zeros(len(neg))]     # 1 = held-out positive
    bg_pairs = train_pos[:max_bg]                             # frozen positive manifold

    # --- ESM2 (sequence) manifold ---
    esm_grid, esm_best, esm_scores = _surprisal_auroc(
        esm, test_pairs, bg_pairs, y, ops=("hadamard", "avg", "l1", "concat"), k_list=k_list)
    # --- spectral (graph) manifold ---
    sp_grid, sp_best, sp_scores = _surprisal_auroc(
        spectral, test_pairs, bg_pairs, y, ops=("hadamard", "avg"), k_list=k_list)

    # --- negaverse topology risk (graph, local) as a third axis ---
    tg = TypedInteractionGraph.from_edges(
        train_pos, {n: "protein" for n in nodes},
        admissible_types=[("protein", "protein")], name="dryad-train")
    tf = TopologyFilter(); tf.fit(tg)
    topo = np.asarray([(tf.score(tg, u, v).evidence or {}).get("risk", 0.0)
                       for u, v in test_pairs])
    topo_auroc = round(float(roc_auc_score(y, topo)), 4)

    # --- correlations + fusion (unsupervised z-score average) ---
    zs = lambda x: (x - x.mean()) / (x.std() if x.std() > 1e-9 else 1.0)
    axes = {"esm2": esm_scores, "spectral": sp_scores, "topology": topo}
    corr = {}
    names = list(axes)
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            corr[f"{names[i]}~{names[j]}"] = round(
                float(np.corrcoef(axes[names[i]], axes[names[j]])[0, 1]), 3)
    fused = {
        "esm2+spectral": round(float(roc_auc_score(y, zs(esm_scores) + zs(sp_scores))), 4),
        "esm2+topology": round(float(roc_auc_score(y, zs(esm_scores) + zs(topo))), 4),
        "esm2+spectral+topology": round(float(roc_auc_score(
            y, zs(esm_scores) + zs(sp_scores) + zs(topo))), 4),
    }

    report = {
        "dataset": "DRYAD PPI (benchmarks/positives_and_negatives.tsv)",
        "n_train_pos": len(train_pos), "n_test_pos": len(test_pos), "n_neg": len(neg),
        "esm2_manifold": {"best": {"rep": esm_best[0], "auroc": esm_best[1]}, "grid": esm_grid},
        "spectral_manifold": {"best": {"rep": sp_best[0], "auroc": sp_best[1]}, "grid": sp_grid},
        "topology_risk_auroc": topo_auroc,
        "axis_correlations": corr,
        "fused_auroc": fused,
    }
    os.makedirs("out", exist_ok=True)
    with open("out/esm2_manifold_eval.json", "w") as fh:
        json.dump(report, fh, indent=2)

    print("=" * 70)
    print("ESM2 (sequence) vs spectral (graph) manifold — DRYAD PPI, leakage-free")
    print("=" * 70)
    print(f"train_pos={len(train_pos)}  test_pos={len(test_pos)}  neg={len(neg)}\n")
    print("Single-axis separation (AUROC, positive vs negative):")
    print(f"   ESM2 sequence manifold  : {esm_best[1]:.4f}  ({esm_best[0]})")
    print(f"   spectral graph manifold : {sp_best[1]:.4f}  ({sp_best[0]})")
    print(f"   negaverse topology risk : {topo_auroc:.4f}")
    print("\nAxis correlations (low = complementary):")
    for k, v in corr.items():
        print(f"   {k:<22} {v:+.3f}")
    print("\nFused (unsupervised z-score average):")
    for k, v in fused.items():
        print(f"   {k:<24} {v:.4f}")
    print("\nESM2 representation sweep:", esm_grid)
    print("(full report -> out/esm2_manifold_eval.json)")


if __name__ == "__main__":
    main()
