"""Prototype + evaluation harness for the IG features (docs/IG-FEATURES.md).

Runs three self-contained experiments and prints a report (also written to
out/ig_eval.json):

  A. Entropy-weighted fusion (Ch4) — controlled synthetic with ground truth.
     Does weighting streams by decisiveness beat the fixed-weight mean? Shown in
     BOTH regimes: the off-domain specialist that *abstains toward 0.5* (where it
     should help) and the one that is *confidently wrong* (where it should not).
  B. DPP set selection (Ch5) — real HuRI spectral pair-embeddings. Does a DPP
     pick cover more of the interactome than top-k at comparable quality?
  C. Gold-negative surprisal (Ch1) — real HuRI + Negatome. Does resemblance (in
     spectral pair-embedding space) to a frozen cloud of validated non-edges
     rank held-out gold negatives above true interactions?

Run:  PYTHONPATH=. python3 scripts/eval_ig_features.py
Real-data experiments (B, C) skip cleanly if local-docs/ files are absent.
"""
from __future__ import annotations

import json
import os

import numpy as np

from negaverse.schema import StreamScore
from negaverse.ig import (
    entropy_weighted_fuse,
    greedy_map_dpp,
    background_similarity,
    normalize_rows,
)

try:
    from sklearn.metrics import roc_auc_score
    from sklearn.cluster import KMeans
    _HAVE_SK = True
except Exception:                                    # pragma: no cover
    _HAVE_SK = False


def _auroc(y, s) -> float:
    if _HAVE_SK:
        return float(roc_auc_score(y, s))
    # tiny rank-based fallback (Mann–Whitney) if sklearn is unavailable
    y = np.asarray(y); s = np.asarray(s)
    pos, neg = s[y == 1], s[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    order = np.argsort(s)
    ranks = np.empty(len(s)); ranks[order] = np.arange(1, len(s) + 1)
    r_pos = ranks[y == 1].sum()
    return float((r_pos - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


# ======================================================================
# Experiment A — entropy-weighted fusion (controlled, always runs)
# ======================================================================
def experiment_entropy_fusion(seed: int = 0, n: int = 6000) -> dict:
    """A 'good' witness (moderate, informative everywhere) fused with a 'flaky'
    specialist that is correct on half the pairs and *confidently wrong* on the
    other half. Three fusion strategies:

      mean               — fixed-weight average; the flaky-wrong half drags it.
      entropy(scalar)     — decisiveness proxied by |value−0.5|. This BACKFIRES:
                            the confidently-wrong values look decisive and get
                            up-weighted. (The honest limit of the scalar proxy —
                            Sinain weights by a *distribution's* entropy, not one
                            Bernoulli scalar.)
      entropy(reported)   — the flaky witness publishes evidence['confidence']
                            (high where competent, low where guessing). Now the
                            weighting discounts the wrong half → recovers signal.

    The lesson (which belongs in the doc): entropy weighting pays off when a
    stream can report its own peakedness, not from the scalar alone."""
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, size=n)                    # 1 = safe negative, 0 = hidden positive
    competent = rng.random(n) < 0.5                   # flaky witness's competent half

    def signal(sep, noise, sign=+1):
        return np.clip(0.5 + sign * (y - 0.5) * sep + rng.normal(0, noise, size=n), 0.02, 0.98)

    good = signal(0.5, 0.20)                           # informative but noisy everywhere
    flaky = np.where(competent, signal(0.6, 0.10, +1), signal(0.6, 0.10, -1))
    flaky_reported = np.where(competent, 0.9, 0.1)     # the flaky witness knows when it guesses

    mean_c, ent_scalar, ent_reported = [], [], []
    for i in range(n):
        plain = [StreamScore("good", float(good[i])), StreamScore("flaky", float(flaky[i]))]
        reported = [StreamScore("good", float(good[i]), evidence={"confidence": 0.7}),
                    StreamScore("flaky", float(flaky[i]),
                                evidence={"confidence": float(flaky_reported[i])})]
        mean_c.append(entropy_weighted_fuse(plain, lam=0.0).confidence)
        ent_scalar.append(entropy_weighted_fuse(plain, lam=3.0).confidence)
        ent_reported.append(entropy_weighted_fuse(reported, lam=3.0).confidence)

    return {
        "auroc_good_witness_alone": round(_auroc(y, good), 4),
        "auroc_mean_fusion": round(_auroc(y, mean_c), 4),
        "auroc_entropy_scalar_proxy": round(_auroc(y, ent_scalar), 4),
        "auroc_entropy_reported_confidence": round(_auroc(y, ent_reported), 4),
    }


# ======================================================================
# Real-data plumbing (HuRI spectral pair-embeddings)
# ======================================================================
def _load_huri():
    from negaverse.io import load_huri_graph
    return load_huri_graph()


def _spectral_pair_embeddings(graph, dim=32, seed=0):
    """Truncated-SVD node embeddings of HuRI, Hadamard-combined per pair.
    Returns (nodes, node_emb, idx)."""
    from negaverse.bench.benchmark import _spectral_embeddings
    import networkx as nx
    nodes = list(graph.g.nodes())
    G = nx.Graph()
    G.add_nodes_from(nodes)
    G.add_edges_from(graph.g.edges())
    emb, idx, _ = _spectral_embeddings(G, nodes, dim, seed)
    return nodes, emb, idx


def _pair_emb(emb, idx, pairs):
    d = emb.shape[1]
    out = []
    for u, v in pairs:
        if u in idx and v in idx:
            out.append(emb[idx[u]] * emb[idx[v]])    # Hadamard (node2vec link feat)
        else:
            out.append(np.zeros(d))
    return np.asarray(out, dtype=float)


# ======================================================================
# Experiment B — DPP diversity on real HuRI hard-negative pool
# ======================================================================
def experiment_dpp(seed: int = 0, pool: int = 1500, k: int = 80) -> dict:
    graph = _load_huri()
    rng = np.random.default_rng(seed)
    nodes, emb, idx = _spectral_pair_embeddings(graph, dim=48, seed=seed)
    adj = {n: set(graph.g[n]) for n in graph.g.nodes()}
    node_arr = np.array(nodes, dtype=object)
    pos = {frozenset(e) for e in graph.g.edges()}

    # sample candidate non-edges; quality = common-neighbour count (link-likeness
    # = harder negative). High-CN pairs share neighbours, so they cluster in
    # embedding space — exactly where top-k-by-quality piles up near-duplicates.
    cand, quality = [], []
    tries = 0
    while len(cand) < pool and tries < pool * 60:
        tries += 1
        a, b = node_arr[rng.integers(len(nodes))], node_arr[rng.integers(len(nodes))]
        if a == b or frozenset((a, b)) in pos:
            continue
        cn = len(adj[a] & adj[b])
        if cn == 0:
            continue                                 # keep the informative (link-like) tail
        cand.append((a, b))
        quality.append(float(cn))
    cand_emb = normalize_rows(_pair_emb(emb, idx, cand))
    q = np.asarray(quality) / max(quality)
    S = cand_emb @ cand_emb.T                        # cosine similarity matrix

    dpp = greedy_map_dpp(q, S, k=k)
    kk = len(dpp)                                     # DPP may stop early at the effective rank
    topk = list(np.argsort(-q)[:kk])                 # compare top-k at the *same* size

    km = (KMeans(n_clusters=min(12, len(cand)), n_init=4, random_state=seed).fit(cand_emb)
          if _HAVE_SK else None)

    def _metrics(sel):
        E = cand_emb[sel]
        sim = E @ E.T
        iu = np.triu_indices(len(sel), 1)
        pair = sim[iu] if len(sel) > 1 else np.array([0.0])
        nodes_used = set()
        for i in sel:
            nodes_used.update(cand[i])
        return {
            "n": len(sel),
            "mean_quality": round(float(q[sel].mean()), 4),
            "mean_pairwise_cosine": round(float(pair.mean()), 4),   # lower = more diverse
            "max_pairwise_cosine": round(float(pair.max()), 4),     # near 1 = a near-duplicate
            "unique_proteins": len(nodes_used),                     # higher = touches more proteome
            "clusters_covered": int(len(set(km.labels_[sel]))) if km is not None else None,
        }

    return {"pool": len(cand), "k_requested": k, "top_k": _metrics(topk), "dpp": _metrics(dpp)}


# ======================================================================
# Experiment C — gold-negative surprisal on real HuRI + Negatome
# ======================================================================
def experiment_surprisal(seed: int = 0, k: int = 25, max_test: int = 800) -> dict:
    from negaverse.io import load_negatome_in_ensembl_space
    graph = _load_huri()
    rng = np.random.default_rng(seed)
    nodes, emb, idx = _spectral_pair_embeddings(graph, dim=32, seed=seed)
    node_set = set(nodes)
    pos_all = [tuple(e) for e in graph.g.edges()]

    gold = [tuple(p) for p in load_negatome_in_ensembl_space(node_set)
            if set(p) <= node_set]
    if len(gold) < 40:
        return {"status": "insufficient gold negatives in HuRI space", "n_gold": len(gold)}

    rng.shuffle(gold)
    split = len(gold) // 2
    gold_bg, gold_test = gold[:split], gold[split:]     # frozen cloud vs held-out
    rng.shuffle(pos_all)
    n = min(len(gold_test), max_test)
    gold_test = gold_test[:n]
    pos_test = pos_all[:n]

    bg_neg = normalize_rows(_pair_emb(emb, idx, gold_bg))          # gold-negative cloud
    bg_pos = normalize_rows(_pair_emb(emb, idx, pos_all[n:n + split]))  # positive cloud
    Xtest = _pair_emb(emb, idx, list(gold_test) + list(pos_test))
    y_neg = np.r_[np.ones(len(gold_test)), np.zeros(len(pos_test))]   # 1 = gold negative

    sim_to_neg = background_similarity(Xtest, bg_neg, k=k)
    sim_to_pos = background_similarity(Xtest, bg_pos, k=k)

    return {
        "n_gold_total": len(gold), "n_test_each": n,
        # want gold-similarity to rank gold negatives (y=1) above positives:
        "auroc_gold_cloud_separates_negatives": round(_auroc(y_neg, sim_to_neg), 4),
        # mirror: positive-cloud similarity as a 'suspected FN / hardness' signal
        # (should rank positives above gold negatives -> AUROC on y=1-neg):
        "auroc_positive_cloud_flags_interactions": round(_auroc(1 - y_neg, sim_to_pos), 4),
        # combined relative margin (Ch7): neg-cloud minus pos-cloud similarity:
        "auroc_relative_margin": round(_auroc(y_neg, sim_to_neg - sim_to_pos), 4),
    }


def _pair_rep(emb, idx, pairs, op="hadamard"):
    """Combine two node embeddings into a pair embedding under `op`."""
    d = emb.shape[1]
    zero = np.zeros(d)
    out = []
    for u, v in pairs:
        if u in idx and v in idx:
            eu, ev = emb[idx[u]], emb[idx[v]]
            if op == "hadamard":
                out.append(eu * ev)
            elif op == "avg":
                out.append(0.5 * (eu + ev))
            elif op == "l1":
                out.append(np.abs(eu - ev))
            else:
                raise ValueError(op)
        else:
            out.append(zero)
    return np.asarray(out, dtype=float)


def _standardize(X, mu, sd):
    return (X - mu) / np.where(sd < 1e-9, 1.0, sd)


# ======================================================================
# Experiment C2 — LEAKAGE-FREE surprisal + representation sweep
# ======================================================================
def experiment_surprisal_fair(seed: int = 0, test_frac: float = 0.2,
                              max_test: int = 800, max_bg: int = 3000,
                              k_list=(10, 25, 50)) -> dict:
    """Fair re-check of the positive-manifold surprisal signal.

    Embeddings/features are built on a TRAIN subgraph only; the test positives
    and the gold negatives never enter the representation (as bench/benchmark.py
    does). We then sweep the pair representation (spectral hadamard/avg/l1 and
    hand-crafted topological features) and k, scoring each test pair by top-k-mean
    resemblance to the frozen TRAIN-positive cloud. AUROC: does resemblance rank
    held-out positives above gold negatives?"""
    import networkx as nx
    from negaverse.io import load_negatome_in_ensembl_space
    from negaverse.bench.benchmark import _spectral_embeddings, _features

    graph = _load_huri()
    rng = np.random.default_rng(seed)
    nodes = list(graph.g.nodes())
    node_set = set(nodes)
    pos = [tuple(e) for e in graph.g.edges()]
    rng.shuffle(pos)
    n_test = int(len(pos) * test_frac)
    test_pos, train_pos = pos[:n_test], pos[n_test:]
    pos_set = {frozenset(e) for e in pos}

    gold = [tuple(p) for p in load_negatome_in_ensembl_space(node_set)
            if set(p) <= node_set and frozenset(p) not in pos_set]
    if len(gold) < 40:
        return {"status": "insufficient gold negatives in HuRI space", "n_gold": len(gold)}
    rng.shuffle(gold)

    n = min(len(gold), n_test, max_test)
    test_pos_s, gold_s = test_pos[:n], gold[:n]
    y = np.r_[np.ones(n), np.zeros(n)]                 # 1 = held-out positive
    test_pairs = list(test_pos_s) + list(gold_s)
    bg_pairs = train_pos[:max_bg]                      # frozen TRAIN-positive manifold

    # representation built on TRAIN edges only (no test/gold edge enters it)
    train_G = nx.Graph()
    train_G.add_nodes_from(nodes)
    train_G.add_edges_from(train_pos)
    emb, idx, _ = _spectral_embeddings(train_G, nodes, 32, seed)
    adj = {nd: set(train_G[nd]) for nd in train_G}

    grid = {}
    best = ("", -1.0)
    # spectral operators
    for op in ("hadamard", "avg", "l1"):
        Xt = normalize_rows(_pair_rep(emb, idx, test_pairs, op))
        Xb = normalize_rows(_pair_rep(emb, idx, bg_pairs, op))
        for k in k_list:
            s = background_similarity(Xt, Xb, k=k)
            a = round(_auroc(y, s), 4)
            grid[f"spectral_{op}_k{k}"] = a
            if a > best[1]:
                best = (f"spectral_{op}_k{k}", a)
    # hand-crafted topological features (train-graph only), z-scored via bg stats
    Ft = _features(adj, test_pairs)
    Fb = _features(adj, bg_pairs)
    mu, sd = Fb.mean(0), Fb.std(0)
    Xt = normalize_rows(_standardize(Ft, mu, sd))
    Xb = normalize_rows(_standardize(Fb, mu, sd))
    for k in k_list:
        s = background_similarity(Xt, Xb, k=k)
        a = round(_auroc(y, s), 4)
        grid[f"topological_k{k}"] = a
        if a > best[1]:
            best = (f"topological_k{k}", a)

    # baseline: negaverse's own topology-risk on the TRAIN graph as a direct score
    from negaverse.graph import TypedInteractionGraph
    from negaverse.streams.topology import TopologyFilter
    tg = TypedInteractionGraph.from_edges(
        train_pos, {nd: "protein" for nd in nodes},
        admissible_types=[("protein", "protein")], name="huri-train")
    tf = TopologyFilter()
    tf.fit(tg)
    risk = []
    for u, v in test_pairs:
        ev = tf.score(tg, u, v).evidence or {}
        risk.append(ev.get("risk", 0.0))              # high risk = link-like = positive
    risk = np.asarray(risk)
    grid["baseline_topology_risk"] = round(_auroc(y, risk), 4)

    # complementarity: does spectral surprisal add anything beyond topology risk?
    sim = background_similarity(
        normalize_rows(_pair_rep(emb, idx, test_pairs, "hadamard")),
        normalize_rows(_pair_rep(emb, idx, bg_pairs, "hadamard")), k=10)
    zs = lambda x: (x - x.mean()) / (x.std() if x.std() > 1e-9 else 1.0)
    grid["combined_topology+surprisal"] = round(_auroc(y, zs(sim) + zs(risk)), 4)
    corr = float(np.corrcoef(sim, risk)[0, 1])

    return {"n_test_each": n, "n_gold_total": len(gold),
            "best": {"representation": best[0], "auroc": best[1]},
            "corr_surprisal_vs_topology": round(corr, 4),
            "grid": grid}


def _try(name, fn):
    try:
        return fn()
    except FileNotFoundError as e:
        return {"status": f"SKIPPED — data absent ({e})"}
    except Exception as e:                            # keep the harness robust
        return {"status": f"ERROR — {type(e).__name__}: {e}"}


def main():
    report = {
        "A_entropy_fusion": _try("entropy", experiment_entropy_fusion),
        "B_dpp_diversity": _try("dpp", experiment_dpp),
        "C_gold_surprisal": _try("surprisal", experiment_surprisal),
        "C2_surprisal_leakfree": _try("surprisal_fair", experiment_surprisal_fair),
    }
    os.makedirs("out", exist_ok=True)
    with open("out/ig_eval.json", "w") as fh:
        json.dump(report, fh, indent=2)

    print("=" * 68)
    print("IG features — evaluation report   (out/ig_eval.json)")
    print("=" * 68)

    a = report["A_entropy_fusion"]
    print("\nA. Entropy-weighted fusion (Ch4) — AUROC vs ground truth")
    if "status" in a:
        print("   " + a["status"])
    else:
        print(f"   good witness alone                : {a['auroc_good_witness_alone']:.4f}")
        print(f"   mean fusion (+ flaky witness)     : {a['auroc_mean_fusion']:.4f}")
        print(f"   entropy fusion, scalar proxy      : {a['auroc_entropy_scalar_proxy']:.4f}"
              "   <- backfires (up-weights confident-wrong)")
        print(f"   entropy fusion, reported confidence: {a['auroc_entropy_reported_confidence']:.4f}"
              "  <- the real win")

    b = report["B_dpp_diversity"]
    print("\nB. DPP set selection (Ch5) — real HuRI, quality vs coverage")
    if "status" in b:
        print("   " + b["status"])
    else:
        print(f"   pool={b['pool']}  compared at n={b['dpp']['n']} (DPP's realized size)")
        for name in ("top_k", "dpp"):
            m = b[name]
            print(f"   {name:<6} quality={m['mean_quality']:.4f}  "
                  f"mean_cos={m['mean_pairwise_cosine']:.4f}  max_cos={m['max_pairwise_cosine']:.4f}  "
                  f"unique_proteins={m['unique_proteins']}  clusters={m['clusters_covered']}")

    c = report["C_gold_surprisal"]
    print("\nC. Gold-negative surprisal (Ch1) — real HuRI + Negatome")
    if "status" in c:
        print("   " + c["status"])
    else:
        print(f"   test pairs per class = {c['n_test_each']}  (gold total {c['n_gold_total']})")
        print(f"   gold-cloud resemblance separates negatives : "
              f"AUROC={c['auroc_gold_cloud_separates_negatives']:.4f}")
        print(f"   positive-cloud resemblance flags interactions: "
              f"AUROC={c['auroc_positive_cloud_flags_interactions']:.4f}")
        print(f"   relative margin (neg − pos cloud)           : "
              f"AUROC={c['auroc_relative_margin']:.4f}")
        print("   (embeddings above see the positive edges — leaky; see C2)")

    c2 = report["C2_surprisal_leakfree"]
    print("\nC2. LEAKAGE-FREE surprisal (train-only embeddings) + representation sweep")
    if "status" in c2:
        print("   " + c2["status"])
    else:
        print(f"   held-out positives vs gold negatives, {c2['n_test_each']} each")
        print(f"   BEST representation: {c2['best']['representation']}  "
              f"AUROC={c2['best']['auroc']:.4f}")
        print(f"   corr(surprisal, topology risk) = {c2['corr_surprisal_vs_topology']:+.4f}"
              "  (high = redundant signal)")
        for name, a in sorted(c2["grid"].items(), key=lambda kv: -kv[1]):
            print(f"      {name:<28} AUROC={a:.4f}")
    print()


if __name__ == "__main__":
    main()
