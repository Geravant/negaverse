"""Corrected negative-sampling benchmark — removes the artifacts that made the
headline "filters worse than random" result (see docs/FILTER-EFFECTIVENESS.md).

Two independent analyses converged on the same diagnosis: the −0.097 HuRI deficit
was mostly a BENCHMARK ARTIFACT, not evidence the filters pick bad negatives:

  1. AGGRESSIVE POSITIVE CAP (6,000 of 52,068 edges) → an artificially sparse
     training graph → most proteins isolated → zero SVD embeddings → the test set
     is dominated by an "either endpoint isolated ⇒ negative" shortcut. Random
     negatives (81% all-zero features) reproduce that shortcut; topology-hard
     negatives (0% isolated, since topology can't call an isolated pair hard)
     never learn it. ~85% of the deficit rode on this. Raising to 20k positives
     already flipped Δ to +0.007.
  2. UNEQUAL POOLS: random didn't get the external veto, so it accidentally
     included ~35-44 known positives and ~4-7 full-HuRI edges — dirtier than the
     veto-cleaned topology set, yet scored higher. AUROC rewarded the shortcut,
     not purity.
  3. 100% HARD-TAIL REPLACEMENT: selecting only the topology-hardest negatives is
     a narrow, hidden-positive-enriched distribution — risky even once the
     artifacts are fixed.

This bench fixes all three:
  * ONE frozen, veto-cleaned candidate pool shared by every arm (equal purity);
  * configurable, un-capped-by-default positive set (`--max-positives 0` = full);
  * five arms drawn from that same pool: raw random, veto random, topology-HARD,
    topology-SAFE (highest-confidence across the FULL pool — the representative
    clean selection the pipeline never offered), and stacked (hard tail re-ranked
    by fused biology confidence);
  * degree/coverage-stratified reporting (all pairs vs both-endpoints-non-isolated),
    which isolates the shortcut;
  * AUROC + AUPRC + PPIHits@TopK/PPNIHits@BottomK.

    PYTHONPATH=. python3 scripts/bench_corrected.py [--max-positives 20000] [--seeds 0 1 2]
"""
from __future__ import annotations

import argparse
import numpy as np
import networkx as nx
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, average_precision_score

from negaverse.bench.benchmark import _spectral_embeddings, _hadamard_features
from negaverse.candidates import generate_candidates
from negaverse.graph import TypedInteractionGraph
from negaverse.io import load_huri_graph, load_negatome_in_ensembl_space
from negaverse.rule_engine import load_rules
from negaverse.streams import build_filters
from negaverse.streams.rules import RuleGradedFilter

ARMS = ["random_raw", "random_veto", "topology_hard", "topology_safe", "stacked"]


def _score_pool(train_pos, max_pool, seed, full_pos_set):
    """One frozen candidate pool, each pair scored ONCE. Returns dicts with
    confidence (mean fuse of graded values), hardness (topology-risk percentile),
    and a `dirty` flag (pair is actually a positive somewhere — veto target)."""
    nodes = {p for e in train_pos for p in e}
    tg = TypedInteractionGraph.from_edges(
        list(train_pos), {n: "protein" for n in nodes},
        admissible_types=[("protein", "protein")], name="pool")
    veto = build_filters("ppi", ["known_positive_veto"])[0]
    graded = build_filters("ppi", ["structured", "topology"]) + [RuleGradedFilter(rules=_RULES)]
    for f in [veto] + graded:
        f.fit(tg)

    cand = generate_candidates(tg, max_pool=max_pool, seed=seed)
    rows, risks = [], []
    for (u, v) in cand:
        sc = veto.score(tg, u, v)
        vetoed = bool(sc.veto)                                  # known positive (DB or graph)
        vals, risk = [], 0.0
        for f in graded:
            s = f.score(tg, u, v)
            if s.value is not None:
                vals.append(s.value)
            if f.name == "topology":
                risk = float((s.evidence or {}).get("risk", 0.0))
        conf = float(np.mean(vals)) if vals else 0.5
        rows.append({"u": u, "v": v, "conf": conf, "risk": risk, "vetoed": vetoed,
                     "dirty": vetoed or (frozenset((u, v)) in full_pos_set)})
        risks.append(risk)
    # hardness = percentile of topology risk across the pool (matches the pipeline)
    order = np.argsort(np.argsort(risks))
    for i, r in enumerate(rows):
        r["hardness"] = order[i] / max(len(rows) - 1, 1)
    return rows


def _select(pool, arm, n, rng):
    clean = [r for r in pool if not r["vetoed"]]               # veto-cleaned pool
    if arm == "random_raw":
        idx = rng.choice(len(pool), size=min(n, len(pool)), replace=False)
        return [(pool[i]["u"], pool[i]["v"]) for i in idx]
    if arm == "random_veto":
        idx = rng.choice(len(clean), size=min(n, len(clean)), replace=False)
        return [(clean[i]["u"], clean[i]["v"]) for i in idx]
    if arm == "topology_hard":
        s = sorted(clean, key=lambda r: r["hardness"], reverse=True)
        return [(r["u"], r["v"]) for r in s[:n]]
    if arm == "topology_safe":
        s = sorted(clean, key=lambda r: r["conf"], reverse=True)   # most-confident SAFE negatives
        return [(r["u"], r["v"]) for r in s[:n]]
    if arm == "stacked":
        hard = sorted(clean, key=lambda r: r["hardness"], reverse=True)[:max(4 * n, n)]
        hard.sort(key=lambda r: r["conf"], reverse=True)          # re-rank hard tail by biology conf
        return [(r["u"], r["v"]) for r in hard[:n]]
    raise ValueError(arm)


def _hits(y, s, k, positive=True):
    order = np.argsort(s)
    idx = order[-k:] if positive else order[:k]
    return float(np.mean(y[idx] == (1 if positive else 0)))


def _evaluate(train_pos, train_neg, nodes, gold_test, seed, emb_dim=32):
    rng = np.random.default_rng(seed)
    tp = list(train_pos)
    n_test = int(len(tp) * 0.2)
    rng.shuffle(tp)
    test_pos, tr_pos = tp[:n_test], tp[n_test:]
    n_bal = min(len(test_pos), len(gold_test))
    test_pos, test_neg = test_pos[:n_bal], list(gold_test)[:n_bal]

    train_G = nx.Graph(); train_G.add_nodes_from(nodes); train_G.add_edges_from(tr_pos)
    emb, idx, _ = _spectral_embeddings(train_G, nodes, emb_dim, seed)
    feat = lambda pairs: _hadamard_features(emb, idx, pairs)
    deg = dict(train_G.degree())
    Xtr = feat(list(tr_pos) + list(train_neg))
    ytr = np.r_[np.ones(len(tr_pos)), np.zeros(len(train_neg))]
    clf = RandomForestClassifier(n_estimators=200, random_state=seed, n_jobs=-1)
    clf.fit(Xtr, ytr)

    te_pairs = list(test_pos) + list(test_neg)
    Xte = feat(te_pairs)
    yte = np.r_[np.ones(len(test_pos)), np.zeros(len(test_neg))].astype(int)
    p = clf.predict_proba(Xte)[:, 1]
    out = {"auroc": roc_auc_score(yte, p), "auprc": average_precision_score(yte, p),
           "ppi@100": _hits(yte, p, min(100, len(test_pos), len(test_neg)), True),
           "ppni@100": _hits(yte, p, min(100, len(test_pos), len(test_neg)), False)}
    # non-isolated stratum: both endpoints have a train-graph edge (no zero-feature shortcut)
    keep = np.array([deg.get(u, 0) > 0 and deg.get(v, 0) > 0 for (u, v) in te_pairs])
    if keep.sum() > 10 and len(set(yte[keep])) == 2:
        out["auroc_noniso"] = roc_auc_score(yte[keep], p[keep])
        out["frac_noniso_test"] = float(keep.mean())
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-positives", type=int, default=20000, help="0 = full HuRI")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--max-pool", type=int, default=200000)
    args = ap.parse_args()

    global _RULES
    _RULES = [r for r in load_rules()
              if r.modality == "ppi" and r.effect in ("safer_negative", "riskier_negative")]

    g = load_huri_graph()
    gold = [tuple(p) for p in load_negatome_in_ensembl_space(set(g.g.nodes()))]
    full_pos_set = {frozenset(e) for e in g.g.edges()}         # FULL HuRI, for purity check
    all_pos = [tuple(e) for e in g.g.edges()]
    nodes = list(g.g.nodes())
    print("=" * 92)
    print("CORRECTED benchmark — HuRI, one frozen veto-cleaned pool, un-capped positives")
    print(f"full HuRI edges={len(all_pos)}  max_positives={args.max_positives or 'ALL'}  "
          f"gold={len(gold)}  seeds={args.seeds}")
    print("=" * 92)

    metrics = {a: {k: [] for k in ["auroc", "auprc", "ppi@100", "ppni@100", "auroc_noniso"]}
               for a in ARMS}
    purity = {a: [] for a in ARMS}
    for seed in args.seeds:
        rng = np.random.default_rng(seed)
        pos = all_pos
        if args.max_positives and len(pos) > args.max_positives:
            pos = [pos[i] for i in rng.choice(len(pos), size=args.max_positives, replace=False)]
        gold_s = [p for p in gold if frozenset(p) not in full_pos_set]
        rng.shuffle(gold_s)
        pool = _score_pool(pos, args.max_pool, seed, full_pos_set)   # ONE pool, all arms share it
        n_neg = int(len(pos) * 0.8)
        for arm in ARMS:
            neg = _select(pool, arm, n_neg, np.random.default_rng(seed))
            if not neg:
                continue
            dirty = sum(1 for u, v in neg if frozenset((u, v)) in full_pos_set)
            purity[arm].append(dirty)
            res = _evaluate(pos, neg, nodes, gold_s, seed)
            for k, val in res.items():
                if k in metrics[arm]:
                    metrics[arm][k].append(val)
            print(f"  seed {seed}  {arm:<15} AUROC={res['auroc']:.3f} "
                  f"noniso={res.get('auroc_noniso', float('nan')):.3f}  "
                  f"n_neg={len(neg)} hidden_pos_in_neg={dirty}")

    mean = lambda a, k: float(np.mean(metrics[a][k])) if metrics[a][k] else float("nan")
    print("\n" + "=" * 92)
    print("  CORRECTED results — mean over seeds. 'noniso' = AUROC on both-endpoints-non-isolated")
    print("  test pairs (removes the zero-feature isolation shortcut).")
    print("-" * 92)
    hdr = f"  {'arm':<15}{'AUROC':>9}{'AUPRC':>9}{'AUROC_noniso':>14}{'PPIHit@100':>12}{'PPNIHit@100':>13}{'hidden+':>9}"
    print(hdr); print("  " + "-" * (len(hdr) - 2))
    for a in ARMS:
        hp = float(np.mean(purity[a])) if purity[a] else float("nan")
        print(f"  {a:<15}{mean(a,'auroc'):>9.3f}{mean(a,'auprc'):>9.3f}"
              f"{mean(a,'auroc_noniso'):>14.3f}{mean(a,'ppi@100'):>12.3f}"
              f"{mean(a,'ppni@100'):>13.3f}{hp:>9.1f}")
    print("-" * 92)
    r = mean("random_veto", "auroc")
    print(f"  Δ AUROC vs veto-random:  " + "   ".join(
        f"{a} {mean(a,'auroc')-r:+.3f}" for a in ("topology_hard", "topology_safe", "stacked")))
    rn = mean("random_veto", "auroc_noniso")
    print(f"  Δ AUROC_noniso vs veto-random:  " + "   ".join(
        f"{a} {mean(a,'auroc_noniso')-rn:+.3f}" for a in ("topology_hard", "topology_safe", "stacked")))
    print("\n  hidden+ = mean # full-HuRI positives leaked into the negative set (purity; lower=cleaner)")


if __name__ == "__main__":
    main()
