"""Per-rule ablation at the WHOLE-STACK, DOWNSTREAM-VALUE level — the question
bench_rules.py defers to negaverse.bench as "the final word" but never actually asks.

bench_rules.py answers "does rule R separate positives from hard negatives?" (Q1,
a proxy, measured on the rule set alone). This asks the real question: does
INCLUDING rule R in the full pipeline (veto + structured + topology + graded rules)
produce training negatives that train a BETTER downstream link-predictor (Q2)?

For each graded rule R we run the full stacked-negative selection twice — once with
ALL rules, once with ALL-BUT-R — feed each negative set to the same downstream model
(spectral features, independent of the selection signal; tested on held-out
positives + Negatome GOLD negatives), and report:

    Δ_downstream(R) = AUROC(all rules) − AUROC(all rules minus R)

Positive Δ => rule R's presence in the stack improves downstream negatives (it earns
its place). ~0 => it's inert downstream. Negative => it HURTS and should be dropped.
Also reports the ALL-rules vs NO-rules endpoints (the graded layer's total worth).

    PYTHONPATH=. python3 scripts/bench_rule_ablation_downstream.py [--seeds 0 1 2] [--n 4000]

No LLM — rules are cheap. Spectral features + gold test = no circularity.
"""
from __future__ import annotations

import argparse
import numpy as np
import networkx as nx
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

from negaverse.bench.benchmark import _spectral_embeddings, _hadamard_features
from negaverse.graph import TypedInteractionGraph
from negaverse.io import load_huri_graph, load_negatome_in_ensembl_space
from negaverse.pipeline import PipelineConfig, run_pipeline
from negaverse.rule_engine import load_rules
from negaverse.streams import build_filters
from negaverse.streams.rules import RuleGradedFilter


def _stacked_negatives(train_pos, node_type, n, seed, max_pool, rule_subset):
    """Same selection as benchmark._negaverse_stacked_negatives, but the graded
    rule layer uses exactly `rule_subset` (None => no graded rules at all).
    Oversamples the pool 4x (== the real stacked strategy) so top-n is a genuine
    most-confident selection, not just 'whatever the pipeline emitted'."""
    tg = TypedInteractionGraph.from_edges(
        train_pos, dict(node_type), admissible_types=[("protein", "protein")], name="abl-train")
    filters = build_filters("ppi", ["known_positive_veto", "structured", "topology"])
    if rule_subset:                                    # inject a graded filter with just this subset
        filters = filters + [RuleGradedFilter(rules=list(rule_subset))]
    cfg = PipelineConfig(modality="ppi", n_eval=0, n_train=max(4 * n, n), max_pool=max_pool, seed=seed)
    res = run_pipeline(tg, cfg, filters=filters)
    hard = [r for r in res.records if r.mode == "train"]
    hard.sort(key=lambda r: r.confidence, reverse=True)
    return [(r.u, r.v) for r in hard[:n]]


def _auroc_for_negatives(train_pos, train_neg, featurize, Xte, yte, seed):
    """Downstream link-predictor AUROC. `featurize` is a single closure over the
    seed's spectral embedding — the SAME one used for Xte, so train and test
    features are guaranteed to share one basis."""
    Xtr = featurize(list(train_pos) + list(train_neg))
    ytr = np.r_[np.ones(len(train_pos)), np.zeros(len(train_neg))]
    clf = RandomForestClassifier(n_estimators=200, random_state=seed, n_jobs=-1)
    clf.fit(Xtr, ytr)
    return float(roc_auc_score(yte, clf.predict_proba(Xte)[:, 1]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--n", type=int, default=4000, help="#training negatives per config")
    ap.add_argument("--max-positives", type=int, default=6000)
    ap.add_argument("--max-pool", type=int, default=40000)
    ap.add_argument("--emb-dim", type=int, default=32)
    args = ap.parse_args()

    graph = load_huri_graph()
    gold = load_negatome_in_ensembl_space(set(graph.g.nodes()))
    rules = [r for r in load_rules()
             if r.modality == "ppi" and r.effect in ("safer_negative", "riskier_negative")]
    print("=" * 80)
    print("Per-rule downstream ablation — HuRI, spectral features, GOLD (Negatome) test")
    print(f"graded ppi rules: {[r.id for r in rules]}")
    print(f"seeds={args.seeds}  n_train_neg={args.n}  gold={len(gold)}")
    print("=" * 80)

    node_type = {n: "protein" for n in graph.g.nodes()}
    nodes = list(graph.g.nodes())
    # configs: all rules, each leave-one-out, and none
    configs = {"ALL": rules, "NONE": []}
    for r in rules:
        configs[f"-{r.id}"] = [x for x in rules if x.id != r.id]

    scores = {k: [] for k in configs}
    for seed in args.seeds:
        rng = np.random.default_rng(seed)
        pos = [tuple(e) for e in graph.g.edges()]
        if args.max_positives and len(pos) > args.max_positives:
            pos = [pos[i] for i in rng.choice(len(pos), size=args.max_positives, replace=False)]
        pos = [pos[i] for i in rng.permutation(len(pos))]
        n_test = int(len(pos) * 0.2)
        test_pos, train_pos = pos[:n_test], pos[n_test:]
        pos_set = {frozenset(e) for e in pos}
        node_set = set(nodes)
        g = [tuple(p) for p in gold if p not in pos_set and set(p) <= node_set]
        rng.shuffle(g)
        n_bal = min(len(test_pos), len(g))
        test_pos, test_neg = test_pos[:n_bal], g[:n_bal]
        # one spectral embedding per seed; ONE featurize closure for train AND test
        train_G = nx.Graph(); train_G.add_nodes_from(nodes); train_G.add_edges_from(train_pos)
        emb, idx, _ = _spectral_embeddings(train_G, nodes, args.emb_dim, seed)
        featurize = lambda pairs: _hadamard_features(emb, idx, pairs)
        Xte = featurize(list(test_pos) + list(test_neg))
        yte = np.r_[np.ones(len(test_pos)), np.zeros(len(test_neg))]

        for name, subset in configs.items():
            neg = _stacked_negatives(train_pos, node_type, args.n, seed, args.max_pool, subset)
            if not neg:
                continue
            auroc = _auroc_for_negatives(train_pos, neg, featurize, Xte, yte, seed)
            scores[name].append(auroc)
            print(f"  seed {seed}  {name:<28} AUROC={auroc:.4f}  n_neg={len(neg)}")

    mean = {k: float(np.mean(v)) if v else float("nan") for k, v in scores.items()}
    all_a, none_a = mean["ALL"], mean["NONE"]
    print("-" * 80)
    print(f"  {'config':<28}{'mean AUROC':>12}{'Δ vs ALL':>12}")
    print(f"  {'ALL rules':<28}{all_a:>12.4f}{'':>12}")
    print(f"  {'NONE (structured+topology)':<28}{none_a:>12.4f}{none_a-all_a:>+12.4f}")
    print("  " + "-" * 52)
    print(f"  {'leave-one-out':<28}{'':>12}{'Δ_downstream(R)':>16}")
    print("  (Δ_downstream(R) = AUROC(ALL) − AUROC(ALL−R);  >0 => rule R helps the stack)")
    loo = sorted(((r.id, all_a - mean[f"-{r.id}"]) for r in rules), key=lambda kv: -kv[1])
    for rid, d in loo:
        verdict = "helps" if d > 0.002 else ("HURTS" if d < -0.002 else "inert")
        print(f"  {'  -'+rid:<28}{mean['-'+rid]:>12.4f}{d:>+13.4f}  {verdict}")
    print("-" * 80)
    print(f"\n  graded-rule layer total worth (ALL − NONE) = {all_a - none_a:+.4f}")


if __name__ == "__main__":
    main()
