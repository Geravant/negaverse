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

from sklearn.linear_model import LogisticRegression  # noqa: F401 (available for extra rows)
from negaverse.bench.benchmark import _spectral_embeddings, _hadamard_features


def _make_model(kind: str, seed: int):
    """Fresh classifier. Model-sensitivity: does the negative-selection verdict
    survive changing the downstream learner (RF -> boosted trees)?"""
    if kind == "rf":
        return RandomForestClassifier(n_estimators=200, random_state=seed, n_jobs=-1)
    if kind == "lgbm":
        import lightgbm as lgb
        return lgb.LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=31,
                                  random_state=seed, n_jobs=-1, verbose=-1)
    raise ValueError(kind)
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
    structured, topology = build_filters("ppi", ["structured", "topology"])
    # score each rule SEPARATELY so any subset's fused confidence can be recomputed
    # without re-scoring the pool (RuleGradedFilter fuses fired rules as Σ(w·v)/Σ(w)).
    per_rule = {r.id: RuleGradedFilter(rules=[r]) for r in _RULES}
    for f in [veto, structured, topology] + list(per_rule.values()):
        f.fit(tg)

    cand = generate_candidates(tg, max_pool=max_pool, seed=seed)
    rows, risks = [], []
    for (u, v) in cand:
        vetoed = bool(veto.score(tg, u, v).veto)                # known positive (DB or graph)
        s_val = structured.score(tg, u, v).value
        t = topology.score(tg, u, v)
        t_val, risk = t.value, float((t.evidence or {}).get("risk", 0.0))
        rule_vals = {}                                          # rid -> (weight, value) for FIRED rules
        for r in _RULES:
            rs = per_rule[r.id].score(tg, u, v)
            if rs.value is not None:
                rule_vals[r.id] = (r.weight, rs.value)
        rows.append({"u": u, "v": v, "struct": s_val, "topo": t_val, "risk": risk,
                     "rules": rule_vals, "vetoed": vetoed,
                     "dirty": vetoed or (frozenset((u, v)) in full_pos_set)})
        risks.append(risk)
    # hardness = percentile of topology risk across the pool (matches the pipeline)
    order = np.argsort(np.argsort(risks))
    for i, r in enumerate(rows):
        r["hardness"] = order[i] / max(len(rows) - 1, 1)
        r["conf"] = _conf(r, [x.id for x in _RULES])           # default conf = all rules
    return rows


def _conf(row, rule_ids):
    """Fused mean confidence for a given rule subset (matches _fuse_confidence's
    default 'mean'): mean of {structured, topology, combined-rules-value}, skipping
    abstentions. Combined-rules-value = Σ(w·v)/Σ(w) over the subset's FIRED rules."""
    vals = [x for x in (row["struct"], row["topo"]) if x is not None]
    fired = [(w, v) for rid, (w, v) in row["rules"].items() if rid in rule_ids]
    if fired:
        num = sum(w * v for w, v in fired); den = sum(w for w, _ in fired) or len(fired)
        vals.append(num / den)
    return float(np.mean(vals)) if vals else 0.5


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


def _select_stacked_subset(pool, n, rule_ids):
    """Stacked selection where the hard-tail re-ranking uses only `rule_ids` — for
    per-rule leave-one-out ablation."""
    clean = [r for r in pool if not r["vetoed"]]
    hard = sorted(clean, key=lambda r: r["hardness"], reverse=True)[:max(4 * n, n)]
    hard.sort(key=lambda r: _conf(r, rule_ids), reverse=True)
    return [(r["u"], r["v"]) for r in hard[:n]]


def _hits(y, s, k, positive=True):
    order = np.argsort(s)
    idx = order[-k:] if positive else order[:k]
    return float(np.mean(y[idx] == (1 if positive else 0)))


def _evaluate(train_pos, train_neg, nodes, gold_test, seed, models, emb_dim=32):
    """Returns {model_kind: metric_dict}. Same features/split across models —
    only the learner changes (model-sensitivity)."""
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
    te_pairs = list(test_pos) + list(test_neg)
    Xte = feat(te_pairs)
    yte = np.r_[np.ones(len(test_pos)), np.zeros(len(test_neg))].astype(int)
    keep = np.array([deg.get(u, 0) > 0 and deg.get(v, 0) > 0 for (u, v) in te_pairs])

    result = {}
    for kind in models:
        clf = _make_model(kind, seed)
        clf.fit(Xtr, ytr)
        p = clf.predict_proba(Xte)[:, 1]
        out = {"auroc": roc_auc_score(yte, p), "auprc": average_precision_score(yte, p),
               "ppi@100": _hits(yte, p, min(100, len(test_pos), len(test_neg)), True),
               "ppni@100": _hits(yte, p, min(100, len(test_pos), len(test_neg)), False)}
        if keep.sum() > 10 and len(set(yte[keep])) == 2:
            out["auroc_noniso"] = roc_auc_score(yte[keep], p[keep])
        result[kind] = out
    return result


def _load_dataset(name):
    """(all_pos edges, gold-negative pairs, node list, full_pos_set)."""
    if name == "huri":
        g = load_huri_graph()
        gold = [tuple(p) for p in load_negatome_in_ensembl_space(set(g.g.nodes()))]
        pos = [tuple(e) for e in g.g.edges()]
        return pos, gold, list(g.g.nodes()), {frozenset(e) for e in g.g.edges()}
    if name == "dryad":
        pos, neg = [], []
        with open("local-docs/dryad-ppi/benchmarks/benchmarks/positives_and_negatives.tsv") as fh:
            next(fh)
            for line in fh:
                pair, cat = line.rstrip("\n").split("\t")
                a, b = pair.split("_")
                (pos if cat == "positive" else neg).append((a, b))
        nodes = sorted({p for e in pos + neg for p in e})
        return pos, neg, nodes, {frozenset(e) for e in pos}
    raise ValueError(name)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["huri", "dryad"], default="huri")
    ap.add_argument("--max-positives", type=int, default=20000, help="0 = all")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--max-pool", type=int, default=200000)
    ap.add_argument("--models", nargs="+", default=["rf", "lgbm"], choices=["rf", "lgbm"])
    ap.add_argument("--rule-ablation", action="store_true",
                    help="leave each graded rule out of the stacked arm one-by-one")
    args = ap.parse_args()

    global _RULES
    _RULES = [r for r in load_rules()
              if r.modality == "ppi" and r.effect in ("safer_negative", "riskier_negative")]

    # arm_list + spec: "builtin" -> _select; None -> veto random; list[str] -> stacked with that rule subset
    if args.rule_ablation:
        all_ids = [r.id for r in _RULES]
        spec = {"random_veto": None, "stacked[ALL rules]": all_ids}
        for r in _RULES:
            spec[f"stacked[-{r.id}]"] = [x for x in all_ids if x != r.id]
        spec["stacked[NO rules]"] = []
    else:
        spec = {a: "builtin" for a in ARMS}
    arm_list = list(spec)

    all_pos, gold, nodes, full_pos_set = _load_dataset(args.dataset)
    print("=" * 92)
    print(f"CORRECTED benchmark — {args.dataset.upper()}, one frozen veto-cleaned pool"
          f"{'  [RULE ABLATION]' if args.rule_ablation else ''}")
    print(f"full edges={len(all_pos)}  nodes={len(nodes)}  max_positives={args.max_positives or 'ALL'}  "
          f"gold={len(gold)}  seeds={args.seeds}  rules={[r.id for r in _RULES]}")
    print("=" * 92)

    # metrics[(arm, model)][metric] = [per-seed values]
    metrics = {(a, m): {k: [] for k in ["auroc", "auprc", "ppi@100", "ppni@100", "auroc_noniso"]}
               for a in arm_list for m in args.models}
    purity = {a: [] for a in arm_list}
    print(f"models: {args.models}")
    for seed in args.seeds:
        rng = np.random.default_rng(seed)
        pos = all_pos
        if args.max_positives and len(pos) > args.max_positives:
            pos = [pos[i] for i in rng.choice(len(pos), size=args.max_positives, replace=False)]
        gold_s = [p for p in gold if frozenset(p) not in full_pos_set]
        rng.shuffle(gold_s)
        pool = _score_pool(pos, args.max_pool, seed, full_pos_set)   # ONE pool, all arms share it
        n_neg = int(len(pos) * 0.8)
        for arm in arm_list:
            sp = spec[arm]
            if sp == "builtin":
                neg = _select(pool, arm, n_neg, np.random.default_rng(seed))
            elif sp is None:
                neg = _select(pool, "random_veto", n_neg, np.random.default_rng(seed))
            else:
                neg = _select_stacked_subset(pool, n_neg, sp)
            if not neg:
                continue
            dirty = sum(1 for u, v in neg if frozenset((u, v)) in full_pos_set)
            purity[arm].append(dirty)
            per_model = _evaluate(pos, neg, nodes, gold_s, seed, args.models)
            for m, res in per_model.items():
                for k, val in res.items():
                    if k in metrics[(arm, m)]:
                        metrics[(arm, m)][k].append(val)
            au = "  ".join(f"{m}={per_model[m]['auroc']:.3f}" for m in args.models)
            print(f"  seed {seed}  {arm:<28} AUROC[{au}]  n_neg={len(neg)} hidden_pos={dirty}")

    mean = lambda a, m, k: float(np.mean(metrics[(a, m)][k])) if metrics[(a, m)][k] else float("nan")
    w = 34 if args.rule_ablation else 15
    ref = "stacked[ALL rules]" if args.rule_ablation else "random_veto"
    for m in args.models:
        print("\n" + "=" * 100)
        print(f"  CORRECTED results — model={m.upper()}, mean over seeds. 'noniso' = both-endpoints-non-isolated.")
        print("-" * 100)
        hdr = f"  {'arm':<{w}}{'AUROC':>9}{'AUPRC':>9}{'AUROC_noniso':>14}{'PPIHit@100':>12}{'PPNIHit@100':>13}{'hidden+':>9}"
        print(hdr); print("  " + "-" * (len(hdr) - 2))
        for a in arm_list:
            hp = float(np.mean(purity[a])) if purity[a] else float("nan")
            print(f"  {a:<{w}}{mean(a,m,'auroc'):>9.3f}{mean(a,m,'auprc'):>9.3f}"
                  f"{mean(a,m,'auroc_noniso'):>14.3f}{mean(a,m,'ppi@100'):>12.3f}"
                  f"{mean(a,m,'ppni@100'):>13.3f}{hp:>9.1f}")
        print("-" * 100)
        if args.rule_ablation:
            ra, rn = mean(ref, m, "auroc"), mean(ref, m, "auroc_noniso")
            print(f"  Δ vs {ref} (a rule that HELPS the stack shows a NEGATIVE Δ when removed):")
            for a in arm_list:
                if a.startswith("stacked[-"):
                    print(f"    {a:<34} Δ AUROC {mean(a,m,'auroc')-ra:+.4f}   "
                          f"Δ noniso {mean(a,m,'auroc_noniso')-rn:+.4f}")
        else:
            r, rn = mean("random_veto", m, "auroc"), mean("random_veto", m, "auroc_noniso")
            print("  Δ AUROC vs veto-random:  " + "   ".join(
                f"{a} {mean(a,m,'auroc')-r:+.3f}" for a in ("topology_hard", "topology_safe", "stacked")))
            print("  Δ AUROC_noniso vs veto-random:  " + "   ".join(
                f"{a} {mean(a,m,'auroc_noniso')-rn:+.3f}" for a in ("topology_hard", "topology_safe", "stacked")))
    print("\n  hidden+ = mean # true positives leaked into the negative set (purity; lower=cleaner)")


if __name__ == "__main__":
    main()
