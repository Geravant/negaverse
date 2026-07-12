"""Step-4 precondition: does scalar PSM leave residual MULTIVARIATE imbalance?

Sinkhorn optimal-transport selection is only worth building if the shipped
scalar propensity-score matching (`_select_psm`, degree-sum) leaves a nuisance
distribution that a *multivariate* matcher would meaningfully close. We test that
directly with a classifier two-sample test (C2ST): train a classifier to tell
PSM-selected negatives from the positives, using (a) only the matched scalar
covariate vs (b) the full multivariate nuisance vector. If (b) ≫ (a), there is
residual multivariate imbalance and Sinkhorn is justified; if (b) ≈ (a), it is
not — the imbalance is scalar/structural and OT cannot fix it.

    PYTHONPATH=. python3 scripts/eval_psm_residual.py [--seeds 0 1 2]
"""
from __future__ import annotations

import argparse
import numpy as np
import networkx as nx
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

import scripts.bench_corrected as bench
from scripts.bench_corrected import _load_dataset, _score_pool, _select_psm
from negaverse.rule_engine import load_rules


def _c2st(Xa, Xb, cols, seed):
    X = np.vstack([Xa[:, cols], Xb[:, cols]])
    y = np.r_[np.zeros(len(Xa)), np.ones(len(Xb))]
    clf = RandomForestClassifier(n_estimators=150, random_state=seed, n_jobs=-1)
    return float(cross_val_score(clf, X, y, cv=4, scoring="roc_auc").mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["huri", "dryad"], default="huri")
    ap.add_argument("--max-positives", type=int, default=6000)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--max-pool", type=int, default=14000)
    ap.add_argument("--psm-cap", type=float, default=0.7)
    args = ap.parse_args()

    bench._RULES = [r for r in load_rules()
                    if r.modality == "ppi" and r.effect in ("safer_negative", "riskier_negative")]
    all_pos, gold, nodes, full = _load_dataset(args.dataset)
    print("=" * 92)
    print(f"PSM RESIDUAL-IMBALANCE C2ST — {args.dataset.upper()}  (AUC 0.5 = matched; higher = imbalance)")
    print(f"  gain = full-multivariate C2ST − scalar-degsum C2ST  (>~0.05 would justify Sinkhorn)")
    print("=" * 92)
    scalar, joint, fullc = [], [], []
    for seed in args.seeds:
        rng = np.random.default_rng(seed)
        pos = all_pos
        if args.max_positives and len(pos) > args.max_positives:
            pos = [pos[i] for i in rng.choice(len(pos), size=args.max_positives, replace=False)]
        pos = [pos[i] for i in rng.permutation(len(pos))]
        nt = int(len(pos) * 0.2)
        test_pos, tr_pos = pos[:nt], pos[nt:]
        G = nx.Graph(tr_pos); dg = dict(G.degree()); cc = nx.clustering(G)
        pool = _score_pool(tr_pos, args.max_pool, seed, full, exclude={frozenset(p) for p in test_pos})
        pos_degsums = [dg.get(u, 0) + dg.get(v, 0) for (u, v) in tr_pos]
        neg = _select_psm(pool, pos_degsums, len(tr_pos), np.random.default_rng(seed), args.psm_cap)

        def feats(pairs):
            out = []
            for (u, v) in pairs:
                du, dv = dg.get(u, 0), dg.get(v, 0)
                out.append([du + dv, min(du, dv), max(du, dv), cc.get(u, 0) + cc.get(v, 0)])
            return np.array(out, float)

        Xn, Xp = feats(neg), feats(tr_pos)
        a_sum = _c2st(Xn, Xp, [0], seed)             # matched scalar covariate
        a_joint = _c2st(Xn, Xp, [1, 2], seed)        # joint endpoint degree
        a_full = _c2st(Xn, Xp, [0, 1, 2, 3], seed)   # + clustering
        scalar.append(a_sum); joint.append(a_joint); fullc.append(a_full)
        print(f"  seed {seed}  scalar-degsum={a_sum:.3f}  joint-degree={a_joint:.3f}  "
              f"full(+clustering)={a_full:.3f}  multivariate gain={a_full - a_sum:+.3f}")
    m = lambda xs: float(np.mean(xs))
    print("-" * 92)
    print(f"  MEAN  scalar={m(scalar):.3f}  joint={m(joint):.3f}  full={m(fullc):.3f}  "
          f"multivariate gain over scalar={m(fullc) - m(scalar):+.3f}")
    print("  Verdict: gain ≪ 0.05 ⇒ no meaningful residual multivariate imbalance ⇒ Sinkhorn NOT justified.")
    print("=" * 92)


if __name__ == "__main__":
    main()
