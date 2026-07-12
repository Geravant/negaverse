"""Step-3 evaluation: Fisher-information (D-optimal) batch selection.

Fisher selection presupposes a DIFFERENTIABLE model; the bench's downstream
learners are RF/LGBM (no θ-gradient). So the Fisher weights come from a logistic
SURROGATE on the same spectral-Hadamard features, and the selected batch is
transferred to the tree model — exactly the reframe from the brainstorm.

Compares, at matched scale (one negative per training positive), the downstream
AUROC + hidden-positive injection rate + parameter-space diversity of:
  stacked (baseline) · topology_hard · random_veto · fisher (weighted) ·
  fisher_unweighted (pure D-optimal / de-dup only).

    PYTHONPATH=. python3 scripts/eval_fisher.py [--dataset huri|dryad]
        [--max-positives 8000] [--seeds 0 1 2] [--max-pool 12000] [--inject-k 600]
"""
from __future__ import annotations

import argparse
import numpy as np

import scripts.bench_corrected as bench
from scripts.bench_corrected import _conf, _evaluate, _load_dataset, _score_pool, _select
from negaverse.bench.benchmark import _spectral_embeddings, _hadamard_features
from negaverse.ig.fisher import greedy_fisher_design, logistic_fisher_weights
from negaverse.rule_engine import load_rules


def _logdet(X, ridge=1.0):
    """Parameter-space diversity proxy: mean log-eigenvalue of the (standardised)
    information matrix — bounded regardless of batch size, higher = more spread."""
    if len(X) == 0:
        return float("nan")
    Xs = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-9)   # unit rows
    d = Xs.shape[1]
    M = np.eye(d) * ridge + Xs.T @ Xs
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        return float(np.mean(np.log(np.linalg.eigvalsh(M).clip(min=1e-12))))


def _fisher_select(clean, X_clean, tr_pos, emb, idx, n, weighted, rng, l2=1.0,
                   hardness_cap=None):
    """Fit the logistic surrogate (edges=1 vs sampled clean candidates=0) on
    Hadamard features, take Fisher weights p(1−p), and D-optimally pick n.

    hardness_cap restricts the candidate set to the SAFE region first (hardness ≤
    cap) — the safety gate the brainstorm requires *before* maximising log-det, so
    the D-optimal batch can't chase boundary points that are hidden positives."""
    sub = [i for i, r in enumerate(clean) if hardness_cap is None or r["hardness"] <= hardness_cap]
    Xsub = X_clean[sub]
    Xpos = _hadamard_features(emb, idx, list(tr_pos))
    m = min(len(sub), len(Xpos))
    bg = rng.choice(len(sub), size=m, replace=False)
    X_fit = np.vstack([Xpos, Xsub[bg]])
    y_fit = np.r_[np.ones(len(Xpos)), np.zeros(m)]
    w, p = logistic_fisher_weights(X_fit, y_fit, Xsub, l2=l2)
    weights = w if weighted else None
    sel = greedy_fisher_design(Xsub, weights=weights, k=n, ridge=1.0)
    return [(clean[sub[i]]["u"], clean[sub[i]]["v"]) for i in sel], Xsub[sel]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["huri", "dryad"], default="huri")
    ap.add_argument("--max-positives", type=int, default=8000)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--max-pool", type=int, default=12000)
    ap.add_argument("--inject-k", type=int, default=600)
    ap.add_argument("--emb-dim", type=int, default=32)
    ap.add_argument("--hardness-cap", type=float, default=0.7, help="safety gate for fisher_safe")
    ap.add_argument("--models", nargs="+", default=["rf", "lgbm"], choices=["rf", "lgbm"])
    args = ap.parse_args()

    bench._RULES = [r for r in load_rules()
                    if r.modality == "ppi" and r.effect in ("safer_negative", "riskier_negative")]
    rule_ids = [r.id for r in bench._RULES]
    arms = ["random_veto", "topology_hard", "stacked", "fisher", "fisher_unweighted", "fisher_safe"]

    all_pos, gold, nodes, full_pos_set = _load_dataset(args.dataset)
    print("=" * 96)
    print(f"FISHER (D-OPTIMAL) SELECTION EVAL — {args.dataset.upper()}  emb_dim={args.emb_dim}  models={args.models}")
    print(f"full edges={len(all_pos)} gold={len(gold)} max_positives={args.max_positives} seeds={args.seeds}")
    print("=" * 96)

    au = {(a, m): [] for a in arms for m in args.models}
    inj = {a: [] for a in arms}
    ld = {a: [] for a in arms}
    for seed in args.seeds:
        rng = np.random.default_rng(seed)
        pos = all_pos
        if args.max_positives and len(pos) > args.max_positives:
            pos = [pos[i] for i in rng.choice(len(pos), size=args.max_positives, replace=False)]
        pos = [pos[i] for i in rng.permutation(len(pos))]
        n_test = int(len(pos) * 0.2)
        test_pos, tr_pos = pos[:n_test], pos[n_test:]
        test_excl = {frozenset(p) for p in test_pos}
        tr_nodes = {n for e in tr_pos for n in e}
        gold_s = [p for p in gold if frozenset(p) not in full_pos_set]
        # hidden-positive injection (veto-bypassed), for the selection-rate metric
        held = [tuple(p) for p in full_pos_set
                if frozenset(p) not in {frozenset(e) for e in tr_pos}
                and frozenset(p) not in test_excl and set(p) <= tr_nodes and len(p) == 2]
        rng.shuffle(held)
        inject = held[:args.inject_k]
        inj_keys = {frozenset(p) for p in inject}
        pool = _score_pool(tr_pos, args.max_pool, seed, full_pos_set, exclude=test_excl, inject=inject)
        clean = [r for r in pool if not r["vetoed"] or r.get("injected")]
        n_inj = sum(1 for r in pool if r.get("injected"))
        n_neg = len(tr_pos)

        # spectral-Hadamard pair-embeddings on the train graph (surrogate features)
        import networkx as nx
        train_G = nx.Graph(); train_G.add_nodes_from(nodes); train_G.add_edges_from(tr_pos)
        emb, idx, _ = _spectral_embeddings(train_G, nodes, args.emb_dim, seed)
        X_clean = _hadamard_features(emb, idx, [(r["u"], r["v"]) for r in clean])

        for a in arms:
            if a == "fisher":
                neg, Xsel = _fisher_select(clean, X_clean, tr_pos, emb, idx, n_neg, True, np.random.default_rng(seed))
            elif a == "fisher_unweighted":
                neg, Xsel = _fisher_select(clean, X_clean, tr_pos, emb, idx, n_neg, False, np.random.default_rng(seed))
            elif a == "fisher_safe":
                neg, Xsel = _fisher_select(clean, X_clean, tr_pos, emb, idx, n_neg, False,
                                           np.random.default_rng(seed), hardness_cap=args.hardness_cap)
            else:
                neg = _select(pool, a, n_neg, np.random.default_rng(seed))
                key2i = {(r["u"], r["v"]): i for i, r in enumerate(clean)}
                Xsel = X_clean[[key2i[uv] for uv in neg if uv in key2i]]
            sel_inj = sum(1 for uv in neg if frozenset(uv) in inj_keys)
            inj[a].append(sel_inj / max(n_inj, 1))
            ld[a].append(_logdet(Xsel))
            res = _evaluate(tr_pos, test_pos, neg, nodes, gold_s, seed, args.models)
            for m in args.models:
                au[(a, m)].append(res[m]["auroc"])
            aus = "  ".join(f"{m}={res[m]['auroc']:.3f}" for m in args.models)
            print(f"  seed {seed}  {a:<18} AUROC[{aus}]  inj={sel_inj}/{n_inj}  logdet={ld[a][-1]:.1f}")

    mean = lambda xs: float(np.mean(xs)) if xs else float("nan")
    print("\n" + "=" * 96)
    hdr = f"  {'arm':<18}" + "".join(f"{'AUROC('+m+')':>13}" for m in args.models) + f"{'hidden-pos sel%':>18}{'param logdet':>15}"
    print(hdr); print("  " + "-" * (len(hdr) - 2))
    for a in arms:
        row = f"  {a:<18}" + "".join(f"{mean(au[(a,m)]):>13.3f}" for m in args.models)
        print(row + f"{100*mean(inj[a]):>17.1f}%{mean(ld[a]):>15.1f}")
    print("=" * 96)
    print("  higher AUROC + higher param-logdet (diversity) + lower hidden-pos sel% = better.")


if __name__ == "__main__":
    main()
