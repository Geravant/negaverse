"""Grammar arc, experiment 1: does a SPARSE PROOF FINGERPRINT beat the fused rule
score as a SAFETY model? (docs/IG-FEATURES.md — the grammar hypothesis.)

Evaluate-first, safety-only. Arms:
  1. fused        — the shipped mean of {structured, topology, combined-rules}.
  2. fingerprint+LR   — branch-separated proof fingerprint, logistic regression.
  3. fingerprint+LGBM — same fingerprint, gradient-boosted trees.
(Fisher-whitened fingerprint = arm 4, built only if these earn it.)

The label is VERIFIED: gold Negatome negatives (y=1 = defensible non-interaction)
vs held-out real edges (y=0). We report, per arm:
  * AUROC / AUPRC              — safety discrimination
  * Brier / ECE (cross-fit)   — calibration
  * injected-hidden-positive FALSE-NEGATIVE rate — of injected hidden positives,
    the fraction the safety model wrongly calls a confident negative (lower=safer)
  * protein-disjoint AUROC    — both endpoints held out (no hub memorisation)

The hypothesis survives only if the fingerprint improves calibration or the
hidden-positive FN rate over the fused score.

    PYTHONPATH=. python3 scripts/eval_grammar.py [--max-positives 8000] [--seeds 0 1 2]
"""
from __future__ import annotations

import argparse
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.model_selection import StratifiedKFold

import scripts.bench_corrected as bench
from scripts.bench_corrected import _conf, _load_dataset, _score_pool
from negaverse.grammar.features import fingerprint_matrix
from negaverse.rule_engine import load_rules


def _raw_feats(rows, rule_ids):
    """The SAME signals the fused score uses, UN-separated — the ablation that
    isolates 'grammar structure helps' from 'a classifier beats a mean'."""
    out = []
    for r in rows:
        fired = [(w, v) for rid, (w, v) in r.get("rules", {}).items() if rid in rule_ids]
        crv = (sum(w * v for w, v in fired) / (sum(w for w, _ in fired) or 1)) if fired else 0.5
        out.append([r.get("struct") or 0.0, r.get("topo") or 0.0,
                    float(r.get("risk", 0.0) or 0.0), float(r.get("hardness", 0.0) or 0.0), crv])
    return np.array(out, dtype=float)


def _brier(y, p):
    return float(np.mean((p - y) ** 2))


def _ece(y, p, bins=10):
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, bins - 1)
    return float(sum((idx == b).mean() * abs(y[idx == b].mean() - p[idx == b].mean())
                     for b in range(bins) if (idx == b).any()))


def _make_lgbm(seed):
    import lightgbm as lgb
    return lgb.LGBMClassifier(n_estimators=200, learning_rate=0.05, num_leaves=15,
                              min_child_samples=10, random_state=seed, n_jobs=-1, verbose=-1)


def _xfit(model_fn, X, y, seed):
    """Cross-fitted out-of-fold probabilities."""
    p = np.zeros(len(y))
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=seed).split(X, y):
        m = model_fn(); m.fit(X[tr], y[tr]); p[te] = m.predict_proba(X[te])[:, 1]
    return p


def _protein_disjoint_auroc(X, y, keys, seed, model_fn, folds=5):
    """AUROC under splits where BOTH endpoints of a test pair are held out of
    training — no protein appears in both, so no hub memorisation. `model_fn` must
    be the SAME learner as the arm being reported (else the number is mislabelled)."""
    prots = sorted({p for k in keys for p in k})
    rng = np.random.default_rng(seed)
    assign = {p: int(rng.integers(folds)) for p in prots}
    aucs = []
    for f in range(folds):
        te = np.array([all(assign[p] == f for p in k) for k in keys])
        tr = np.array([all(assign[p] != f for p in k) for k in keys])
        if te.sum() < 15 or tr.sum() < 30 or len(set(y[te])) < 2 or len(set(y[tr])) < 2:
            continue
        m = model_fn().fit(X[tr], y[tr])
        aucs.append(roc_auc_score(y[te], m.predict_proba(X[te])[:, 1]))
    return float(np.mean(aucs)) if aucs else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["huri", "dryad"], default="huri")
    ap.add_argument("--max-positives", type=int, default=8000)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--max-pool", type=int, default=12000)
    ap.add_argument("--inject-k", type=int, default=600)
    args = ap.parse_args()

    bench._RULES = [r for r in load_rules()
                    if r.modality == "ppi" and r.effect in ("safer_negative", "riskier_negative")]
    rules = bench._RULES
    rule_ids = [r.id for r in rules]

    all_pos, gold, nodes, full = _load_dataset(args.dataset)
    print("=" * 96)
    print(f"GRAMMAR EXP-1: sparse proof fingerprint vs fused score — {args.dataset.upper()}")
    print(f"gold={len(gold)} rules={rule_ids} max_positives={args.max_positives} seeds={args.seeds}")
    print("=" * 96)

    arms = ["fused", "raw+LGBM", "fingerprint+LR", "fingerprint+LGBM"]
    M = {a: {k: [] for k in ("auroc", "auprc", "brier", "ece", "fn_inj", "pdisj")} for a in arms}
    for seed in args.seeds:
        rng = np.random.default_rng(seed)
        pos = all_pos
        if args.max_positives and len(pos) > args.max_positives:
            pos = [pos[i] for i in rng.choice(len(pos), size=args.max_positives, replace=False)]
        pos = [pos[i] for i in rng.permutation(len(pos))]
        nt = int(len(pos) * 0.2)
        test_pos, tr_pos = pos[:nt], pos[nt:]
        test_excl = {frozenset(p) for p in test_pos}
        tr_nodes = {n for e in tr_pos for n in e}
        gold_s = [tuple(p) for p in gold if frozenset(p) not in full and set(p) <= tr_nodes]
        pos_lab = [p for p in test_pos if set(p) <= tr_nodes]
        held = [tuple(p) for p in full if frozenset(p) not in {frozenset(e) for e in tr_pos}
                and frozenset(p) not in test_excl and set(p) <= tr_nodes and len(p) == 2]
        rng.shuffle(held)
        inject = held[:args.inject_k]

        pool = _score_pool(tr_pos, 2000, seed, full, exclude=test_excl,
                           inject=list(gold_s) + list(pos_lab) + list(inject))
        by = {frozenset((r["u"], r["v"])): r for r in pool if r.get("injected")}
        neg_keys = {frozenset(p) for p in gold_s}
        pos_keys = {frozenset(p) for p in pos_lab}
        inj_keys = {frozenset(p) for p in inject}
        lab_keys = [k for k in list(neg_keys) + list(pos_keys) if k in by]
        y = np.array([1 if k in neg_keys else 0 for k in lab_keys])
        if len(set(y)) < 2:
            continue
        rows = [by[k] for k in lab_keys]
        Xfp, _ = fingerprint_matrix(rows, rules)
        Xraw = _raw_feats(rows, rule_ids)
        fused = np.array([_conf(by[k], rule_ids) for k in lab_keys])

        preds = {"fused": fused,
                 "raw+LGBM": _xfit(lambda: _make_lgbm(seed), Xraw, y, seed),
                 "fingerprint+LR": _xfit(lambda: LogisticRegression(C=1.0, max_iter=2000), Xfp, y, seed),
                 "fingerprint+LGBM": _xfit(lambda: _make_lgbm(seed), Xfp, y, seed)}
        # injected hidden positives → each arm's safety model calls them how safe?
        inj_rows = [by[k] for k in inj_keys if k in by]
        Xinj, _ = fingerprint_matrix(inj_rows, rules)
        Xinj_raw = _raw_feats(inj_rows, rule_ids)
        fused_inj = np.array([_conf(by[k], rule_ids) for k in inj_keys if k in by])
        lr = LogisticRegression(C=1.0, max_iter=2000).fit(Xfp, y)
        lg = _make_lgbm(seed).fit(Xfp, y)
        lg_raw = _make_lgbm(seed).fit(Xraw, y)
        _pp = lambda m, X: m.predict_proba(X)[:, 1] if len(X) else np.array([])
        inj_pred = {"fused": fused_inj, "raw+LGBM": _pp(lg_raw, Xinj_raw),
                    "fingerprint+LR": _pp(lr, Xinj), "fingerprint+LGBM": _pp(lg, Xinj)}

        for a in arms:
            p = preds[a]
            M[a]["auroc"].append(roc_auc_score(y, p))
            M[a]["auprc"].append(average_precision_score(y, p))
            M[a]["brier"].append(_brier(y, p))
            M[a]["ece"].append(_ece(y, p))
            # FN rate: threshold = median p on gold negatives (admit 50% of true negs);
            # fraction of injected hidden positives scored >= that (wrongly "safe")
            tau = float(np.quantile(p[y == 1], 0.5))
            pi = inj_pred[a]
            M[a]["fn_inj"].append(float(np.mean(pi >= tau)) if len(pi) else float("nan"))
            Xpd = Xfp if a.startswith("fingerprint") else (Xraw if a == "raw+LGBM" else None)
            pd_model = ((lambda: LogisticRegression(C=1.0, max_iter=2000)) if a.endswith("+LR")
                        else (lambda: _make_lgbm(seed)))
            M[a]["pdisj"].append(_protein_disjoint_auroc(Xpd, y, lab_keys, seed, pd_model)
                                 if Xpd is not None else float("nan"))
        print(f"  seed {seed}: labeled={len(y)} (neg={int(y.sum())}/pos={int((1-y).sum())})  "
              f"injected={len(inj_rows)}")

    m = lambda a, k: float(np.nanmean(M[a][k])) if M[a][k] else float("nan")
    print("\n" + "=" * 96)
    hdr = f"  {'arm':<18}{'AUROC':>9}{'AUPRC':>9}{'Brier':>9}{'ECE':>9}{'inj-FN%':>10}{'prot-disj AUROC':>17}"
    print(hdr); print("  " + "-" * (len(hdr) - 2))
    for a in arms:
        print(f"  {a:<18}{m(a,'auroc'):>9.3f}{m(a,'auprc'):>9.3f}{m(a,'brier'):>9.3f}"
              f"{m(a,'ece'):>9.3f}{100*m(a,'fn_inj'):>9.1f}%{m(a,'pdisj'):>17.3f}")
    print("=" * 96)
    print("  Hypothesis survives only if fingerprint LOWERS Brier/ECE or inj-FN% vs fused.")


if __name__ == "__main__":
    main()
