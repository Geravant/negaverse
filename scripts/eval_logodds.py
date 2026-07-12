"""Step-1 evaluation: log-odds fusion + calibrated stack + LCB vs the mean.

Reuses the corrected-benchmark machinery (`_score_pool`, `_conf`, `_load_dataset`)
so the streams and pool are IDENTICAL to the shipped bench — only the fusion of
the per-stream sub-scores changes. Three questions, three tables:

  (A) CALIBRATION — is σ(L) a probability the mean is not? Score gold negatives
      (label N=1) and held-out real edges (label P=0) through the streams; report
      Brier + ECE + AUROC for mean vs log-odds pool vs cross-fitted stack.
  (B) LCB SAFETY — at a threshold that admits the same fraction of true negatives,
      does gating on the lower bound admit FEWER real edges (false negatives)?
  (C) SELECTION — inject hidden positives; what fraction does a confidence-ranked
      pick wrongly select, ranking by mean vs log-odds vs stack vs LCB?

    PYTHONPATH=. python3 scripts/eval_logodds.py [--dataset huri|dryad]
        [--max-positives 20000] [--seeds 0 1 2] [--inject-k 1000] [--l2 1.0]
"""
from __future__ import annotations

import argparse
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

import scripts.bench_corrected as bench
from scripts.bench_corrected import _conf, _load_dataset, _score_pool
from negaverse.ig.logodds_fusion import LogOddsStack, logit, logodds_pool, sigmoid
from negaverse.rule_engine import load_rules


def _feat(row, rule_ids):
    """Per-stream logit feature vector (abstention -> 0 = logit(prior 0.5))."""
    f = [logit(row["struct"]) if row["struct"] is not None else 0.0,
         logit(row["topo"]) if row["topo"] is not None else 0.0]
    for rid in rule_ids:
        wv = row["rules"].get(rid)
        f.append(logit(wv[1]) if wv is not None else 0.0)
    return f


def _pool_conf(row, rule_ids):
    """Log-odds pool over the SAME components the mean fuses (apples-to-apples
    coordinate change): {structured, topology, combined-rule-value}."""
    comps = {"structured": row["struct"], "topology": row["topo"]}
    fired = [(w, v) for rid, (w, v) in row["rules"].items() if rid in rule_ids]
    if fired:
        num = sum(w * v for w, v in fired); den = sum(w for w, _ in fired) or len(fired)
        comps["rules"] = num / den
    return sigmoid(logodds_pool(comps))


def _brier(y, p):
    return float(np.mean((p - y) ** 2))


def _ece(y, p, bins=10):
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, bins - 1)
    e = 0.0
    for b in range(bins):
        m = idx == b
        if m.any():
            e += m.mean() * abs(y[m].mean() - p[m].mean())
    return float(e)


def _score_labeled(tr_pos, pairs, seed, full_pos_set, test_excl, max_pool):
    """Score explicit `pairs` through the streams via the veto-bypassed inject path.
    Returns {frozenset(pair): row}."""
    pool = _score_pool(tr_pos, max_pool, seed, full_pos_set, exclude=test_excl, inject=pairs)
    return {frozenset((r["u"], r["v"])): r for r in pool if r.get("injected")}


def _calibration_and_lcb(rows_by_key, neg_keys, pos_keys, rule_ids, l2, alpha, out):
    keys = [k for k in list(neg_keys) + list(pos_keys) if k in rows_by_key]
    y = np.array([1 if k in neg_keys else 0 for k in keys])          # 1 = non-interaction
    if len(set(y)) < 2 or len(y) < 40:
        return None
    Z = np.array([_feat(rows_by_key[k], rule_ids) for k in keys])
    mean_p = np.array([_conf(rows_by_key[k], rule_ids) for k in keys])
    pool_p = np.array([_pool_conf(rows_by_key[k], rule_ids) for k in keys])
    # cross-fitted stack (out-of-fold proba + lcb)
    stack_p = np.zeros(len(y)); stack_lcb = np.zeros(len(y))
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    for tr, te in skf.split(Z, y):
        s = LogOddsStack(l2=l2, alpha=alpha).fit(Z[tr], y[tr])
        stack_p[te] = s.proba(Z[te]); stack_lcb[te] = s.lcb(Z[te])
    for name, p in [("mean", mean_p), ("logodds_pool", pool_p), ("stack", stack_p)]:
        out.setdefault(name, {"brier": [], "ece": [], "auroc": []})
        out[name]["brier"].append(_brier(y, p))
        out[name]["ece"].append(_ece(y, p))
        out[name]["auroc"].append(roc_auc_score(y, p))
    # (B) LCB safety: threshold admits q=80% of true negatives; count real edges admitted
    out.setdefault("_lcb", {"point": [], "lcb": []})
    for name, p in [("point", stack_p), ("lcb", stack_lcb)]:
        tau = np.quantile(p[y == 1], 0.20)                            # admit top-80% of negatives
        pos_admit = float(np.mean(p[y == 0] >= tau)) if (y == 0).any() else float("nan")
        out["_lcb"][name].append(pos_admit)
    return len(y)


def _selection(tr_pos, inject, seed, full_pos_set, test_excl, rule_ids, l2, cal_fit, max_pool, out):
    """Rank the clean pool by confidence-of-negative, take the top n; report the
    fraction of injected hidden positives wrongly selected (lower = safer)."""
    pool = _score_pool(tr_pos, max_pool, seed, full_pos_set, exclude=test_excl, inject=inject)
    inj_keys = {frozenset(p) for p in inject}
    clean = [r for r in pool if not r["vetoed"] or r.get("injected")]  # injected bypass veto
    n_inj = sum(1 for r in pool if r.get("injected"))
    n = len(tr_pos)
    Z = np.array([_feat(r, rule_ids) for r in clean])
    rankers = {
        "mean": np.array([_conf(r, rule_ids) for r in clean]),
        "logodds_pool": np.array([_pool_conf(r, rule_ids) for r in clean]),
        "stack": cal_fit.proba(Z),
        "stack_lcb": cal_fit.lcb(Z),
    }
    for name, s in rankers.items():
        top = np.argsort(s)[::-1][:n]                                 # most-confident-negative first
        sel = sum(1 for i in top if frozenset((clean[i]["u"], clean[i]["v"])) in inj_keys)
        out.setdefault(name, []).append(sel / max(n_inj, 1))
    return n_inj


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["huri", "dryad"], default="huri")
    ap.add_argument("--max-positives", type=int, default=20000)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--max-pool", type=int, default=40000)
    ap.add_argument("--inject-k", type=int, default=1000)
    ap.add_argument("--l2", type=float, default=0.2)
    ap.add_argument("--alpha", type=float, default=0.05)
    args = ap.parse_args()

    bench._RULES = [r for r in load_rules()
                    if r.modality == "ppi" and r.effect in ("safer_negative", "riskier_negative")]
    rule_ids = [r.id for r in bench._RULES]

    all_pos, gold, nodes, full_pos_set = _load_dataset(args.dataset)
    print("=" * 92)
    print(f"LOG-ODDS FUSION EVAL — {args.dataset.upper()}  rules={rule_ids}")
    print(f"full edges={len(all_pos)} gold={len(gold)} max_positives={args.max_positives} seeds={args.seeds} l2={args.l2}")
    print("=" * 92)

    cal, lcb_acc, sel = {}, None, {}
    n_lab = []
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
        gold_s = [tuple(p) for p in gold if frozenset(p) not in full_pos_set and set(p) <= tr_nodes]
        pos_lab = [p for p in test_pos if set(p) <= tr_nodes]
        # (A/B) calibration + lcb on labeled gold(N) / held-out edges(P)
        rows = _score_labeled(tr_pos, list(gold_s) + list(pos_lab), seed, full_pos_set,
                              test_excl, max_pool=2000)
        neg_keys = {frozenset(p) for p in gold_s}
        pos_keys = {frozenset(p) for p in pos_lab}
        nl = _calibration_and_lcb(rows, neg_keys, pos_keys, rule_ids, args.l2, args.alpha, cal)
        if nl:
            n_lab.append(nl)
        # fit the stack on ALL labeled (for the selection ranker)
        keys = [k for k in list(neg_keys) + list(pos_keys) if k in rows]
        y = np.array([1 if k in neg_keys else 0 for k in keys])
        Z = np.array([_feat(rows[k], rule_ids) for k in keys])
        cal_fit = LogOddsStack(l2=args.l2, alpha=args.alpha).fit(Z, y)
        # (C) hidden-positive injection selection
        held = [tuple(p) for p in full_pos_set
                if frozenset(p) not in {frozenset(e) for e in tr_pos}
                and frozenset(p) not in test_excl and set(p) <= tr_nodes and len(p) == 2]
        rng.shuffle(held)
        inject = held[:args.inject_k]
        _selection(tr_pos, inject, seed, full_pos_set, test_excl, rule_ids, args.l2,
                   cal_fit, args.max_pool, sel)

    m = lambda xs: float(np.mean(xs)) if xs else float("nan")
    print(f"\n(A) CALIBRATION — labeled pairs/seed≈{int(m(n_lab))}  (1=non-interaction, 0=real edge)")
    print(f"  {'method':<16}{'Brier':>10}{'ECE':>10}{'AUROC':>10}   (lower Brier/ECE better; AUROC=discrimination)")
    print("  " + "-" * 52)
    for name in ["mean", "logodds_pool", "stack"]:
        d = cal.get(name, {})
        print(f"  {name:<16}{m(d.get('brier',[])):>10.4f}{m(d.get('ece',[])):>10.4f}{m(d.get('auroc',[])):>10.4f}")
    print(f"\n(B) LCB SAFETY — real edges wrongly admitted as 'confident negatives'")
    print(f"    (threshold fixed to admit 80% of true negatives; lower = safer)")
    lb = cal.get("_lcb", {})
    print(f"  point estimate : {100*m(lb.get('point',[])):>5.1f}%   real edges admitted")
    print(f"  lower bound    : {100*m(lb.get('lcb',[])):>5.1f}%   real edges admitted")
    print(f"\n(C) HIDDEN-POSITIVE SELECTION — fraction of injected positives picked (lower = safer)")
    for name in ["mean", "logodds_pool", "stack", "stack_lcb"]:
        print(f"  {name:<16}{100*m(sel.get(name,[])):>6.1f}%")
    print("=" * 92)


if __name__ == "__main__":
    main()
