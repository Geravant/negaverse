"""Calibrate/evaluate the colocalization_mismatch rule against the same
gold-standard PPI benchmarks used for hydrophobicity_interface and
evolutionary_coupling_absence: DRYAD and UPNA-PPI. Reuses the generic
dataset-loading and threshold/AUROC helpers from
scripts/calibrate_hydrophobicity_threshold.py.

The rule's `when` is `disjoint(a.compartments, b.compartments)` — a binary
predicate, not a threshold. To evaluate how well GO cellular-component
overlap separates interacting from non-interacting pairs, this script uses a
continuous stand-in feature, Jaccard(a.compartments, b.compartments), and:

  - computes AUROC treating Jaccard as the score (higher overlap -> more
    likely interacting, matching the rule's premise) — the usual
    optimistic + protein-disjoint honesty split;
  - separately reports the *actual currently-shipped rule's* confusion
    numbers: among true positives (real interactions), what fraction have
    jaccard==0 (disjoint) and would be wrongly flagged as a safer negative
    by this rule as it exists today; among true negatives, what fraction
    are correctly flagged.

This does NOT change the rule's `when` (it stays a clean disjoint-set
predicate — AUTHORING.md's whitelisted grammar has no direct "jaccard
below threshold" comparison needed here, disjoint is already the right
shape); this script exists purely to answer "does the GO CC coverage fix
(0% -> 93% on DRYAD) actually translate into real separation," not to
re-derive a new threshold.

    PYTHONPATH=. python scripts/calibrate_colocalization_threshold.py
    PYTHONPATH=. python scripts/calibrate_colocalization_threshold.py --n-per-class 50   # quick check

Writes out/colocalization_calibration.png (ROC curve + Jaccard histograms by
class, one row per dataset), matching house style.
"""
from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve

from scripts.calibrate_hydrophobicity_threshold import (
    Pair, _load_dryad_pairs, _load_upna_pairs, _map_upna_to_uniprot,
    _subsample, _youden_threshold, _precision_recall_at,
)
from negaverse.io.localization import load_localization_tsv

SEED = 0
N_PER_CLASS = 300   # pure local lookup, no network/compute cost — go straight to the full size

OUT_PNG = Path("out/colocalization_calibration.png")


def _jaccard(sa: set[str], sb: set[str]) -> float:
    union = sa | sb
    return len(sa & sb) / len(union) if union else 0.0


def evaluate(name: str, pos: list[Pair], neg: list[Pair],
             compartments: dict[str, set[str]]) -> dict:
    pairs = pos + neg
    print(f"[{name}] {len(pos)} positive / {len(neg)} negative pairs")

    y_all = np.r_[np.ones(len(pos)), np.zeros(len(neg))]
    scores_all_raw = [
        _jaccard(compartments[a], compartments[b]) if a in compartments and b in compartments else None
        for a, b in pairs
    ]
    keep = [i for i, s in enumerate(scores_all_raw) if s is not None]
    y = y_all[keep]
    scores = np.array([scores_all_raw[i] for i in keep])
    n_covered = len(scores)
    print(f"[{name}] {n_covered}/{len(pairs)} pairs had compartments for both proteins")

    # actual shipped rule's confusion numbers: fires (predicts negative) when jaccard == 0
    disjoint_mask = scores == 0.0
    y_pos_covered = y == 1
    y_neg_covered = y == 0
    tp_rule = int(np.sum(disjoint_mask & y_neg_covered))   # correctly flagged non-interacting
    fp_rule = int(np.sum(disjoint_mask & y_pos_covered))   # wrongly flagged real interactions
    fired_on_pos_rate = fp_rule / np.sum(y_pos_covered) if np.sum(y_pos_covered) else float("nan")
    fired_on_neg_rate = tp_rule / np.sum(y_neg_covered) if np.sum(y_neg_covered) else float("nan")
    print(f"[{name}] shipped rule (disjoint==True fires): fires on "
          f"{fired_on_neg_rate:.1%} of true negatives (correct), "
          f"{fired_on_pos_rate:.1%} of true positives (wrong)")

    if len(set(y)) < 2 or n_covered == 0:
        print(f"[{name}] not enough covered pairs in both classes; skipping AUROC")
        return dict(name=name, y=y, scores=scores, auroc_raw=float("nan"), flip=None,
                    auroc_optimistic=float("nan"), threshold_optimistic=float("nan"),
                    precision_optimistic=float("nan"), recall_optimistic=float("nan"),
                    fired_on_pos_rate=fired_on_pos_rate, fired_on_neg_rate=fired_on_neg_rate,
                    disjoint={})

    auroc_raw = roc_auc_score(y, scores)
    flip = auroc_raw < 0.5
    print(f"[{name}] raw AUROC (unflipped, high jaccard = predicted interacting) = {auroc_raw:.3f} -> "
          f"{'reversed' if flip else 'as expected'} direction")

    t_opt, tpr_opt, fpr_opt = _youden_threshold(y, scores, flip=flip)
    prec_opt, rec_opt = _precision_recall_at(y, scores, t_opt, flip=flip)
    auroc_optimistic = auroc_raw if not flip else 1.0 - auroc_raw

    proteins = sorted({p for pair in pos + neg for p in pair})
    rng = np.random.default_rng(SEED)
    half_a = set(rng.choice(proteins, size=len(proteins) // 2, replace=False))
    half_b = set(proteins) - half_a

    def _both_in(ps: list[Pair], half: set[str]) -> list[Pair]:
        return [(a, b) for a, b in ps if a in half and b in half]

    def _scores_for(ps: list[Pair]) -> np.ndarray:
        out = [_jaccard(compartments[a], compartments[b]) if a in compartments and b in compartments else None
               for a, b in ps]
        return np.array([s for s in out if s is not None])

    fit_pos, fit_neg = _both_in(pos, half_a), _both_in(neg, half_a)
    rep_pos, rep_neg = _both_in(pos, half_b), _both_in(neg, half_b)
    fit_scores = _scores_for(fit_pos), _scores_for(fit_neg)
    rep_scores = _scores_for(rep_pos), _scores_for(rep_neg)

    disjoint: dict = {}
    if len(fit_scores[0]) and len(fit_scores[1]) and len(rep_scores[0]) and len(rep_scores[1]):
        y_fit = np.r_[np.ones(len(fit_scores[0])), np.zeros(len(fit_scores[1]))]
        s_fit = np.r_[fit_scores[0], fit_scores[1]]
        t_disjoint, _, _ = _youden_threshold(y_fit, s_fit, flip=flip)

        y_rep = np.r_[np.ones(len(rep_scores[0])), np.zeros(len(rep_scores[1]))]
        s_rep = np.r_[rep_scores[0], rep_scores[1]]
        auroc_rep_raw = roc_auc_score(y_rep, s_rep) if len(set(y_rep)) > 1 else float("nan")
        auroc_disjoint = auroc_rep_raw if not flip else 1.0 - auroc_rep_raw
        prec_disjoint, rec_disjoint = _precision_recall_at(y_rep, s_rep, t_disjoint, flip=flip)
        disjoint = dict(threshold=t_disjoint, auroc=auroc_disjoint,
                         precision=prec_disjoint, recall=rec_disjoint,
                         n_fit=len(s_fit), n_report=len(s_rep))
    else:
        print(f"[{name}] protein-disjoint split left too few both-in-half pairs; skipping that number")

    return dict(name=name, y=y, scores=scores, auroc_raw=auroc_raw, flip=flip,
                auroc_optimistic=auroc_optimistic, threshold_optimistic=t_opt,
                precision_optimistic=prec_opt, recall_optimistic=rec_opt,
                fired_on_pos_rate=fired_on_pos_rate, fired_on_neg_rate=fired_on_neg_rate,
                disjoint=disjoint)


def _plot(results: list[dict]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    results = [r for r in results if len(r["scores"])]
    if not results:
        print("nothing to plot")
        return
    fig, axes = plt.subplots(len(results), 2, figsize=(10, 4.5 * len(results)))
    if len(results) == 1:
        axes = axes.reshape(1, 2)
    for row, r in enumerate(results):
        ax = axes[row, 0]
        fpr, tpr, _ = roc_curve(r["y"], r["scores"])
        ax.plot(fpr, tpr, label=f"AUROC(raw)={r['auroc_raw']:.3f}")
        ax.plot([0, 1], [0, 1], "k--", lw=0.7)
        ax.set_title(f"{r['name']}: ROC (raw direction)")
        ax.set_xlabel("FPR")
        ax.set_ylabel("TPR")
        ax.legend()

        ax = axes[row, 1]
        pos_s = r["scores"][r["y"] == 1]
        neg_s = r["scores"][r["y"] == 0]
        ax.hist(neg_s, bins=20, alpha=0.6, label="negative (non-interacting)", density=True)
        ax.hist(pos_s, bins=20, alpha=0.6, label="positive (interacting)", density=True)
        ax.set_title(f"{r['name']}: Jaccard(compartments) by class")
        ax.set_xlabel("jaccard(a.compartments, b.compartments)")
        ax.legend()
    fig.tight_layout()
    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PNG, dpi=130)
    print(f"wrote {OUT_PNG}")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="calibrate_colocalization_threshold")
    ap.add_argument("--n-per-class", type=int, default=N_PER_CLASS)
    ap.add_argument("--dataset", choices=["both", "dryad", "upna"], default="both")
    args = ap.parse_args(argv)
    rng = random.Random(SEED)
    compartments = load_localization_tsv()
    print(f"loaded compartments for {len(compartments)} nodes")

    results = []
    if args.dataset in ("both", "dryad"):
        print("=== DRYAD ===")
        d_pos, d_neg = _load_dryad_pairs()
        d_pos_s = _subsample(d_pos, args.n_per_class, rng)
        d_neg_s = _subsample(d_neg, args.n_per_class, rng)
        results.append(evaluate("DRYAD", d_pos_s, d_neg_s, compartments))

    if args.dataset in ("both", "upna"):
        print("\n=== UPNA-PPI ===")
        u_pos, u_neg = _load_upna_pairs()
        u_pos_s = _subsample(u_pos, args.n_per_class, rng)
        u_neg_s = _subsample(u_neg, args.n_per_class, rng)
        u_pos_u, u_neg_u = _map_upna_to_uniprot(u_pos_s, u_neg_s)
        results.append(evaluate("UPNA-PPI", u_pos_u, u_neg_u, compartments))

    print("\n=== Summary ===")
    for r in results:
        print(f"{r['name']}: AUROC(raw)={r['auroc_raw']:.3f}, flip={r['flip']}, "
              f"AUROC(as-deployed)={r['auroc_optimistic']:.3f}, "
              f"threshold={r['threshold_optimistic']:.4f}, "
              f"precision={r['precision_optimistic']:.3f}, recall={r['recall_optimistic']:.3f}")
        print(f"  shipped rule fires on {r['fired_on_neg_rate']:.1%} of true negatives (correct), "
              f"{r['fired_on_pos_rate']:.1%} of true positives (wrong)")
        if r["disjoint"]:
            d = r["disjoint"]
            print(f"  protein-disjoint: AUROC={d['auroc']:.3f}, threshold={d['threshold']:.4f}, "
                  f"precision={d['precision']:.3f}, recall={d['recall']:.3f} "
                  f"(fit n={d['n_fit']}, report n={d['n_report']})")

    _plot(results)


if __name__ == "__main__":
    main()
