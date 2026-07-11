"""Calibrate the evolutionary_coupling_absence rule's threshold against the
same two gold-standard PPI benchmarks used for hydrophobicity_interface: DRYAD
and UPNA-PPI. Reuses the generic dataset-loading and threshold/AUROC helpers
from scripts/calibrate_hydrophobicity_threshold.py directly (pair loading,
UPNA->UniProt mapping, subsampling, Youden's-J/precision-recall with
flip-awareness) rather than re-deriving them — only the feature being
calibrated differs (ERC pair scores instead of max(h_a, h_b)).

Unlike hydrophobicity, the score here is already pairwise (one Evolutionary
Rate Covariation value per pair from scripts/compute_evolutionary_coupling.py
— no max(h_a, h_b) reduction needed).

**Direction is not assumed.** The rule's original premise ("low coupling ->
safer negative") matches the standard coevolution-literature expectation
(interacting proteins tend to show correlated evolutionary rates, i.e. high
ERC -> more likely interacting), unlike hydrophobicity where the naive premise
turned out backwards — but this script still checks empirically rather than
hard-coding that assumption: it computes AUROC in the raw (unflipped)
direction first, and only applies `flip=True` (see
calibrate_hydrophobicity_threshold._youden_threshold's docstring) if that
raw AUROC comes out below 0.5, exactly the same discipline that caught
hydrophobicity's direction being backwards.

**Runtime note**: the ortholog-fetch + MAFFT + phangorn (fixed-topology
branch-length fitting) + RERconverge pipeline is heavier per-protein than DSSP
was, so N_PER_CLASS defaults much lower than hydrophobicity's — scale up once
real per-protein runtime is measured.

    PYTHONPATH=. python scripts/calibrate_evolutionary_coupling_threshold.py
    PYTHONPATH=. python scripts/calibrate_evolutionary_coupling_threshold.py --n-per-class 10   # smoke test

Writes out/evolutionary_coupling_calibration.png (ROC curve + score histograms
by class, one row per dataset), matching house style.
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
from scripts.compute_evolutionary_coupling import compute as compute_erc

SEED = 0
N_PER_CLASS = 50   # heavier per-protein pipeline than hydrophobicity's DSSP path

OUT_PNG = Path("out/evolutionary_coupling_calibration.png")


def evaluate(name: str, pos: list[Pair], neg: list[Pair],
             min_trees_all: int = 20, min_valid: int = 20) -> dict:
    pairs = pos + neg
    print(f"[{name}] {len(pos)} positive / {len(neg)} negative pairs")
    # {frozenset({a, b}): score}, gated pairs simply absent
    erc = compute_erc(pairs, min_trees_all=min_trees_all, min_valid=min_valid)

    y_all = np.r_[np.ones(len(pos)), np.zeros(len(neg))]
    scores_all_raw = [erc.get(frozenset(p)) for p in pairs]
    keep = [i for i, s in enumerate(scores_all_raw) if s is not None]
    y = y_all[keep]
    scores = np.array([scores_all_raw[i] for i in keep])
    print(f"[{name}] {len(scores)}/{len(pairs)} pairs scored (passed the quality gate)")

    if len(set(y)) < 2 or len(scores) == 0:
        print(f"[{name}] not enough scored pairs in both classes; skipping")
        return dict(name=name, y=y, scores=scores, auroc_raw=float("nan"), flip=None,
                    auroc_optimistic=float("nan"), threshold_optimistic=float("nan"),
                    precision_optimistic=float("nan"), recall_optimistic=float("nan"),
                    disjoint={})

    # empirical direction check — do not assume; see module docstring
    auroc_raw = roc_auc_score(y, scores)
    flip = auroc_raw < 0.5
    print(f"[{name}] raw AUROC (unflipped) = {auroc_raw:.3f} -> "
          f"{'reversed' if flip else 'as expected'} direction "
          f"({'high ERC = safer negative' if flip else 'low ERC = safer negative'})")

    t_opt, tpr_opt, fpr_opt = _youden_threshold(y, scores, flip=flip)
    prec_opt, rec_opt = _precision_recall_at(y, scores, t_opt, flip=flip)
    auroc_optimistic = auroc_raw if not flip else 1.0 - auroc_raw

    # protein-disjoint split, same honesty convention as hydrophobicity
    proteins = sorted({p for pair in pos + neg for p in pair})
    rng = np.random.default_rng(SEED)
    half_a = set(rng.choice(proteins, size=len(proteins) // 2, replace=False))
    half_b = set(proteins) - half_a

    def _both_in(ps: list[Pair], half: set[str]) -> list[Pair]:
        return [(a, b) for a, b in ps if a in half and b in half]

    def _scores_for(ps: list[Pair]) -> np.ndarray:
        out = [erc.get(frozenset(p)) for p in ps]
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
        ax.hist(neg_s, bins=30, alpha=0.6, label="negative (non-interacting)", density=True)
        ax.hist(pos_s, bins=30, alpha=0.6, label="positive (interacting)", density=True)
        ax.axvline(r["threshold_optimistic"], color="k", ls="--", label="Youden threshold")
        ax.set_title(f"{r['name']}: ERC score by class")
        ax.set_xlabel("evolutionary_coupling_score_with_b")
        ax.legend()
    fig.tight_layout()
    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PNG, dpi=130)
    print(f"wrote {OUT_PNG}")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="calibrate_evolutionary_coupling_threshold")
    ap.add_argument("--n-per-class", type=int, default=N_PER_CLASS)
    ap.add_argument("--min-trees-all", type=int, default=20,
                     help="RERconverge's own master-tree-estimation minimum; lower for small runs")
    ap.add_argument("--min-valid", type=int, default=20,
                     help="RERconverge's own min.valid (getAllResiduals); lower for small runs")
    args = ap.parse_args(argv)
    rng = random.Random(SEED)

    print("=== DRYAD ===")
    d_pos, d_neg = _load_dryad_pairs()
    d_pos_s = _subsample(d_pos, args.n_per_class, rng)
    d_neg_s = _subsample(d_neg, args.n_per_class, rng)
    dryad_result = evaluate("DRYAD", d_pos_s, d_neg_s,
                             min_trees_all=args.min_trees_all, min_valid=args.min_valid)

    print("\n=== UPNA-PPI ===")
    u_pos, u_neg = _load_upna_pairs()
    u_pos_s = _subsample(u_pos, args.n_per_class, rng)
    u_neg_s = _subsample(u_neg, args.n_per_class, rng)
    u_pos_u, u_neg_u = _map_upna_to_uniprot(u_pos_s, u_neg_s)
    upna_result = evaluate("UPNA-PPI", u_pos_u, u_neg_u,
                            min_trees_all=args.min_trees_all, min_valid=args.min_valid)

    results = [dryad_result, upna_result]
    print("\n=== Summary ===")
    for r in results:
        print(f"{r['name']}: AUROC(raw)={r['auroc_raw']:.3f}, flip={r['flip']}, "
              f"AUROC(as-deployed)={r['auroc_optimistic']:.3f}, "
              f"threshold={r['threshold_optimistic']:.4f}, "
              f"precision={r['precision_optimistic']:.3f}, recall={r['recall_optimistic']:.3f}")
        if r["disjoint"]:
            d = r["disjoint"]
            print(f"  protein-disjoint: AUROC={d['auroc']:.3f}, threshold={d['threshold']:.4f}, "
                  f"precision={d['precision']:.3f}, recall={d['recall']:.3f} "
                  f"(fit n={d['n_fit']}, report n={d['n_report']})")

    _plot(results)


if __name__ == "__main__":
    main()
