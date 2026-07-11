"""Calibrate the hydrophobicity_interface rule's threshold against two
gold-standard PPI benchmarks already vendored in this repo: DRYAD
(UniProt-keyed, both positive and negative pairs in one file) and UPNA-PPI
(gene-symbol keyed; positives from PPI_part_1.csv, hard negatives from its
own contrastive-L3 TPPNI_1.csv — file 1 of each only, ~1M rows apiece,
plenty for a representative random subsample without loading all 5/4 parts).

For each dataset: subsample N_PER_CLASS positive and N_PER_CLASS negative
pairs, compute each protein's surface_hydrophobicity (two-tier,
scripts/compute_surface_hydrophobicity.py, Kyte-Doolittle scale), reduce to
the pair-level feature the rule's `when` uses — max(h_a, h_b) — then report:

  - AUROC via roc_auc_score(y=is_positive, score=max_hydrophobicity).
  - An "optimistic" number: fit and reported on the same sample.
  - A "protein-disjoint" number (same honesty convention as
    dryad_structure_separation.py / upna_topology_separation.py): proteins
    are split into two non-overlapping halves; only pairs with BOTH
    endpoints in half A are used to pick the threshold, only pairs with BOTH
    endpoints in half B are used to report AUROC/precision/recall.
  - A recommended threshold T: the fitting-half ROC point maximizing
    Youden's J (TPR - FPR).

CONFIRMED FINDING (n=300/class, both datasets, see rules/ppi.yaml's
hydrophobicity_interface rationale for the full writeup): AUROC came out
*below* 0.5 in all four measurements (optimistic + protein-disjoint, both
datasets) — real interactions have LOWER exposed hydrophobicity than real
non-interactions, the opposite of this rule's original premise. Also tested:
Wimley-White interfacial scale and Spatial Aggregation Propensity (SAP) —
neither beat plain Kyte-Doolittle once you account for SAP's much lower
pair-coverage (structure-only, no sequence fallback), so this script stays
Kyte-Doolittle-only. Because of the confirmed direction, `_youden_threshold`/
`_precision_recall_at` operate in the FLIPPED sense by default (high score
-> predicted negative) — see their docstrings.

    PYTHONPATH=. python scripts/calibrate_hydrophobicity_threshold.py
    PYTHONPATH=. python scripts/calibrate_hydrophobicity_threshold.py --n-per-class 20   # quick smoke test

Writes out/hydrophobicity_calibration.png (ROC curve + score histograms by
class, one row per dataset), matching house style.
"""
from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve

from scripts.compute_surface_hydrophobicity import compute as compute_hydrophobicity
from scripts.build_gene_symbol_uniprot_map import _fetch as _fetch_gene_map

SEED = 0
N_PER_CLASS = 400

DRYAD_PATH = Path("local-docs/dryad-ppi/benchmarks/positives_and_negatives.tsv")
UPNA_DIR = Path("local-docs/upna-ppi")
OUT_PNG = Path("out/hydrophobicity_calibration.png")

Pair = tuple[str, str]


def _load_dryad_pairs() -> tuple[list[Pair], list[Pair]]:
    df = pd.read_csv(DRYAD_PATH, sep="\t")
    split = df["Protein pairs"].str.split("_", n=1, expand=True)
    df = df.assign(a=split[0], b=split[1])
    pos_df, neg_df = df[df["Category"] == "positive"], df[df["Category"] == "negative"]
    pos = list(zip(pos_df["a"], pos_df["b"]))
    neg = list(zip(neg_df["a"], neg_df["b"]))
    return pos, neg


def _load_upna_pairs() -> tuple[list[Pair], list[Pair]]:
    pos_df = pd.read_csv(UPNA_DIR / "PPI_part_1.csv", usecols=["SymbolA", "SymbolB"])
    neg_df = pd.read_csv(UPNA_DIR / "TPPNI_1.csv", usecols=["SymbolA", "SymbolB"])
    pos = list(zip(pos_df["SymbolA"].astype(str), pos_df["SymbolB"].astype(str)))
    neg = list(zip(neg_df["SymbolA"].astype(str), neg_df["SymbolB"].astype(str)))
    return pos, neg


def _map_upna_to_uniprot(pos: list[Pair], neg: list[Pair]) -> tuple[list[Pair], list[Pair]]:
    symbols = sorted({s for pair in pos + neg for s in pair})
    print(f"  mapping {len(symbols)} UPNA gene symbols to UniProt...")
    mapping: dict[str, str] = {}
    for i in range(0, len(symbols), 50):
        chunk = symbols[i:i + 50]
        for attempt in range(3):
            try:
                mapping.update(_fetch_gene_map(chunk))
                break
            except Exception as e:
                print(f"    chunk {i // 50}: retry {attempt + 1} ({e})")

    def _convert(pairs: list[Pair]) -> list[Pair]:
        return [(mapping[a], mapping[b]) for a, b in pairs if a in mapping and b in mapping]

    pos_u, neg_u = _convert(pos), _convert(neg)
    print(f"  mapped {len(pos_u)}/{len(pos)} positive, {len(neg_u)}/{len(neg)} negative pairs to UniProt")
    return pos_u, neg_u


def _subsample(pairs: list[Pair], n: int, rng: random.Random) -> list[Pair]:
    return pairs if len(pairs) <= n else rng.sample(pairs, n)


def _pair_scores(pairs: list[Pair], hydro: dict[str, float]) -> np.ndarray:
    out = []
    for a, b in pairs:
        ha, hb = hydro.get(a), hydro.get(b)
        if ha is not None and hb is not None:
            out.append(max(ha, hb))
    return np.array(out)


def _youden_threshold(y: np.ndarray, scores: np.ndarray, flip: bool = True) -> tuple[float, float, float]:
    """Returns (threshold, tpr, fpr) at the point maximizing TPR - FPR for
    predicting y=1 (positive/interacting) from `scores`. Confirmed direction
    for hydrophobicity_interface is "high score -> safer negative" (opposite
    of the rule's original premise), so `flip=True` (default) computes the
    ROC/threshold for that direction: low scores predict y=1, high scores
    predict y=0. Implemented by running roc_curve on -scores (sklearn assumes
    high score -> y=1) and negating the resulting threshold back to real
    score units."""
    sign = -1 if flip else 1
    fpr, tpr, thresh = roc_curve(y, sign * scores)
    j = tpr - fpr
    i = int(np.argmax(j))
    return float(sign * thresh[i]), float(tpr[i]), float(fpr[i])


def _precision_recall_at(y: np.ndarray, scores: np.ndarray, threshold: float,
                          flip: bool = True) -> tuple[float, float]:
    """Precision/recall of the *negative* call. flip=True (confirmed
    direction, matches _youden_threshold's default): predicted negative when
    score > threshold. flip=False: score < threshold (the rule's original,
    since-disproven premise)."""
    pred_negative = scores > threshold if flip else scores < threshold
    true_negative = y == 0
    tp = int(np.sum(pred_negative & true_negative))          # correctly called non-interacting
    fp = int(np.sum(pred_negative & ~true_negative))          # wrongly called non-interacting
    fn = int(np.sum(~pred_negative & true_negative))          # missed non-interacting pairs
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    return precision, recall


def evaluate(name: str, pos: list[Pair], neg: list[Pair]) -> dict:
    proteins = sorted({p for pair in pos + neg for p in pair})
    print(f"[{name}] {len(pos)} positive / {len(neg)} negative pairs, {len(proteins)} unique proteins")
    hydro, tiers = compute_hydrophobicity(proteins)
    n_structure = sum(1 for t in tiers.values() if t == "structure")
    print(f"[{name}] {n_structure}/{len(tiers)} proteins scored via structure (Tier 1)")

    y_all = np.r_[np.ones(len(pos)), np.zeros(len(neg))]
    scores_all_raw = [max(hydro[a], hydro[b]) if a in hydro and b in hydro else None
                       for a, b in pos + neg]
    keep = [i for i, s in enumerate(scores_all_raw) if s is not None]
    y = y_all[keep]
    scores = np.array([scores_all_raw[i] for i in keep])
    print(f"[{name}] {len(scores)}/{len(pos) + len(neg)} pairs scored (both proteins had hydrophobicity)")

    auroc_optimistic = roc_auc_score(y, scores) if len(set(y)) > 1 else float("nan")
    t_opt, tpr_opt, fpr_opt = _youden_threshold(y, scores)
    prec_opt, rec_opt = _precision_recall_at(y, scores, t_opt)

    # protein-disjoint split
    rng = np.random.default_rng(SEED)
    half_a = set(rng.choice(proteins, size=len(proteins) // 2, replace=False))
    half_b = set(proteins) - half_a

    def _both_in(pairs: list[Pair], half: set[str]) -> list[Pair]:
        return [(a, b) for a, b in pairs if a in half and b in half]

    fit_pos, fit_neg = _both_in(pos, half_a), _both_in(neg, half_a)
    rep_pos, rep_neg = _both_in(pos, half_b), _both_in(neg, half_b)
    fit_scores = _pair_scores(fit_pos, hydro), _pair_scores(fit_neg, hydro)
    rep_scores = _pair_scores(rep_pos, hydro), _pair_scores(rep_neg, hydro)

    disjoint = {}
    if len(fit_scores[0]) and len(fit_scores[1]) and len(rep_scores[0]) and len(rep_scores[1]):
        y_fit = np.r_[np.ones(len(fit_scores[0])), np.zeros(len(fit_scores[1]))]
        s_fit = np.r_[fit_scores[0], fit_scores[1]]
        t_disjoint, _, _ = _youden_threshold(y_fit, s_fit)

        y_rep = np.r_[np.ones(len(rep_scores[0])), np.zeros(len(rep_scores[1]))]
        s_rep = np.r_[rep_scores[0], rep_scores[1]]
        auroc_disjoint = roc_auc_score(y_rep, s_rep) if len(set(y_rep)) > 1 else float("nan")
        prec_disjoint, rec_disjoint = _precision_recall_at(y_rep, s_rep, t_disjoint)
        disjoint = dict(threshold=t_disjoint, auroc=auroc_disjoint,
                         precision=prec_disjoint, recall=rec_disjoint,
                         n_fit=len(s_fit), n_report=len(s_rep))
    else:
        print(f"[{name}] protein-disjoint split left too few both-in-half pairs; skipping that number")

    return dict(name=name, y=y, scores=scores,
                auroc_optimistic=auroc_optimistic, threshold_optimistic=t_opt,
                precision_optimistic=prec_opt, recall_optimistic=rec_opt,
                disjoint=disjoint)


def _plot(results: list[dict]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(len(results), 2, figsize=(10, 4.5 * len(results)))
    if len(results) == 1:
        axes = axes.reshape(1, 2)
    for row, r in enumerate(results):
        ax = axes[row, 0]
        fpr, tpr, _ = roc_curve(r["y"], r["scores"])
        ax.plot(fpr, tpr, label=f"AUROC={r['auroc_optimistic']:.3f}")
        ax.plot([0, 1], [0, 1], "k--", lw=0.7)
        ax.set_title(f"{r['name']}: ROC (optimistic)")
        ax.set_xlabel("FPR")
        ax.set_ylabel("TPR")
        ax.legend()

        ax = axes[row, 1]
        pos_s = r["scores"][r["y"] == 1]
        neg_s = r["scores"][r["y"] == 0]
        ax.hist(neg_s, bins=30, alpha=0.6, label="negative (non-interacting)", density=True)
        ax.hist(pos_s, bins=30, alpha=0.6, label="positive (interacting)", density=True)
        ax.axvline(r["threshold_optimistic"], color="k", ls="--", label="Youden threshold")
        ax.set_title(f"{r['name']}: max(h_a, h_b) by class")
        ax.set_xlabel("surface_hydrophobicity (max of pair)")
        ax.legend()
    fig.tight_layout()
    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PNG, dpi=130)
    print(f"wrote {OUT_PNG}")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="calibrate_hydrophobicity_threshold")
    ap.add_argument("--n-per-class", type=int, default=N_PER_CLASS)
    args = ap.parse_args(argv)
    rng = random.Random(SEED)

    print("=== DRYAD ===")
    d_pos, d_neg = _load_dryad_pairs()
    d_pos_s = _subsample(d_pos, args.n_per_class, rng)
    d_neg_s = _subsample(d_neg, args.n_per_class, rng)
    dryad_result = evaluate("DRYAD", d_pos_s, d_neg_s)

    print("\n=== UPNA-PPI ===")
    u_pos, u_neg = _load_upna_pairs()
    u_pos_s = _subsample(u_pos, args.n_per_class, rng)
    u_neg_s = _subsample(u_neg, args.n_per_class, rng)
    u_pos_u, u_neg_u = _map_upna_to_uniprot(u_pos_s, u_neg_s)
    upna_result = evaluate("UPNA-PPI", u_pos_u, u_neg_u)

    results = [dryad_result, upna_result]
    print("\n=== Summary ===")
    for r in results:
        print(f"{r['name']}: AUROC(optimistic)={r['auroc_optimistic']:.3f}, "
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
