"""The value question, cleanly: is negaverse better than random negatives?

For each dataset, train a link-predictor on real positives + one negative set,
test on a FIXED held-out set (held-out positives + that dataset's GOLD/hard
negatives), and compare strategies. Every cell is a real downstream model AUROC —
nothing to caveat, all rows comparable:

  * features are SPECTRAL (SVD embeddings) — independent of negaverse's topology
    selection, so a win isn't the selection signal read back out (no circularity);
  * test negatives are the dataset's biologically/structurally validated GOLD
    negatives (not easy random), so it's the meaningful test;
  * the test set is identical across strategies — only the *training* negatives
    differ (random vs negaverse), which is exactly the thing under test.

Answer to read off: is `negaverse − random` (Δ) positive? If not, we should just
use random negatives.

    PYTHONPATH=. python3 scripts/bench_negaverse_vs_random.py
"""
from __future__ import annotations

import numpy as np

from negaverse.bench import run_benchmark
from negaverse.graph import TypedInteractionGraph
from negaverse.io import load_huri_graph, load_negatome_in_ensembl_space

_STRATS = ("random", "negaverse", "negaverse_stacked")
_SEEDS = (0, 1, 2)


def _load_huri():
    g = load_huri_graph()
    gold = load_negatome_in_ensembl_space(set(g.g.nodes()))
    return g, gold


def _load_dryad():
    pos, neg = [], []
    path = "local-docs/dryad-ppi/benchmarks/benchmarks/positives_and_negatives.tsv"
    with open(path) as fh:
        next(fh)
        for line in fh:
            pair, cat = line.rstrip("\n").split("\t")
            a, b = pair.split("_")
            (pos if cat == "positive" else neg).append((a, b))
    nodes = {p for pr in pos + neg for p in pr}
    g = TypedInteractionGraph.from_edges(
        pos, {n: "protein" for n in nodes},
        admissible_types=[("protein", "protein")], name="dryad")
    gold = {frozenset(p) for p in neg}
    return g, gold


def main():
    datasets = [("HuRI", _load_huri), ("DRYAD", _load_dryad)]
    print("=" * 78)
    print("Is negaverse better than random?  (downstream link-prediction, spectral")
    print(" features, tested on held-out positives + GOLD negatives — 3 seeds)")
    print("=" * 78)
    hdr = (f"{'dataset':<9}{'random':>9}{'negaverse':>11}{'Δ':>8}"
           f"{'stacked':>10}{'Δ':>8}")
    print(hdr); print("-" * len(hdr))
    for name, loader in datasets:
        graph, gold = loader()
        acc = {s: [] for s in _STRATS}
        for seed in _SEEDS:
            try:
                res = run_benchmark(graph, seed=seed, test_frac=0.2,
                                    max_positives=6000, max_pool=40000,
                                    strategies=_STRATS, feature_set="spectral",
                                    gold_test_neg=gold)
                for s, m in res.strategies.items():
                    acc[s].append(m["auroc"])
            except Exception as e:
                print(f"  {name}: seed {seed} failed ({type(e).__name__}: {e})")
        mean = {s: float(np.mean(v)) if v else float("nan") for s, v in acc.items()}
        rnd = mean["random"]
        print(f"{name:<9}{rnd:>9.3f}{mean['negaverse']:>11.3f}"
              f"{mean['negaverse']-rnd:>+8.3f}"
              f"{mean['negaverse_stacked']:>10.3f}{mean['negaverse_stacked']-rnd:>+8.3f}")
    print("-" * len(hdr))
    print("\n  Δ = strategy − random.  Positive Δ => negaverse's negatives train a")
    print("  better model than random.  Negative/zero => just use random.")


if __name__ == "__main__":
    main()
