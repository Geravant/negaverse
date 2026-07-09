"""Run the downstream-model benchmark on HuRI.

    python -m negaverse.bench                 # random vs negaverse negatives
    python -m negaverse.bench --max-positives 8000 --seed 0
"""
from __future__ import annotations

import argparse

from . import run_benchmark
from ..io import load_huri_graph


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="negaverse.bench")
    ap.add_argument("--max-positives", type=int, default=10_000,
                    help="subsample positives for a faster run (None-like 0 = all)")
    ap.add_argument("--max-pool", type=int, default=40_000)
    ap.add_argument("--test-frac", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--features", choices=["topological", "spectral"], default="topological",
                    help="spectral (SVD+Hadamard) is independent of the selection signal")
    args = ap.parse_args(argv)

    print("Loading HuRI ...")
    graph = load_huri_graph()
    print("  graph:", graph.summary())
    print("Benchmarking (train on positives + random vs negaverse negatives; "
          "test on held-out positives + unbiased random negatives) ...")
    result = run_benchmark(
        graph, seed=args.seed, test_frac=args.test_frac,
        max_positives=args.max_positives or None, max_pool=args.max_pool,
        feature_set=args.features)
    print("\n=== benchmark ===")
    print(result.summary())


if __name__ == "__main__":
    main()
