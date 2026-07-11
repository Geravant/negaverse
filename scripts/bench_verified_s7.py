"""§7 of docs/FILTER-EFFECTIVENESS.md — the decision-grade test.

The stacked strategy reaches near-parity but stays slightly negative (Δ≈−0.006
on HuRI). The hypothesis: the residual harm is HIDDEN POSITIVES — topology-hard
"negatives" that are actually real interactions the graph simply doesn't record.
If so, letting the LLM judge (now fed gene-symbol context so it can actually
reason about ENSG nodes) DROP the pairs it flags `suspected_false_negative`
should lift Δ toward / past zero.

This runs, on HuRI, with SPECTRAL features (independent of topology selection —
no circularity) and GOLD (Negatome) test negatives:

    random   vs   negaverse_stacked   vs   negaverse_verified

Read off Δ_verified − Δ_stacked: positive => dropping judged hidden-positives
helps, i.e. the residual harm really was contamination, not the hardness itself.

    PYTHONPATH=. python3 scripts/bench_verified_s7.py [--seeds 0 1 2] [--judge-cap 800]

Needs ANTHROPIC_API_KEY (loaded from .env) — verified strategy calls Haiku.
Verdicts are cached (feature-hashed) so re-runs and overlapping seeds are cheap.
"""
from __future__ import annotations

import argparse
import numpy as np

from negaverse.cli import _load_dotenv
from negaverse.bench import run_benchmark
from negaverse.io import load_huri_graph, load_negatome_in_ensembl_space

_STRATS = ("random", "negaverse_stacked", "negaverse_verified")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--judge-cap", type=int, default=800)
    ap.add_argument("--max-positives", type=int, default=6000)
    args = ap.parse_args()

    _load_dotenv()                                   # so the judge sees ANTHROPIC_API_KEY

    graph = load_huri_graph()
    gold = load_negatome_in_ensembl_space(set(graph.g.nodes()))
    print("=" * 78)
    print(f"§7 verified test — HuRI, spectral features, GOLD (Negatome) test negatives")
    print(f"seeds={args.seeds}  judge_cap={args.judge_cap}  gold_negatives={len(gold)}")
    print("=" * 78)

    acc = {s: [] for s in _STRATS}
    for seed in args.seeds:
        try:
            res = run_benchmark(graph, seed=seed, test_frac=0.2,
                                max_positives=args.max_positives, max_pool=40000,
                                strategies=_STRATS, feature_set="spectral",
                                gold_test_neg=gold)
            for s, m in res.strategies.items():
                acc[s].append(m["auroc"])
                print(f"  seed {seed}  {s:<20} AUROC={m['auroc']:.4f}  "
                      f"n_train_neg={m['n_train_neg']}")
        except Exception as e:
            print(f"  seed {seed} failed ({type(e).__name__}: {e})")

    mean = {s: float(np.mean(v)) if v else float("nan") for s, v in acc.items()}
    rnd = mean["random"]
    print("-" * 78)
    print(f"  {'strategy':<20}{'mean AUROC':>12}{'Δ vs random':>14}")
    for s in _STRATS:
        print(f"  {s:<20}{mean[s]:>12.4f}{mean[s]-rnd:>+14.4f}")
    print("-" * 78)
    d_stacked = mean["negaverse_stacked"] - rnd
    d_verified = mean["negaverse_verified"] - rnd
    print(f"\n  Δ_stacked  = {d_stacked:+.4f}")
    print(f"  Δ_verified = {d_verified:+.4f}")
    print(f"  lift from dropping judged hidden-positives = {d_verified - d_stacked:+.4f}")
    if d_verified >= 0:
        print("  => verified negatives reach/exceed parity: the residual harm was contamination.")
    elif d_verified > d_stacked:
        print("  => dropping hidden-positives helps but doesn't fully close the gap.")
    else:
        print("  => no lift: the residual harm is the hardness itself, not hidden positives.")


if __name__ == "__main__":
    main()
