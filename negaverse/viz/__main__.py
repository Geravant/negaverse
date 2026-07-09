"""Render the Phase-1 demo panels.

    python -m negaverse.viz                 # SARS-CoV-2 demo graph
    python -m negaverse.viz --dataset huri  # human PPI graph

Writes out/separability.png and out/funnel.png.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from ..pipeline import PipelineConfig, run_pipeline
from ..io import load_sars_cov2_graph, load_huri_graph
from . import render_all


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="negaverse.viz")
    ap.add_argument("--dataset", choices=["sars", "huri"], default="sars")
    ap.add_argument("--out", default="out")
    ap.add_argument("--n-train", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    if args.dataset == "huri":
        graph = load_huri_graph()
        cfg = PipelineConfig(modality="ppi", n_eval=0, n_train=args.n_train,
                             max_pool=40_000, seed=args.seed,
                             filters=["known_positive_veto", "structured", "topology"])
    else:
        graph = load_sars_cov2_graph()
        cfg = PipelineConfig(n_eval=0, n_train=args.n_train, seed=args.seed,
                             match_on_type="viral",
                             filters=["known_positive_veto", "structured", "topology"])

    print(f"graph: {graph.summary()}")
    result = run_pipeline(graph, cfg)
    written = render_all(graph, result.records, args.out, stats=result.stats,
                         seed=args.seed)
    print("wrote:", ", ".join(str(Path(p)) for p in written))


if __name__ == "__main__":
    main()
