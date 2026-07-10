"""Render the Phase-1 demo panels + a single-page HTML dashboard.

    python -m negaverse.viz                 # SARS-CoV-2 demo graph
    python -m negaverse.viz --dataset huri  # human PPI graph

Writes out/*.png and out/report.html (open it in a browser).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from ..pipeline import PipelineConfig, run_pipeline
from .. import eval as ev
from ..io import load_sars_cov2_graph, load_huri_graph
from . import render_all, build_report


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="negaverse.viz")
    ap.add_argument("--dataset", choices=["sars", "huri"], default="sars")
    ap.add_argument("--out", default="out")
    ap.add_argument("--n-train", type=int, default=400)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    if args.dataset == "huri":
        graph = load_huri_graph()
        cfg = PipelineConfig(modality="ppi", n_eval=args.n_train, n_train=args.n_train,
                             max_pool=40_000, seed=args.seed,
                             filters=["known_positive_veto", "structured", "topology"])
    else:
        graph = load_sars_cov2_graph()
        cfg = PipelineConfig(n_eval=args.n_train, n_train=args.n_train, seed=args.seed,
                             match_on_type="viral",
                             filters=["known_positive_veto", "structured", "topology"])

    print(f"graph: {graph.summary()}")
    result = run_pipeline(graph, cfg)

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    validation = {
        "leakage_known_positive": ev.leakage(graph, result.records),
        "hardness_split": ev.hardness_split(result.records),
    }
    (out / "stats.json").write_text(json.dumps(
        {"stats": result.stats, "validation": validation}, indent=2))

    render_all(graph, result.records, out, stats=result.stats, seed=args.seed)
    report = build_report(out, title="negaverse", subtitle=f"{args.dataset} demo run")
    print(f"wrote dashboard: {report}")


if __name__ == "__main__":
    main()
