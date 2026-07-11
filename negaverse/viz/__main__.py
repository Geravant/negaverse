"""Render the Phase-1 demo panels + a single-page HTML dashboard.

    python -m negaverse.viz                  # SARS-CoV-2 demo graph
    python -m negaverse.viz --dataset huri   # human PPI graph
    python -m negaverse.viz --dataset dryad  # DRYAD PPI benchmark (built from its positives)

Writes out/*.png and out/report.html (open it in a browser).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from ..graph import TypedInteractionGraph
from ..pipeline import PipelineConfig, run_pipeline
from .. import eval as ev
from ..io import load_sars_cov2_graph, load_huri_graph
from . import render_all, build_report

_DRYAD_TSV = "local-docs/dryad-ppi/benchmarks/benchmarks/positives_and_negatives.tsv"


def _load_dryad_graph() -> TypedInteractionGraph:
    """DRYAD ships as labelled positive/negative pairs, not a graph. Build the
    interaction graph from its positives so the same pipeline can run on it."""
    pos = []
    with open(_DRYAD_TSV) as fh:
        next(fh)
        for line in fh:
            pair, cat = line.rstrip("\n").split("\t")
            if cat == "positive":
                a, b = pair.split("_")
                pos.append((a, b))
    nodes = {p for e in pos for p in e}
    return TypedInteractionGraph.from_edges(
        pos, {n: "protein" for n in nodes},
        admissible_types=[("protein", "protein")], name="dryad")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="negaverse.viz")
    ap.add_argument("--dataset", choices=["sars", "huri", "dryad"], default="sars")
    ap.add_argument("--out", default="out")
    ap.add_argument("--n-train", type=int, default=400)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    if args.dataset == "huri":
        graph = load_huri_graph()
        cfg = PipelineConfig(modality="ppi", n_eval=args.n_train, n_train=args.n_train,
                             max_pool=40_000, seed=args.seed,
                             filters=["known_positive_veto", "structured", "topology"])
    elif args.dataset == "dryad":
        graph = _load_dryad_graph()
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
