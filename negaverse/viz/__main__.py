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
_DRYAD_ESM2_NPZ = "local-docs/dryad-ppi/esm2_t6_emb.npz"


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


def _dryad_sequence_axis(graph, seed):
    """A supervised "looks like a real interaction" x-axis for DRYAD.

    Topology is inert here (the graph is too sparse) and ESM2 manifold-resemblance
    is flat (AUROC ~0.57). But an ESM2 classifier trained on positives vs DRYAD's
    GOLD non-interactors (concat features, RandomForest) separates at AUROC ~0.93,
    so read its P(real) as x. Leakage-free: positives are scored out-of-fold; the
    other regimes plotted on the map (random, our chosen negatives, risky) were
    never in training and were selected by topology — independent of ESM2 — so
    scoring them here is honest, and reveals whether any "look real" by sequence.
    Returns (x_fn, title, missing_note) for compute_traces; None where unbuildable."""
    import numpy as np
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import cross_val_predict
    from ..io.embeddings import load_embeddings_npz

    try:
        emb = load_embeddings_npz(_DRYAD_ESM2_NPZ)
    except Exception:
        return None

    def _f(u, v):
        if u not in emb or v not in emb:
            return None
        a, b = np.asarray(emb[u], float), np.asarray(emb[v], float)
        return np.concatenate([np.minimum(a, b), np.maximum(a, b)])   # order-invariant concat

    pos = [tuple(e) for e in graph.g.edges()]
    gold = []
    for line in Path(_DRYAD_TSV).read_text().splitlines()[1:]:
        pr, cat = line.split("\t")
        if cat == "negative":
            a, b = pr.split("_"); gold.append((a, b))

    X, y, keys = [], [], []
    for e, lab in [(e, 1) for e in pos] + [(e, 0) for e in gold]:
        f = _f(*e)
        if f is not None:
            X.append(f); y.append(lab); keys.append(frozenset(e))
    if len(set(y)) < 2:
        return None
    X, y = np.asarray(X), np.asarray(y)
    oof = cross_val_predict(RandomForestClassifier(200, random_state=seed, n_jobs=-1),
                            X, y, cv=5, method="predict_proba")[:, 1]
    oof_map = {k: float(s) for k, s in zip(keys, oof)}        # training pairs -> OOF score
    full = RandomForestClassifier(200, random_state=seed, n_jobs=-1).fit(X, y)

    def x_fn(u, v):
        k = frozenset((u, v))
        if k in oof_map:
            return oof_map[k]                                 # positive/gold -> out-of-fold
        f = _f(u, v)
        return None if f is None else float(full.predict_proba([f])[0, 1])

    return (x_fn, "looks real (ESM2 model vs gold non-interactors)", "no sequence embedding")


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

    # DRYAD's graph is too sparse for topology to separate anything (only ~0.2%
    # of non-edges share a neighbour → every candidate sits at the x-floor and our
    # negatives land on top of random), and ESM2 manifold-*resemblance* is flat
    # too (AUROC ~0.57). What *does* separate DRYAD is a supervised ESM2 model
    # trained on positives vs the GOLD non-interactors (AUROC ~0.93). Use its
    # P(real) as the x-axis (see _dryad_sequence_axis).
    x_axis = _dryad_sequence_axis(graph, args.seed) if args.dataset == "dryad" else None

    render_all(graph, result.records, out, stats=result.stats, seed=args.seed, x_axis=x_axis)
    report = build_report(out, title="negaverse", subtitle=f"{args.dataset} demo run")
    print(f"wrote dashboard: {report}")


if __name__ == "__main__":
    main()
