"""Layer 1-6 orchestration (ARCHITECTURE.md §4).

candidate generation -> hard exclusion -> three-stream scoring -> fusion ->
matching & balancing -> train/eval products with provenance. This is the
walking skeleton: every architectural seam is exercised end-to-end; individual
streams and layers are thin and thicken in place.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np

from .candidates import generate_candidates
from .fusion import Fused, fuse
from .graph import TypedInteractionGraph
from .matching import Scored, degree_matched_eval, hard_train
from .schema import NegativeRecord
from .streams import EmbeddingStream, LiteratureStream, StructuredStream, Stream


@dataclass
class PipelineConfig:
    n_eval: int = 300
    n_train: int = 300
    max_pool: int = 200_000
    seed: int = 0
    weights: dict[str, float] | None = None
    # node type whose positive-degree distribution the eval set is matched to
    # (the leakage-prone confounder, e.g. "host" for host-pathogen). None = match
    # on the summed degree of both endpoints.
    match_on_type: str | None = None
    # the lowest-confidence fraction of emitted negatives to flag as suspected
    # false negatives (pool-relative — the pairs to route to the literature/self-
    # training review, per P4). Set to 0 to disable.
    false_negative_pct: float = 0.03
    sources_version: str = "sars-cov2-network/v1+negatome2"


@dataclass
class PipelineResult:
    records: list[NegativeRecord]
    stats: dict = field(default_factory=dict)


def default_streams() -> list[Stream]:
    return [StructuredStream(), EmbeddingStream(), LiteratureStream()]


def run_pipeline(
    graph: TypedInteractionGraph,
    config: PipelineConfig | None = None,
    streams: list[Stream] | None = None,
) -> PipelineResult:
    cfg = config or PipelineConfig()
    streams = streams or default_streams()
    for s in streams:
        s.fit(graph)

    # Layer 1 — candidates (admissible non-edges)
    candidates = generate_candidates(graph, max_pool=cfg.max_pool, seed=cfg.seed)

    # Layers 2/3/5.x + 6 — score, veto, fuse
    kept: list[Scored] = []
    n_vetoed = 0
    topo_raw: list[float] = []
    fused_pairs: list[tuple[str, str, Fused, float]] = []
    for (u, v) in candidates:
        scores = [s.score(graph, u, v) for s in streams]
        f = fuse(scores, cfg.weights)
        if f.vetoed:
            n_vetoed += 1
            continue
        emb = next((sc for sc in scores if sc.stream == "embedding"), None)
        topo = (emb.evidence.get("topo", 0.0) if emb and emb.evidence else 0.0)
        topo_raw.append(topo)
        fused_pairs.append((u, v, f, topo))

    # hardness = topo percentile across the surviving pool (Layer 5 distance knob).
    # Use "fraction strictly below" so the many topo==0 pairs share hardness 0
    # rather than being spread arbitrarily across the range by argsort ties.
    if topo_raw:
        arr = np.array(topo_raw)
        srt = np.sort(arr)
        pct = np.searchsorted(srt, arr, side="left") / max(len(arr) - 1, 1)
    else:
        pct = np.zeros(0)
    for i, (u, v, f, topo) in enumerate(fused_pairs):
        kept.append(Scored(u=u, v=v, confidence=f.confidence,
                           hardness=round(float(pct[i]), 4), sub_scores=f.sub_scores))

    # Layer 5 — two products from the same pool (P1). Match the eval set on the
    # confounder node type's positive-degree distribution.
    pos_degree = {n: graph.degree(n) for n in graph.g.nodes()}

    def _match_weight(s: Scored) -> float:
        if cfg.match_on_type is None:
            return pos_degree.get(s.u, 0) + pos_degree.get(s.v, 0)
        return sum(pos_degree.get(nd, 0) for nd in (s.u, s.v)
                   if graph.node_type.get(nd) == cfg.match_on_type)

    weights = np.array([_match_weight(s) for s in kept], dtype=float)
    eval_set = degree_matched_eval(kept, weights, cfg.n_eval, seed=cfg.seed)
    eval_keys = {(s.u, s.v) for s in eval_set}
    train_set = hard_train(kept, cfg.n_train, exclude=eval_keys)

    records: list[NegativeRecord] = []
    records += [_record(graph, s, "eval", cfg) for s in eval_set]
    records += [_record(graph, s, "train", cfg) for s in train_set]

    # suspected false negatives: pool-relative lowest-confidence tail (P4) — the
    # pairs the streams are least sure are true negatives, to route to review.
    if records and cfg.false_negative_pct > 0:
        thresh = float(np.quantile([r.confidence for r in records],
                                   cfg.false_negative_pct))
        n_fn = 0
        for r in records:
            if r.confidence <= thresh:
                r.flags.append("suspected_false_negative")
                n_fn += 1

    stats = {
        "graph": graph.summary(),
        "candidates": len(candidates),
        "vetoed_known_positive": n_vetoed,
        "scored_pool": len(kept),
        "emitted": {"eval": len(eval_set), "train": len(train_set)},
        "streams": [s.name for s in streams],
    }
    return PipelineResult(records=records, stats=stats)


def _record(graph: TypedInteractionGraph, s: Scored, mode: str,
            cfg: PipelineConfig) -> NegativeRecord:
    flags: list[str] = []
    if s.hardness >= 0.9:
        flags.append("near_boundary")
    prov = {
        "source_graph": graph.name,
        "sources_version": cfg.sources_version,
        "filters_fired": ["candidate_nonedge", "known_positive_exclusion"],
        "node_types": [graph.node_type.get(s.u), graph.node_type.get(s.v)],
        "sub_scores": s.sub_scores,
    }
    return NegativeRecord(
        u=s.u, v=s.v, mode=mode, confidence=s.confidence, hardness=s.hardness,
        streams={"structured": s.sub_scores.get("structured"),
                 "literature": s.sub_scores.get("literature"),
                 "embedding": s.sub_scores.get("embedding")},
        provenance=prov, flags=flags,
    )
