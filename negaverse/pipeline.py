"""Hourglass orchestration (docs/IMPLEMENTATION-PLAN.md §Phase 0).

    candidates
      → VETO pass    (cheap hard filters; drop known positives)      [funnel]
      → GRADED pass  (cheap graded filters in parallel; merge)       [parallel]
      → GATED pass   (expensive filters on the contested tail)       [funnel]
      → matching & split → train/eval products with provenance

Filters are discovered from the registry by modality; adding one requires no
edits here (see docs/ADDING-A-FILTER.md).
"""
from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np

from .candidates import generate_candidates
from .fusion import fuse
from .graph import TypedInteractionGraph
from .matching import Scored, degree_matched_eval, hard_train
from .schema import NegativeRecord, StreamScore
from .streams import Filter, Stage, build_filters


@dataclass
class PipelineConfig:
    modality: str = "ppi"
    filters: list[str] | None = None     # None = registry defaults for the modality
    n_eval: int = 300
    n_train: int = 300
    max_pool: int = 200_000
    seed: int = 0
    weights: dict[str, float] | None = None
    # node type whose positive-degree distribution the eval set is matched to
    # (the leakage-prone confounder). None = match on both endpoints' summed degree.
    match_on_type: str | None = None
    # lowest-confidence fraction to flag as suspected false negatives and to route
    # to the GATED stage. Set to 0 to disable.
    false_negative_pct: float = 0.03
    # max pairs sent to the (expensive) GATED filters; None = no cap.
    gated_max: int | None = 8
    sources_version: str = "sars-cov2-network/v1+negatome2"


@dataclass
class PipelineResult:
    records: list[NegativeRecord]
    stats: dict = field(default_factory=dict)


def _fuse_confidence(sub_scores: dict[str, float | None],
                     weights: dict[str, float] | None) -> float:
    w = weights or {}
    num = den = 0.0
    for name, val in sub_scores.items():
        if val is None:
            continue
        wt = w.get(name, 1.0)
        num += wt * val
        den += wt
    return round(num / den, 4) if den > 0 else 0.5


def run_pipeline(
    graph: TypedInteractionGraph,
    config: PipelineConfig | None = None,
    filters: list[Filter] | None = None,
) -> PipelineResult:
    cfg = config or PipelineConfig()
    filters = filters or build_filters(cfg.modality, cfg.filters)
    for f in filters:
        f.fit(graph)
    veto_f = [f for f in filters if f.stage == Stage.VETO]
    graded_f = [f for f in filters if f.stage == Stage.GRADED]
    gated_f = [f for f in filters if f.stage == Stage.GATED]
    score_names = [f.name for f in graded_f + gated_f]

    # Layer 1 — candidates (admissible non-edges)
    candidates = generate_candidates(graph, max_pool=cfg.max_pool, seed=cfg.seed)

    # --- VETO pass (funnel): drop candidates any hard filter rejects ---
    survivors: list[tuple[str, str]] = []
    n_vetoed = 0
    for (u, v) in candidates:
        if any(f.score(graph, u, v).veto for f in veto_f):
            n_vetoed += 1
            continue
        survivors.append((u, v))

    # --- GRADED pass (parallel): score with each graded filter, merge ---
    kept: list[Scored] = []
    topo_raw: list[float] = []
    staged: list[tuple[str, str, dict, float]] = []
    for (u, v) in survivors:
        scores = [f.score(graph, u, v) for f in graded_f]
        fused = fuse(scores, cfg.weights)
        if fused.vetoed:              # a graded filter may still hard-veto
            n_vetoed += 1
            continue
        emb = next((s for s in scores if s.stream == "embedding"), None)
        topo = (emb.evidence.get("topo", 0.0) if emb and emb.evidence else 0.0)
        topo_raw.append(topo)
        staged.append((u, v, dict(fused.sub_scores), topo))

    # hardness = topo percentile across the surviving pool ("fraction strictly below")
    if topo_raw:
        arr = np.array(topo_raw)
        srt = np.sort(arr)
        pct = np.searchsorted(srt, arr, side="left") / max(len(arr) - 1, 1)
    else:
        pct = np.zeros(0)
    for i, (u, v, sub, topo) in enumerate(staged):
        sub = {**{n: None for n in score_names}, **sub}  # ensure all names present
        conf = _fuse_confidence(sub, cfg.weights)
        kept.append(Scored(u=u, v=v, confidence=conf,
                           hardness=round(float(pct[i]), 4), sub_scores=sub))

    # Layer 5 — two products from the same pool. Degree-match the eval set on the
    # confounder node type's positive-degree distribution.
    pos_degree = {n: graph.degree(n) for n in graph.g.nodes()}

    def _match_weight(s: Scored) -> float:
        if cfg.match_on_type is None:
            return pos_degree.get(s.u, 0) + pos_degree.get(s.v, 0)
        return sum(pos_degree.get(nd, 0) for nd in (s.u, s.v)
                   if graph.node_type.get(nd) == cfg.match_on_type)

    mw = np.array([_match_weight(s) for s in kept], dtype=float)
    eval_set = degree_matched_eval(kept, mw, cfg.n_eval, seed=cfg.seed)
    eval_keys = {(s.u, s.v) for s in eval_set}
    train_set = hard_train(kept, cfg.n_train, exclude=eval_keys)

    # --- GATED pass (funnel): run expensive filters only on the contested tail
    # of the *emitted* set, so the LLM verdict is fused into pairs we actually
    # ship (not arbitrary pool pairs that get dropped by matching). ---
    gated_reviewed = 0
    gated_flags: dict[tuple[str, str], list[str]] = {}
    gated_evidence: dict[tuple[str, str], dict] = {}
    if gated_f:
        contested = _contested(eval_set + train_set, cfg.false_negative_pct, cfg.gated_max)
        for s in contested:
            reviewed = merged = False
            for f in gated_f:
                sc = f.score(graph, s.u, s.v)
                ev = sc.evidence or {}
                if ev.get("gated_status") != "reviewed":
                    continue                       # filter abstained / skipped
                gated_evidence.setdefault((s.u, s.v), {})[f.name] = ev
                if sc.flags:
                    gated_flags.setdefault((s.u, s.v), []).extend(sc.flags)
                if sc.value is not None:
                    s.sub_scores[f.name] = sc.value
                    merged = True
                reviewed = True
            if reviewed:
                gated_reviewed += 1
            if merged:
                s.confidence = _fuse_confidence(s.sub_scores, cfg.weights)

    records: list[NegativeRecord] = []
    records += [_record(graph, s, "eval", cfg, score_names, gated_flags, gated_evidence)
                for s in eval_set]
    records += [_record(graph, s, "train", cfg, score_names, gated_flags, gated_evidence)
                for s in train_set]

    # suspected false negatives: pool-relative lowest-confidence tail
    if records and cfg.false_negative_pct > 0:
        thresh = float(np.quantile([r.confidence for r in records], cfg.false_negative_pct))
        for r in records:
            if r.confidence <= thresh and "suspected_false_negative" not in r.flags:
                r.flags.append("suspected_false_negative")

    stats = {
        "graph": graph.summary(),
        "candidates": len(candidates),
        "vetoed": n_vetoed,
        "scored_pool": len(kept),
        "gated_reviewed": gated_reviewed,
        "emitted": {"eval": len(eval_set), "train": len(train_set)},
        "filters": {
            "veto": [f.name for f in veto_f],
            "graded": [f.name for f in graded_f],
            "gated": [f.name for f in gated_f],
        },
    }
    return PipelineResult(records=records, stats=stats)


def _contested(kept: list[Scored], pct: float, gated_max: int | None) -> list[Scored]:
    """Near-boundary (high hardness) or lowest-confidence pairs — the tail worth
    the expensive gated review. Capped at gated_max (lowest-confidence first)."""
    out = [s for s in kept if s.hardness >= 0.9]
    if pct > 0 and kept:
        thresh = float(np.quantile([s.confidence for s in kept], pct))
        out += [s for s in kept if s.confidence <= thresh]
    seen, uniq = set(), []
    for s in out:
        if (s.u, s.v) not in seen:
            seen.add((s.u, s.v))
            uniq.append(s)
    uniq.sort(key=lambda s: s.confidence)      # most contested (lowest conf) first
    if gated_max is not None:
        uniq = uniq[:gated_max]
    return uniq


def _record(graph: TypedInteractionGraph, s: Scored, mode: str, cfg: PipelineConfig,
            score_names: list[str], gated_flags: dict, gated_evidence: dict) -> NegativeRecord:
    flags: list[str] = list(gated_flags.get((s.u, s.v), []))
    if s.hardness >= 0.9:
        flags.append("near_boundary")
    prov = {
        "source_graph": graph.name,
        "sources_version": cfg.sources_version,
        "filters_fired": ["candidate_nonedge", "known_positive_veto"],
        "node_types": [graph.node_type.get(s.u), graph.node_type.get(s.v)],
        "sub_scores": s.sub_scores,
    }
    if (s.u, s.v) in gated_evidence:
        prov["gated"] = gated_evidence[(s.u, s.v)]
    return NegativeRecord(
        u=s.u, v=s.v, mode=mode, confidence=s.confidence, hardness=s.hardness,
        streams={n: s.sub_scores.get(n) for n in score_names},
        provenance=prov, flags=flags,
    )
