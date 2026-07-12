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
from typing import Callable

import numpy as np

from .candidates import generate_candidates
from .fusion import fuse
from .graph import TypedInteractionGraph
from .ig.entropy_fusion import binary_entropy, entropy_weighted_fuse
from .matching import Scored, degree_matched_eval, hard_train, select_train
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
    # how to pick the n_train emitted negatives from the scored pool:
    #   "stacked" (default) — the topology-hard tail RE-RANKED by fused biology
    #      confidence, keeping the pairs every independent signal agrees are true
    #      negatives. Beats random downstream at proper coverage (FILTER-EFFECTIVENESS §11).
    #   "safe" — the highest fused-confidence negatives across the WHOLE pool
    #      (representative + clean); also beats/ties random.
    #   "hard" — the topology-hardest tail alone (nearest the positive manifold).
    #      Historically the default; the ONLY arm that loses to random (§11) — it is
    #      hidden-positive enriched and, on sparse graphs, degenerates into a hub
    #      filter (§10). Kept for ablation, not recommended.
    train_selection: str = "stacked"
    psm_hardness_cap: float = 0.7          # for train_selection="psm": clean-pool hardness ceiling
    # for train_selection="mixture": (representative, safe, hard) fractions of n_train
    mixture_proportions: tuple = (0.6, 0.3, 0.1)
    weights: dict[str, float] | None = None
    # node type whose confounder distribution the eval set is matched to (the
    # leakage-prone confounder). None = match on both endpoints' summed statistic.
    match_on_type: str | None = None
    # per-node confounder statistic to degree-match the eval set on. None = graph
    # degree (the PPI default); supply e.g. molecular weight for other modalities.
    match_weight_fn: Callable[[TypedInteractionGraph, str], float] | None = None
    # lowest-confidence fraction to flag as suspected false negatives and to route
    # to the GATED stage. Set to 0 to disable.
    false_negative_pct: float = 0.03
    # max pairs sent to the (expensive) GATED filters; None = no cap (judge the
    # whole emitted contested tail — scales with n_eval+n_train). The verdict cache
    # (streams/literature.py) makes re-runs cheap, so verify-everything is the
    # default; set an int only to bound cost on a first, un-cached run.
    gated_max: int | None = None
    sources_version: str = "sars-cov2-network/v1+negatome2"
    # fusion strategy: "mean" = fixed-weight mean (default, unchanged);
    # "entropy" = weight each stream by how decisive it is (IG Ch4, ig/).
    fusion_mode: str = "mean"
    fusion_lam: float = 1.0              # entropy sharpness; 0 == "mean"
    # route pairs where two independent signals disagree by at least this margin to
    # the GATED review — that's where an independent signal's unique value lives
    # (docs/IG-FEATURES.md §3c). 0 disables.
    disagree_route_thresh: float = 0.25
    # which stream pairs to check for disagreement. Default is the PPI graph views;
    # a different modality supplies its own, e.g. [("chemistry", "structure")]. No
    # effect unless both named streams are active.
    disagree_pairs: list[tuple[str, str]] = field(
        default_factory=lambda: [("topology", "manifold")])


@dataclass
class PipelineResult:
    records: list[NegativeRecord]
    stats: dict = field(default_factory=dict)


def _fuse_confidence(sub_scores: dict[str, float | None],
                     weights: dict[str, float] | None,
                     mode: str = "mean", lam: float = 1.0,
                     reported: dict[str, float | None] | None = None) -> float:
    w = weights or {}
    rep = reported or {}
    num = den = 0.0
    for name, val in sub_scores.items():
        if val is None:
            continue
        base = w.get(name, 1.0)
        # entropy mode: weight by decisiveness. Prefer a stream's *reported*
        # confidence (evidence["confidence"] — real peakedness, e.g. topology's
        # structural support or the LLM's vote agreement); fall back to the
        # scalar 1−H(value) proxy, which alone can backfire (IG-FEATURES §1).
        if mode == "entropy":
            rc = rep.get(name)
            dec = rc if rc is not None else (1.0 - binary_entropy(val))
            wt = base * (1.0 + lam * dec)
        else:
            wt = base
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
    # which GRADED filter supplies the hardness signal (declared, not hardcoded);
    # first one wins. Empty => hardness is 0 everywhere (split falls back to order).
    hardness_names = {f.name for f in graded_f if getattr(f, "provides_hardness", False)}

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
    staged: list[tuple[str, str, dict, float, dict]] = []
    graded_flags: dict[tuple[str, str], list[str]] = {}
    for (u, v) in survivors:
        scores = [f.score(graph, u, v) for f in graded_f]
        fused = (entropy_weighted_fuse(scores, cfg.weights, cfg.fusion_lam)
                 if cfg.fusion_mode == "entropy" else fuse(scores, cfg.weights))
        if fused.vetoed:              # a graded filter may still hard-veto
            n_vetoed += 1
            continue
        # carry graded-filter flags (e.g. a rule's `different_compartment`,
        # topology's `easy_negative`) through to the emitted record for auditability
        fl = [f for sc in scores for f in sc.flags]
        if fl:
            graded_flags[(u, v)] = fl
        # each stream's *reported* confidence (evidence["confidence"]) — real
        # peakedness that entropy fusion should weight by, kept for the GATED re-fuse.
        reported = {s.stream: (s.evidence or {}).get("confidence") for s in scores}
        # hardness driver = the value from whichever GRADED filter declares
        # provides_hardness (its evidence["hardness"]/["risk"] magnitude); higher =
        # more like a real edge = harder negative. No PPI filter name is hardcoded.
        hs = next((s for s in scores if s.stream in hardness_names), None)
        ev = (hs.evidence or {}) if hs else {}
        topo = ev.get("hardness", ev.get("risk", ev.get("topo", 0.0)))
        topo_raw.append(topo)
        staged.append((u, v, dict(fused.sub_scores), topo, reported))

    # hardness = topo percentile across the surviving pool ("fraction strictly below")
    if topo_raw:
        arr = np.array(topo_raw)
        srt = np.sort(arr)
        pct = np.searchsorted(srt, arr, side="left") / max(len(arr) - 1, 1)
    else:
        pct = np.zeros(0)
    for i, (u, v, sub, topo, reported) in enumerate(staged):
        sub = {**{n: None for n in score_names}, **sub}  # ensure all names present
        conf = _fuse_confidence(sub, cfg.weights, cfg.fusion_mode, cfg.fusion_lam, reported)
        kept.append(Scored(u=u, v=v, confidence=conf, conf_evidence=reported,
                           hardness=round(float(pct[i]), 4), sub_scores=sub,
                           degsum=graph.degree(u) + graph.degree(v)))

    # Layer 5 — two products from the same pool. Match the eval set on a per-node
    # confounder statistic — graph degree by default (the PPI leakage confounder),
    # or any modality-supplied match_weight_fn.
    node_stat_fn = cfg.match_weight_fn or (lambda g, n: g.degree(n))
    node_stat = {n: node_stat_fn(graph, n) for n in graph.g.nodes()}

    def _match_weight(s: Scored) -> float:
        if cfg.match_on_type is None:
            return node_stat.get(s.u, 0) + node_stat.get(s.v, 0)
        return sum(node_stat.get(nd, 0) for nd in (s.u, s.v)
                   if graph.node_type.get(nd) == cfg.match_on_type)

    mw = np.array([_match_weight(s) for s in kept], dtype=float)
    eval_set = degree_matched_eval(kept, mw, cfg.n_eval, seed=cfg.seed)
    eval_keys = {(s.u, s.v) for s in eval_set}
    pos_degsums = [graph.degree(a) + graph.degree(b) for (a, b) in graph.g.edges()]
    train_set = select_train(kept, cfg.n_train, exclude=eval_keys, mode=cfg.train_selection,
                             proportions=cfg.mixture_proportions, seed=cfg.seed,
                             pos_degsums=pos_degsums, psm_cap=cfg.psm_hardness_cap)

    # signal disagreement: two independent streams conflicting is worth the
    # expensive review even when the fused confidence looks unremarkable — an
    # independent signal's unique value lives where it disagrees (IG-FEATURES §3c).
    # Which stream pairs count is config (cfg.disagree_pairs), not hardcoded.
    disagree_flags = _disagreement_flags(eval_set + train_set,
                                         cfg.disagree_route_thresh, cfg.disagree_pairs)
    disagree_keys = set(disagree_flags)
    for k, flags in disagree_flags.items():
        graded_flags.setdefault(k, []).extend(flags)

    # --- GATED pass (funnel): run expensive filters only on the contested tail
    # of the *emitted* set, so the LLM verdict is fused into pairs we actually
    # ship (not arbitrary pool pairs that get dropped by matching). ---
    gated_reviewed = 0
    gated_flags: dict[tuple[str, str], list[str]] = {}
    gated_evidence: dict[tuple[str, str], dict] = {}
    contested_all = _contested(eval_set + train_set, cfg.false_negative_pct, disagree_keys)
    contested = (contested_all[:cfg.gated_max] if cfg.gated_max is not None else contested_all)
    if gated_f:
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
                    s.conf_evidence[f.name] = ev.get("confidence")
                    merged = True
                reviewed = True
            if reviewed:
                gated_reviewed += 1
            if merged:
                s.confidence = _fuse_confidence(s.sub_scores, cfg.weights,
                                                cfg.fusion_mode, cfg.fusion_lam,
                                                s.conf_evidence)

    records: list[NegativeRecord] = []
    records += [_record(graph, s, "eval", cfg, score_names, gated_flags, gated_evidence, graded_flags)
                for s in eval_set]
    records += [_record(graph, s, "train", cfg, score_names, gated_flags, gated_evidence, graded_flags)
                for s in train_set]

    # suspected false negatives: pool-relative lowest-confidence tail
    if records and cfg.false_negative_pct > 0:
        thresh = float(np.quantile([r.confidence for r in records], cfg.false_negative_pct))
        for r in records:
            if r.confidence <= thresh and "suspected_false_negative" not in r.flags:
                r.flags.append("suspected_false_negative")

    # Coverage of the expensive LLM judge over the risky pairs it's meant to
    # vet: every suspected_false_negative is a candidate mislabeled positive, so
    # surface how many actually got a verdict vs. were left to the cheap flag.
    risky = [r for r in records if "suspected_false_negative" in r.flags]
    risky_judged = [r for r in risky
                    if "literature" in gated_evidence.get((r.u, r.v), {})]
    risky_coverage = {
        "risky": len(risky),
        "judged": len(risky_judged),
        "unjudged": len(risky) - len(risky_judged),
        "gated_cap": cfg.gated_max,
        "contested_total": len(contested_all),
    }

    stats = {
        "graph": graph.summary(),
        "candidates": len(candidates),
        "vetoed": n_vetoed,
        "scored_pool": len(kept),
        "gated_reviewed": gated_reviewed,
        "gated_contested": len(contested_all),
        "risky_coverage": risky_coverage,
        "emitted": {"eval": len(eval_set), "train": len(train_set)},
        "filters": {
            "veto": [f.name for f in veto_f],
            "graded": [f.name for f in graded_f],
            "gated": [f.name for f in gated_f],
        },
        "known_positive_sources": next(
            (f.sources_report for f in veto_f if getattr(f, "sources_report", None)), {}),
    }
    return PipelineResult(records=records, stats=stats)


def _disagreement_flags(scored: list[Scored], thresh: float,
                        pairs: list[tuple[str, str]]) -> dict[tuple[str, str], list[str]]:
    """Map each pair to `<a>_<b>_disagreement` flags for every configured stream
    pair (a, b) whose sub-scores differ by ≥ `thresh`. A pair contributes only
    when both its sub-scores are present (opt-in streams may be absent)."""
    out: dict[tuple[str, str], list[str]] = {}
    if not thresh or thresh <= 0 or not pairs:
        return out
    for s in scored:
        for a, b in pairs:
            va, vb = s.sub_scores.get(a), s.sub_scores.get(b)
            if va is not None and vb is not None and abs(va - vb) >= thresh:
                out.setdefault((s.u, s.v), []).append(f"{a}_{b}_disagreement")
    return out


def _contested(kept: list[Scored], pct: float,
               disagree_keys: set[tuple[str, str]] | None = None) -> list[Scored]:
    """The full tail worth the expensive gated review, sorted by priority:
    near-boundary (high hardness), lowest-confidence, or topology-vs-manifold
    disagreement pairs. Returned *uncapped* and priority-sorted — the caller
    applies gated_max, so the number left unjudged by the cap is visible."""
    disagree_keys = disagree_keys or set()
    out = [s for s in kept if s.hardness >= 0.9 or (s.u, s.v) in disagree_keys]
    if pct > 0 and kept:
        thresh = float(np.quantile([s.confidence for s in kept], pct))
        out += [s for s in kept if s.confidence <= thresh]
    seen, uniq = set(), []
    for s in out:
        if (s.u, s.v) not in seen:
            seen.add((s.u, s.v))
            uniq.append(s)
    # disagreement / near-boundary first, then most contested (lowest confidence)
    uniq.sort(key=lambda s: (0 if ((s.u, s.v) in disagree_keys or s.hardness >= 0.9)
                             else 1, s.confidence))
    return uniq


def _record(graph: TypedInteractionGraph, s: Scored, mode: str, cfg: PipelineConfig,
            score_names: list[str], gated_flags: dict, gated_evidence: dict,
            graded_flags: dict | None = None) -> NegativeRecord:
    flags: list[str] = list(gated_flags.get((s.u, s.v), []))
    for f in (graded_flags or {}).get((s.u, s.v), []):
        if f not in flags:
            flags.append(f)
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
