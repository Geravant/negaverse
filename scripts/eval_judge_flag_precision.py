"""Does the LLM judge actually find hidden positives? — measured DIRECTLY, not
through downstream AUROC (which can't detect a 2% relabel; see FILTER-EFFECTIVENESS §7).

The judge flags some HuRI non-edges `suspected_false_negative`. Ground truth for
"this pair really is a hidden positive" = the pair is a recorded interaction in a
database HuRI was NOT built from — BioGRID or IntAct (both already mapped into
HuRI's Ensembl-gene space, restricted to HuRI nodes). A HuRI non-edge that IS a
BioGRID/IntAct interaction is exactly a hidden positive.

For the hard-negative pool we build with the stacked pipeline, we judge the
hardest N pairs and compare, across verdict groups, the hit-rate against
BioGRID∪IntAct:

  precision(suspected_false_negative)   -- what the judge flags
  precision(safe_negative)              -- what the judge clears
  base rate (whole judged pool)         -- random-guess reference
  base rate (random HuRI non-edges)     -- absolute-random reference

If precision(FN) >> base rate AND >> precision(safe_negative), the judge is
finding real hidden positives — regardless of what noise-dominated AUROC does.

    PYTHONPATH=. python3 scripts/eval_judge_flag_precision.py [--n 800] [--seed 0]

Needs ANTHROPIC_API_KEY (.env). Verdicts are cached (feature-hashed), so this
reuses any judging already done by bench_verified_s7.py.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np

from negaverse.cli import _load_dotenv
from negaverse.graph import TypedInteractionGraph
from negaverse.io import load_huri_graph
from negaverse.pipeline import PipelineConfig, run_pipeline
from negaverse.streams import LiteratureFilter

BIOGRID = Path("local-docs/biogrid/biogrid_human_pairs_ensembl.tsv")
INTACT = Path("local-docs/intact/intact_human_pairs_ensembl.tsv")
SYMS = Path("local-docs/mappings/ensg_symbol.tsv")


def _load_pairs(path: Path) -> set:
    out = set()
    if path.exists():
        for line in path.read_text().splitlines():
            if line and not line.startswith("#"):
                p = line.split("\t")
                if len(p) >= 2:
                    out.add(frozenset((p[0], p[1])))
    return out


def _hit_rate(pairs, evidence):
    if not pairs:
        return float("nan"), 0, 0
    hits = sum(1 for u, v in pairs if frozenset((u, v)) in evidence)
    return hits / len(pairs), hits, len(pairs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=800, help="#hardest pairs to judge")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-pool", type=int, default=40000)
    args = ap.parse_args()
    _load_dotenv()

    huri = load_huri_graph()
    huri_edges = {frozenset(e) for e in huri.g.edges()}
    biogrid, intact = _load_pairs(BIOGRID), _load_pairs(INTACT)
    evidence = (biogrid | intact) - huri_edges       # hidden-positive evidence: not a HuRI edge
    print(f"HuRI edges {len(huri_edges):,}; BioGRID {len(biogrid):,}; IntAct {len(intact):,}; "
          f"independent evidence (non-HuRI interactions) {len(evidence):,}")

    # build the stacked hard-negative pool (same pipeline the strategies use)
    train_pos = [tuple(e) for e in huri.g.edges()]
    node_type = {n: "protein" for n in huri.g.nodes()}
    tg = TypedInteractionGraph.from_edges(
        train_pos, node_type, admissible_types=[("protein", "protein")], name="huri-train")
    cfg = PipelineConfig(modality="ppi", n_eval=0, n_train=max(4 * args.n, args.n),
                         max_pool=args.max_pool, seed=args.seed,
                         filters=["known_positive_veto", "structured", "topology", "rules"])
    res = run_pipeline(tg, cfg)
    hard = [r for r in res.records if r.mode == "train"]
    hard.sort(key=lambda r: r.confidence, reverse=True)          # stacked ordering
    base = hard[:args.n * 5] if len(hard) >= args.n else hard    # judge from the confident set
    order = sorted(range(len(base)), key=lambda i: base[i].confidence)[:args.n]  # hardest N within it

    syms = ({l.split("\t")[0]: l.split("\t")[1].strip() for l in SYMS.open() if l.strip()}
            if SYMS.exists() else {})
    lit = LiteratureFilter(enabled=True, provider="auto", votes=1, names=syms)
    lit.fit(tg)

    groups = {"suspected_false_negative": [], "safe_negative": [], "uncertain": [], "skipped": []}
    for i in order:
        r = base[i]
        sc = lit.score(tg, r.u, r.v); ev = sc.evidence or {}
        if ev.get("gated_status") == "reviewed":
            groups.setdefault(ev.get("verdict", "uncertain"), []).append((r.u, r.v))
        else:
            groups["skipped"].append((r.u, r.v))

    judged_pool = [p for g in ("suspected_false_negative", "safe_negative", "uncertain")
                   for p in groups[g]]
    # random HuRI non-edges as an absolute reference
    rng = np.random.default_rng(args.seed)
    nodes = list(huri.g.nodes())
    rand_pairs, seen = [], set()
    while len(rand_pairs) < len(judged_pool) and len(seen) < 50000:
        a, b = nodes[rng.integers(len(nodes))], nodes[rng.integers(len(nodes))]
        k = frozenset((a, b))
        if a != b and k not in huri_edges and k not in seen:
            seen.add(k); rand_pairs.append((a, b))

    print("\n" + "=" * 72)
    print(f"Judge-flag precision vs independent (BioGRID∪IntAct) hidden-positive evidence")
    print(f"seed={args.seed}  judged={len(judged_pool)} of {args.n} hardest stacked negatives")
    print("=" * 72)
    print(f"  {'group':<28}{'n':>6}{'hits':>7}{'hit-rate':>11}")
    print("  " + "-" * 50)
    rows = [("suspected_false_negative (FLAGGED)", groups["suspected_false_negative"]),
            ("safe_negative (CLEARED)", groups["safe_negative"]),
            ("uncertain", groups["uncertain"]),
            ("— whole judged pool (base)", judged_pool),
            ("— random HuRI non-edges (base)", rand_pairs)]
    rates = {}
    for label, pairs in rows:
        rate, hits, n = _hit_rate(pairs, evidence)
        rates[label] = rate
        print(f"  {label:<28}{n:>6}{hits:>7}{rate:>10.1%}")
    print("  " + "-" * 50)
    fn = rates["suspected_false_negative (FLAGGED)"]
    base_pool = rates["— whole judged pool (base)"]
    safe = rates["safe_negative (CLEARED)"]
    if base_pool and not np.isnan(fn):
        print(f"\n  lift(FLAGGED / pool base)     = {fn / base_pool:.2f}×")
    if safe and not np.isnan(fn):
        print(f"  lift(FLAGGED / CLEARED)       = {fn / safe:.2f}×")
    print("\n  Reads: FLAGGED hit-rate >> base and >> CLEARED  => the judge finds")
    print("  real hidden positives (its drops are enriched for true interactions).")


if __name__ == "__main__":
    main()
