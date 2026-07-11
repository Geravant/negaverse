"""Can the LLM judge distinguish a REAL hidden interaction from a true negative,
given gene-symbol context? — a clean labeled-detection test.

Why not the pool-based precision test (eval_judge_flag_precision.py)? Because the
pipeline's KnownPositiveVeto already loads BioGRID+IntAct (rules/sources.yaml) and
strips every such pair from the candidate pool BEFORE the judge sees it — so 0% of
the hard pool is a recorded interaction (verified). Any hidden positive left in the
pool is, by construction, one NO wired database records, and can't be validated
against those same databases. This test sidesteps the veto entirely: it builds its
own labeled set and judges it directly.

  * POSITIVES = BioGRID∪IntAct interactions that are HuRI non-edges — real
    interactions HuRI simply didn't record (genuine hidden positives).
  * NEGATIVES = random HuRI non-edges absent from HuRI, BioGRID and IntAct —
    the best-available true negatives (not in any wired DB).

Judge all pairs (gene symbols wired in) and read off:
  recall  = P(flagged suspected_false_negative | hidden positive)
  FPR     = P(flagged suspected_false_negative | true negative)
A judge that helps has recall > FPR (it flags real interactions more than negatives).

    PYTHONPATH=. python3 scripts/eval_judge_holdout_recall.py [--k 150] [--seed 0] [--votes 1]

Needs ANTHROPIC_API_KEY (.env). Verdicts cached (feature-hashed).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from negaverse.cli import _load_dotenv
from negaverse.graph import TypedInteractionGraph
from negaverse.io import load_huri_graph
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


def _verdict(lit, tg, u, v):
    sc = lit.score(tg, u, v); ev = sc.evidence or {}
    if ev.get("gated_status") != "reviewed":
        return "skipped"
    return ev.get("verdict", "uncertain")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=150, help="#positives and #negatives to judge")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--votes", type=int, default=1)
    args = ap.parse_args()
    _load_dotenv()

    huri = load_huri_graph()
    nodes = list(huri.g.nodes())
    node_set = set(nodes)
    huri_edges = {frozenset(e) for e in huri.g.edges()}
    biogrid, intact = _load_pairs(BIOGRID), _load_pairs(INTACT)
    db = biogrid | intact

    # POSITIVES: DB interactions that are HuRI non-edges, both nodes in HuRI
    hidden_pos = [tuple(p) for p in db
                  if p not in huri_edges and set(p) <= node_set and len(p) == 2]
    rng = np.random.default_rng(args.seed)
    rng.shuffle(hidden_pos)
    pos = hidden_pos[:args.k]

    # NEGATIVES: random HuRI non-edges absent from every DB
    neg, seen = [], set()
    while len(neg) < args.k and len(seen) < 200000:
        a, b = nodes[rng.integers(len(nodes))], nodes[rng.integers(len(nodes))]
        k = frozenset((a, b))
        if a != b and k not in huri_edges and k not in db and k not in seen:
            seen.add(k); neg.append((a, b))

    print("=" * 72)
    print("Judge as a hidden-positive detector — labeled test (independent of the veto)")
    print(f"seed={args.seed}  votes={args.votes}  positives={len(pos)}  negatives={len(neg)}")
    print(f"(hidden-positive pool available: {len(hidden_pos):,})")
    print("=" * 72)

    syms = ({l.split("\t")[0]: l.split("\t")[1].strip() for l in SYMS.open() if l.strip()}
            if SYMS.exists() else {})
    tg = TypedInteractionGraph.from_edges(
        [tuple(e) for e in huri.g.edges()], {n: "protein" for n in nodes},
        admissible_types=[("protein", "protein")], name="huri")
    lit = LiteratureFilter(enabled=True, provider="auto", votes=args.votes, names=syms)
    lit.fit(tg)

    pv = [_verdict(lit, tg, u, v) for u, v in pos]
    nv = [_verdict(lit, tg, u, v) for u, v in neg]

    def frac(verdicts, label):
        return sum(1 for x in verdicts if x == label) / len(verdicts) if verdicts else float("nan")

    print(f"\n  {'verdict':<28}{'on POSITIVES':>16}{'on NEGATIVES':>16}")
    print("  " + "-" * 58)
    for label in ("suspected_false_negative", "safe_negative", "uncertain", "skipped"):
        print(f"  {label:<28}{frac(pv, label):>15.1%}{frac(nv, label):>16.1%}")
    print("  " + "-" * 58)
    recall = frac(pv, "suspected_false_negative")
    fpr = frac(nv, "suspected_false_negative")
    print(f"\n  recall (flag | hidden positive) = {recall:.1%}")
    print(f"  FPR    (flag | true negative)   = {fpr:.1%}")
    if not (np.isnan(recall) or np.isnan(fpr)):
        print(f"  separation (recall − FPR)       = {recall - fpr:+.1%}")
        if recall - fpr > 0.05:
            print("  => the judge DISCRIMINATES: it flags real hidden interactions more")
            print("     than true negatives — with symbols it adds real signal.")
        else:
            print("  => no separation: the judge cannot tell hidden positives from")
            print("     negatives here; its flags are not enriched for real interactions.")


if __name__ == "__main__":
    main()
