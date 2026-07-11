"""HuRI-space structure-aware surface hydrophobicity for Lucy's
`hydrophobicity_interface` rule.

Lucy's scripts/compute_surface_hydrophobicity.py keys output by UniProt accession
(AlphaFold is UniProt-keyed), but HuRI graph nodes are Ensembl gene ids (ENSG).
This wrapper bridges that:
  1. map HuRI's ENSG nodes -> UniProt via the cached UniProt<->ENSG map
     (local-docs/mappings/uniprot_ensg_human.tsv, from build_known_positive_sources.py),
  2. run Lucy's structure-aware compute() over those UniProt accessions
     (AlphaFold fetch + DSSP; Tier-2 sequence fallback for low-pLDDT models),
  3. write BOTH the UniProt-keyed and the re-keyed ENSG-keyed scores into
     local-docs/annotations/hydrophobicity.tsv (merge), so the rule fires on HuRI.

    PYTHONPATH=. python scripts/build_huri_surface_hydrophobicity.py [--limit N]

WARNING: full HuRI is ~8k proteins -> ~8k AlphaFold downloads + DSSP runs
(multi-hour, several GB under local-docs/alphafold/). Use --limit to bound it.
"""
from __future__ import annotations

import argparse
from pathlib import Path

MAP = Path("local-docs/mappings/uniprot_ensg_human.tsv")


def _ensg_to_uniprot() -> dict[str, str]:
    """ENSG -> one UniProt accession (first seen), inverted from the UniProt->ENSG map."""
    ensg2uni: dict[str, str] = {}
    for line in MAP.read_text().splitlines():
        if line and not line.startswith("#"):
            u, es = line.split("\t")[:2]
            for e in es.split(","):
                ensg2uni.setdefault(e, u)
    return ensg2uni


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="cap #proteins (0 = all HuRI)")
    args = ap.parse_args()

    from negaverse.io import load_huri_graph
    from scripts.compute_surface_hydrophobicity import compute, _write_merged, OUT, TIER_OUT

    huri = list(load_huri_graph().g.nodes())
    ensg2uni = _ensg_to_uniprot()
    ensg_with_uni = [(e, ensg2uni[e]) for e in huri if e in ensg2uni]
    if args.limit:
        ensg_with_uni = ensg_with_uni[:args.limit]
    accs = sorted({u for _, u in ensg_with_uni})
    print(f"HuRI genes: {len(huri):,}; mapped to UniProt: {len(ensg_with_uni):,} "
          f"({len(accs):,} unique accessions)")

    scores, tiers = compute(accs)                       # AlphaFold + DSSP (the heavy part)

    # UniProt-keyed (for UniProt graphs) + re-keyed to ENSG (for HuRI)
    out_scores = {u: str(v) for u, v in scores.items()}
    out_tiers = dict(tiers)
    for e, u in ensg_with_uni:
        if u in scores:
            out_scores[e] = str(scores[u])
            out_tiers[e] = tiers.get(u, "sequence")
    n = _write_merged(OUT, out_scores)
    _write_merged(TIER_OUT, out_tiers)
    struct = sum(1 for e, u in ensg_with_uni if tiers.get(u) == "structure")
    print(f"\nwrote {n} new scores to {OUT}; {struct:,}/{len(ensg_with_uni):,} HuRI genes "
          f"got a structure-based (Tier 1) value")


if __name__ == "__main__":
    main()
