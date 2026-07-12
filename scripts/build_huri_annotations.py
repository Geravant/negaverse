"""Produce HuRI-space (Ensembl-gene) annotations so the biology rules fire on the
benchmark graph — the #1 gap for validating "independent signals beat topology".

HuRI nodes are Ensembl gene ids (ENSG); our rules read `compartments` (GO
cellular-component) and `surface_hydrophobicity`. This maps HuRI's ENSG nodes to
UniProt via the human reviewed proteome, pulls GO CC, and MERGES:
  * ENSG -> compartments   into local-docs/localization/go_cc.tsv
  * ENSG -> hydrophobicity into local-docs/annotations/hydrophobicity.tsv
  * ENSG -> hydrophobicity tier into local-docs/annotations/hydrophobicity_tier.tsv
so build_annotation_table() picks them up automatically (co-localization is then an
independent, non-topology signal on HuRI).

Hydrophobicity is computed via scripts/compute_surface_hydrophobicity.py's real
two-tier method (DSSP solvent-exposure + AlphaFold pLDDT disorder-masking when a
confident structure exists, sequence-mean fallback otherwise) — the exact method
`hydrophobicity_interface`'s 0.44 threshold was calibrated against
(scripts/calibrate_hydrophobicity_threshold.py). An earlier version of this script
computed its own separate, simplified whole-sequence Kyte-Doolittle score inline,
which doesn't match what the threshold means; fixed to call the real `compute()`
instead of duplicating a weaker proxy.

    PYTHONPATH=. python scripts/build_huri_annotations.py

Network required (UniProt REST, paginated, plus AlphaFold structure fetch + DSSP
for the Tier 1 upgrade — see compute_surface_hydrophobicity.py). ENSG keys never
collide with the UniProt keys already in those files, so SARS annotations are
preserved.
"""
from __future__ import annotations

import json
import re
import ssl
import time
import urllib.parse
import urllib.request
from pathlib import Path

from scripts.compute_surface_hydrophobicity import compute as compute_hydrophobicity

try:
    import certifi
    _SSL = ssl.create_default_context(cafile=certifi.where())
except Exception:
    _SSL = ssl._create_unverified_context()

LOC = Path("local-docs/localization/go_cc.tsv")
HYD = Path("local-docs/annotations/hydrophobicity.tsv")
HYD_TIER = Path("local-docs/annotations/hydrophobicity_tier.tsv")
SEARCH = "https://rest.uniprot.org/uniprotkb/search"


def _get(url):
    req = urllib.request.Request(url, headers={"Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=60, context=_SSL) as fh:
        link = fh.headers.get("Link", "")
        data = json.load(fh)
    # the next-page URL contains commas (fields=a,b,c), so match the whole <...>
    m = re.search(r'<([^>]+)>;\s*rel="next"', link)
    return data, (m.group(1) if m else None)


def _iter_proteome():
    url = SEARCH + "?" + urllib.parse.urlencode({
        "query": "organism_id:9606 AND reviewed:true",
        "fields": "accession,xref_ensembl,go_c", "format": "json", "size": "500"})
    page = 0
    while url:
        for attempt in range(3):
            try:
                data, url = _get(url)
                break
            except Exception as e:
                print(f"  page {page}: retry {attempt+1} ({e})")
                time.sleep(2 * (attempt + 1))
        else:
            break
        for e in data.get("results", []):
            yield e
        page += 1
        if page % 10 == 0:
            print(f"  proteome page {page} ...")


def _merge(path: Path, new: dict, joiner):
    path.parent.mkdir(parents=True, exist_ok=True)
    cur = {}
    if path.exists():
        for line in path.read_text().splitlines():
            if line.strip() and not line.startswith("#"):
                k, v = line.split("\t")[:2]
                cur[k] = v
    for k, v in new.items():
        cur[k] = joiner(v)
    path.write_text("".join(f"{k}\t{cur[k]}\n" for k in sorted(cur)))
    return len(new), len(cur)


def main():
    from negaverse.io import load_huri_graph
    huri = set(load_huri_graph().g.nodes())
    print(f"HuRI ENSG nodes: {len(huri):,}")

    ensg_comp: dict[str, set] = {}
    ensg_uni: dict[str, str] = {}
    print("Scanning human reviewed proteome (UniProt) ...")
    for e in _iter_proteome():
        acc = e.get("primaryAccession")
        ensgs, comps = set(), set()
        for x in e.get("uniProtKBCrossReferences", []):
            db = x.get("database")
            if db == "Ensembl":
                for p in x.get("properties", []):
                    if p.get("key") == "GeneId" and p.get("value"):
                        ensgs.add(p["value"].split(".")[0])
            elif db == "GO":
                for p in x.get("properties", []):
                    if p.get("key") == "GoTerm" and p.get("value", "").startswith("C:"):
                        comps.add(p["value"][2:].strip().lower())
        for g in ensgs & huri:
            if comps:
                ensg_comp.setdefault(g, set()).update(comps)
            ensg_uni.setdefault(g, acc)
    print(f"  mapped {len(ensg_uni):,} HuRI genes to UniProt; "
          f"{len(ensg_comp):,} have GO compartments")

    # hydrophobicity for the mapped UniProts — real two-tier method (DSSP +
    # AlphaFold structure when available, sequence fallback otherwise), the
    # same one hydrophobicity_interface's threshold was calibrated against.
    print("Scoring hydrophobicity (two-tier: structure when available, sequence fallback) ...")
    uni_list = sorted(set(ensg_uni.values()))
    scores, tiers = compute_hydrophobicity(uni_list)
    ensg_hyd = {}
    ensg_tier = {}
    for g, acc in ensg_uni.items():
        if acc in scores:
            ensg_hyd[g] = scores[acc]
            ensg_tier[g] = tiers[acc]

    n1, t1 = _merge(LOC, ensg_comp, lambda s: ",".join(sorted(s)))
    n2, t2 = _merge(HYD, ensg_hyd, lambda v: str(v))
    n3, t3 = _merge(HYD_TIER, ensg_tier, lambda v: v)
    print(f"\nmerged {n1} HuRI compartment rows into {LOC} (total {t1})")
    print(f"merged {n2} HuRI hydrophobicity rows into {HYD} (total {t2})")
    print(f"merged {n3} HuRI hydrophobicity-tier rows into {HYD_TIER} (total {t3})")


if __name__ == "__main__":
    main()
