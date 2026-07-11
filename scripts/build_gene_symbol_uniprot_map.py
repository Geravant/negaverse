"""Build a gene-symbol -> UniProt accession mapping for human (taxon 9606),
for use by scripts/calibrate_hydrophobicity_threshold.py to score UPNA-PPI
(gene-symbol keyed) alongside DRYAD (already UniProt-keyed).

Note: UniProt's async ID-mapping API (POST /idmapping/run) is documented to
support `from=Gene_Name` -> `to=UniProtKB`, but returns empty results for it
in practice (verified live, several attempts, including a single well-known
symbol with no special characters). This uses the plain search endpoint
instead (verified live, works reliably), batching gene symbols into OR
queries against `gene:<symbol> ... AND organism_id:9606 AND reviewed:true`.

Only the primary gene name is matched (not synonyms), and only reviewed
(Swiss-Prot) entries — same "don't fabricate, prefer curated" spirit as the
rest of this pipeline. A symbol with no reviewed human match is simply absent
from the output, not guessed.

    PYTHONPATH=. python scripts/build_gene_symbol_uniprot_map.py --ids-file gene_symbols.txt

Writes local-docs/mappings/gene_symbol_to_uniprot.tsv (gene_symbol<TAB>uniprot),
gitignored, merge-not-clobber like every other annotation/mapping script here.
"""
from __future__ import annotations

import argparse
import json
import ssl
import time
import urllib.parse
import urllib.request
from pathlib import Path

try:
    import certifi
    _SSL = ssl.create_default_context(cafile=certifi.where())
except Exception:
    _SSL = ssl._create_unverified_context()

OUT = Path("local-docs/mappings/gene_symbol_to_uniprot.tsv")
ENDPOINT = "https://rest.uniprot.org/uniprotkb/search"
BATCH = 50


def _fetch(symbols: list[str]) -> dict[str, str]:
    """symbols -> {original-casing symbol: uniprot accession}, matched
    case-insensitively against UniProt's primary gene name only."""
    or_query = " OR ".join(f"gene:{s}" for s in symbols)
    query = f"({or_query}) AND organism_id:9606 AND reviewed:true"
    q = urllib.parse.urlencode({"query": query, "fields": "accession,gene_names",
                                "format": "json", "size": 500})
    req = urllib.request.Request(f"{ENDPOINT}?{q}", headers={"Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=30, context=_SSL) as fh:
        data = json.load(fh)
    by_upper = {s.upper(): s for s in symbols}     # preserve caller's exact casing in output
    out: dict[str, str] = {}
    for entry in data.get("results", []):
        acc = entry.get("primaryAccession")
        for gene in entry.get("genes", []) or []:
            name = (gene.get("geneName") or {}).get("value")
            if not name:
                continue
            orig = by_upper.get(name.upper())
            if orig and orig not in out:
                out[orig] = acc
    return out


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="build_gene_symbol_uniprot_map")
    ap.add_argument("--ids-file", required=True)
    args = ap.parse_args(argv)
    symbols = sorted(set(l.strip() for l in Path(args.ids_file).read_text().splitlines() if l.strip()))
    print(f"{len(symbols)} gene symbols")

    mapping: dict[str, str] = {}
    for i in range(0, len(symbols), BATCH):
        chunk = symbols[i:i + BATCH]
        for attempt in range(3):
            try:
                mapping.update(_fetch(chunk))
                break
            except Exception as e:
                print(f"  chunk {i // BATCH}: retry {attempt + 1} ({e})")
                time.sleep(2 * (attempt + 1))

    OUT.parent.mkdir(parents=True, exist_ok=True)
    merged: dict[str, str] = {}
    if OUT.exists():                                    # merge, don't clobber
        for line in OUT.read_text().splitlines():
            if line.strip() and not line.startswith("#"):
                k, v = line.split("\t")[:2]
                merged[k] = v
    merged.update(mapping)
    with open(OUT, "w") as fh:
        for k in sorted(merged):
            fh.write(f"{k}\t{merged[k]}\n")
    print(f"mapped {len(mapping)}/{len(symbols)} new; {len(merged)} total in {OUT}")


if __name__ == "__main__":
    main()
