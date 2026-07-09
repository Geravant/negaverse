"""Build a UniProt-accession -> Ensembl-gene (ENSG) map for the Negatome IDs.

Negatome is UniProt-keyed; HuRI is Ensembl-gene-keyed. To use Negatome gold
non-interactions as in-space test negatives for the HuRI benchmark we need this
map. We pull ENSG from UniProtKB JSON cross-references (the GeneId property),
batched over the UniProtKB `accessions` endpoint.

    python scripts/build_uniprot_ensembl_map.py

Writes local-docs/mappings/uniprot_to_ensembl.tsv (gitignored). Network required;
re-run only when the Negatome ID set changes (the cache is reused otherwise).
"""
from __future__ import annotations

import json
import ssl
import time
import urllib.parse
import urllib.request
from pathlib import Path

# UniProt is a public read-only endpoint; some Python builds (e.g. the macOS
# python.org framework) ship without a usable CA bundle. Prefer certifi if
# present, else fall back to an unverified context so the fetch still runs.
try:
    import certifi
    _SSL_CTX = ssl.create_default_context(cafile=certifi.where())
except Exception:
    _SSL_CTX = ssl._create_unverified_context()

from negaverse.io.negatome import load_negatome_pairs

OUT = Path("local-docs/mappings/uniprot_to_ensembl.tsv")
ENDPOINT = "https://rest.uniprot.org/uniprotkb/accessions"
CHUNK = 100


def _fetch(accessions: list[str]) -> dict[str, set[str]]:
    q = urllib.parse.urlencode({
        "accessions": ",".join(accessions),
        "fields": "accession,xref_ensembl",
        "format": "json",
    })
    req = urllib.request.Request(f"{ENDPOINT}?{q}", headers={"Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=30, context=_SSL_CTX) as fh:
        data = json.load(fh)
    out: dict[str, set[str]] = {}
    for entry in data.get("results", []):
        acc = entry.get("primaryAccession")
        genes: set[str] = set()
        for x in entry.get("uniProtKBCrossReferences", []):
            if x.get("database") != "Ensembl":
                continue
            for p in x.get("properties", []):
                if p.get("key") == "GeneId" and p.get("value"):
                    genes.add(p["value"].split(".")[0])
        if acc and genes:
            out[acc] = genes
    return out


def main() -> None:
    pairs = load_negatome_pairs()
    ids = sorted({x for pr in pairs for x in pr})
    print(f"{len(ids)} unique UniProt accessions to map")
    OUT.parent.mkdir(parents=True, exist_ok=True)

    mapping: dict[str, set[str]] = {}
    for i in range(0, len(ids), CHUNK):
        chunk = ids[i:i + CHUNK]
        for attempt in range(3):
            try:
                mapping.update(_fetch(chunk))
                break
            except Exception as e:  # transient network / rate limit
                print(f"  chunk {i//CHUNK}: retry {attempt+1} ({e})")
                time.sleep(2 * (attempt + 1))
        if (i // CHUNK) % 10 == 0:
            print(f"  {i+len(chunk)}/{len(ids)} done; mapped so far: {len(mapping)}")

    with open(OUT, "w") as fh:
        fh.write("uniprot\tensembl_genes\n")
        for acc in sorted(mapping):
            fh.write(f"{acc}\t{','.join(sorted(mapping[acc]))}\n")
    print(f"wrote {len(mapping)}/{len(ids)} mappings to {OUT}")


if __name__ == "__main__":
    main()
