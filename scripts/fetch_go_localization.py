"""Fetch GO cellular-component annotations for a set of UniProt accessions and
write the TSV the co-localization filter consumes.

    # UniProt-looking nodes of the SARS-CoV-2 graph (host proteins):
    PYTHONPATH=. python scripts/fetch_go_localization.py --dataset sars
    # or an explicit id list, one UniProt accession per line:
    PYTHONPATH=. python scripts/fetch_go_localization.py --ids-file my_uniprot_ids.txt

Writes local-docs/localization/go_cc.tsv (gitignored):  node<TAB>term1,term2,...
Node IDs are the UniProt accessions themselves; to annotate an Ensembl-keyed graph
(HuRI) first map genes -> UniProt (see scripts/build_uniprot_ensembl_map.py) and
feed the accessions via --ids-file, then relabel — deferred.
"""
from __future__ import annotations

import argparse
import json
import re
import ssl
import time
import urllib.parse
import urllib.request
from pathlib import Path

try:
    import certifi
    _SSL_CTX = ssl.create_default_context(cafile=certifi.where())
except Exception:
    _SSL_CTX = ssl._create_unverified_context()

OUT = Path("local-docs/localization/go_cc.tsv")
ENDPOINT = "https://rest.uniprot.org/uniprotkb/accessions"
CHUNK = 100
_UNIPROT_RE = re.compile(r"^[OPQ][0-9][A-Z0-9]{3}[0-9]|^[A-NR-Z][0-9]([A-Z][A-Z0-9]{2}[0-9]){1,2}$")
_TERM_RE = re.compile(r"\s*\[GO:\d+\]\s*")


def _fetch(accessions: list[str]) -> dict[str, set[str]]:
    q = urllib.parse.urlencode({
        "accessions": ",".join(accessions),
        "fields": "accession,go_c",
        "format": "json",
    })
    req = urllib.request.Request(f"{ENDPOINT}?{q}", headers={"Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=30, context=_SSL_CTX) as fh:
        data = json.load(fh)
    out: dict[str, set[str]] = {}
    for entry in data.get("results", []):
        acc = entry.get("primaryAccession")
        comps: set[str] = set()
        for ref in entry.get("uniProtKBCrossReferences", []):
            if ref.get("database") != "GO":
                continue
            for p in ref.get("properties", []):
                # GoTerm looks like "C:cytoplasm"; keep cellular-component (C:) only
                if p.get("key") == "GoTerm" and p.get("value", "").startswith("C:"):
                    comps.add(p["value"][2:].strip().lower())
        if acc and comps:
            out[acc] = comps
    return out


def _sars_ids() -> list[str]:
    from negaverse.io import load_sars_cov2_graph
    g = load_sars_cov2_graph()
    return [n for n in g.g.nodes() if _UNIPROT_RE.match(n)]


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="fetch_go_localization")
    ap.add_argument("--dataset", choices=["sars"], help="derive UniProt IDs from a bundled graph")
    ap.add_argument("--ids-file", help="one UniProt accession per line")
    args = ap.parse_args(argv)

    if args.ids_file:
        ids = [l.strip() for l in Path(args.ids_file).read_text().splitlines() if l.strip()]
    elif args.dataset == "sars":
        ids = _sars_ids()
    else:
        ap.error("give --dataset or --ids-file")
    ids = sorted(set(ids))
    print(f"{len(ids)} UniProt accessions to annotate")

    mapping: dict[str, set[str]] = {}
    for i in range(0, len(ids), CHUNK):
        chunk = ids[i:i + CHUNK]
        for attempt in range(3):
            try:
                mapping.update(_fetch(chunk))
                break
            except Exception as e:
                print(f"  chunk {i//CHUNK}: retry {attempt+1} ({e})")
                time.sleep(2 * (attempt + 1))

    OUT.parent.mkdir(parents=True, exist_ok=True)
    merged: dict[str, set[str]] = {}
    if OUT.exists():                                    # merge, don't clobber other datasets
        for line in OUT.read_text().splitlines():
            if line.strip() and not line.startswith("#"):
                node, comps = line.split("\t")[:2]
                merged[node] = {c.strip() for c in comps.split(",") if c.strip()}
    merged.update(mapping)
    with open(OUT, "w") as fh:
        for node in sorted(merged):
            fh.write(f"{node}\t{','.join(sorted(merged[node]))}\n")
    print(f"wrote {len(mapping)} new / {len(merged)} total annotations to {OUT}")


if __name__ == "__main__":
    main()
