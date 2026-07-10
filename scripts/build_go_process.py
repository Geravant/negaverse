"""Produce GO biological-process annotations so a functional-compatibility rule
can fire — an independent (non-topology, non-localization) biology signal.

Mirrors scripts/build_huri_annotations.py but pulls GO **biological_process**
(the `P:` terms) instead of cellular-component. Scans the human reviewed
proteome once and writes:
  * ENSG   -> processes   (for the HuRI benchmark graph, restricted to its nodes)
  * UniProt-> processes   (for the SARS-CoV-2 host graph)
into local-docs/annotations/go_bp.tsv, keyed by node id (ENSG and UniProt keys
never collide), so build_annotation_table() picks it up as the `processes` field.

    PYTHONPATH=. python scripts/build_go_process.py

Network required (UniProt REST, paginated ~40 pages); merges, doesn't clobber.
"""
from __future__ import annotations

import json
import re
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

OUT = Path("local-docs/annotations/go_bp.tsv")
SEARCH = "https://rest.uniprot.org/uniprotkb/search"


def _get(url):
    req = urllib.request.Request(url, headers={"Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=60, context=_SSL) as fh:
        link = fh.headers.get("Link", "")
        data = json.load(fh)
    m = re.search(r'<([^>]+)>;\s*rel="next"', link)
    return data, (m.group(1) if m else None)


def _iter_proteome():
    url = SEARCH + "?" + urllib.parse.urlencode({
        "query": "organism_id:9606 AND reviewed:true",
        "fields": "accession,xref_ensembl,go_p", "format": "json", "size": "500"})
    page = 0
    while url:
        for attempt in range(4):
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


def _merge(path: Path, new: dict[str, set]):
    path.parent.mkdir(parents=True, exist_ok=True)
    cur: dict[str, set] = {}
    if path.exists():
        for line in path.read_text().splitlines():
            if line.strip() and not line.startswith("#"):
                k, v = line.split("\t")[:2]
                cur[k] = set(v.split(","))
    for k, v in new.items():
        cur.setdefault(k, set()).update(v)
    path.write_text("# node<TAB>comma-separated GO biological_process terms\n"
                    + "".join(f"{k}\t{','.join(sorted(cur[k]))}\n" for k in sorted(cur)))
    return len(new), len(cur)


def main():
    from negaverse.io import load_huri_graph
    huri = set(load_huri_graph().g.nodes())
    print(f"HuRI ENSG nodes: {len(huri):,}")

    out: dict[str, set] = {}
    n_ensg = 0
    print("Scanning human reviewed proteome (UniProt) for GO biological_process ...")
    for e in _iter_proteome():
        acc = e.get("primaryAccession")
        ensgs, procs = set(), set()
        for x in e.get("uniProtKBCrossReferences", []):
            db = x.get("database")
            if db == "Ensembl":
                for p in x.get("properties", []):
                    if p.get("key") == "GeneId" and p.get("value"):
                        ensgs.add(p["value"].split(".")[0])
            elif db == "GO":
                for p in x.get("properties", []):
                    if p.get("key") == "GoTerm" and p.get("value", "").startswith("P:"):
                        procs.add(p["value"][2:].strip().lower())
        if not procs:
            continue
        if acc:                                    # UniProt key -> SARS host graph
            out.setdefault(acc, set()).update(procs)
        for g in ensgs & huri:                     # ENSG key -> HuRI benchmark
            out.setdefault(g, set()).update(procs)
            n_ensg += 1

    n, tot = _merge(OUT, out)
    print(f"\nmerged {n} rows into {OUT} (total {tot}); {n_ensg} HuRI-gene rows")


if __name__ == "__main__":
    main()
