"""Build an Ensembl-gene-id -> primary gene symbol map for human, so the LLM
judge (negaverse/streams/literature.py) can reason about HuRI's opaque ENSG
node ids instead of being blind to them.

HuRI graph nodes are ENSG ids; the judge needs human-readable symbols. We
already have local-docs/mappings/uniprot_ensg_human.tsv (UniProt accession ->
ENSG, from build_known_positive_sources.py). This script:

  1. reads every UniProt accession in that map,
  2. fetches each accession's PRIMARY gene name from UniProt REST in batches,
     with per-batch retry + exponential backoff (a failed batch is retried, not
     silently dropped — the earlier inline fetch dropped failures via a bare
     `else: continue`, which is why coverage stalled at 629/8245),
  3. inverts UniProt->symbol composed with UniProt->ENSG into ENSG->symbol,
  4. merges (not clobbers) into local-docs/mappings/ensg_symbol.tsv.

    PYTHONPATH=. python scripts/build_ensg_symbol_map.py [--limit N] [--batch 80]

gitignored output, same "prefer curated, don't fabricate" spirit as the rest of
the mapping scripts: an accession with no gene name is simply absent.
"""
from __future__ import annotations

import argparse
import json
import ssl
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

try:
    import certifi
    _SSL = ssl.create_default_context(cafile=certifi.where())
except Exception:
    _SSL = ssl._create_unverified_context()

MAP = Path("local-docs/mappings/uniprot_ensg_human.tsv")
OUT = Path("local-docs/mappings/ensg_symbol.tsv")
ENDPOINT = "https://rest.uniprot.org/uniprotkb/search"


def _load_uniprot_ensg() -> dict[str, list[str]]:
    """UniProt accession -> [ENSG, ...]."""
    m: dict[str, list[str]] = {}
    for line in MAP.read_text().splitlines():
        if line and not line.startswith("#"):
            parts = line.split("\t")
            if len(parts) >= 2:
                m[parts[0]] = parts[1].split(",")
    return m


def _fetch_primary(accs: list[str]) -> dict[str, str]:
    """accessions -> {accession: primary gene symbol}."""
    or_query = " OR ".join(f"accession:{a}" for a in accs)
    q = urllib.parse.urlencode({"query": or_query,
                                "fields": "accession,gene_primary",
                                "format": "json", "size": 500})
    req = urllib.request.Request(f"{ENDPOINT}?{q}", headers={"Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=60, context=_SSL) as fh:
        data = json.load(fh)
    out: dict[str, str] = {}
    for entry in data.get("results", []):
        acc = entry.get("primaryAccession")
        for gene in entry.get("genes", []) or []:
            name = (gene.get("geneName") or {}).get("value")
            if acc and name:
                out[acc] = name
                break
    return out


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="build_ensg_symbol_map")
    ap.add_argument("--limit", type=int, default=0, help="cap #accessions (0 = all)")
    ap.add_argument("--batch", type=int, default=80)
    ap.add_argument("--retries", type=int, default=5)
    args = ap.parse_args(argv)

    uni2ensg = _load_uniprot_ensg()
    accs = sorted(uni2ensg)
    if args.limit:
        accs = accs[:args.limit]
    print(f"{len(accs):,} UniProt accessions -> fetching primary gene symbols "
          f"(batch={args.batch}, retries={args.retries})", flush=True)

    uni2sym: dict[str, str] = {}
    failed_batches = 0
    n_batches = (len(accs) + args.batch - 1) // args.batch
    for bi, i in enumerate(range(0, len(accs), args.batch)):
        chunk = accs[i:i + args.batch]
        ok = False
        for attempt in range(args.retries):
            try:
                uni2sym.update(_fetch_primary(chunk))
                ok = True
                break
            except Exception as e:
                wait = min(30, 2 ** attempt)
                print(f"  batch {bi + 1}/{n_batches}: retry {attempt + 1}/{args.retries} "
                      f"in {wait}s ({type(e).__name__}: {e})", flush=True)
                time.sleep(wait)
        if not ok:
            failed_batches += 1
            print(f"  batch {bi + 1}/{n_batches}: GAVE UP after {args.retries} retries", flush=True)
        if (bi + 1) % 20 == 0:
            print(f"  ... {bi + 1}/{n_batches} batches, {len(uni2sym):,} symbols so far", flush=True)

    # compose UniProt->symbol with UniProt->ENSG  =>  ENSG->symbol
    ensg2sym: dict[str, str] = {}
    for acc, sym in uni2sym.items():
        for ensg in uni2ensg.get(acc, []):
            ensg2sym.setdefault(ensg, sym)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    merged: dict[str, str] = {}
    if OUT.exists():                                    # merge, don't clobber
        for line in OUT.read_text().splitlines():
            if line.strip() and not line.startswith("#"):
                k, v = line.split("\t")[:2]
                merged[k] = v
    before = len(merged)
    merged.update(ensg2sym)
    with open(OUT, "w") as fh:
        for k in sorted(merged):
            fh.write(f"{k}\t{merged[k]}\n")
    print(f"\nfetched {len(uni2sym):,} UniProt symbols; {len(ensg2sym):,} ENSG mapped; "
          f"{len(merged) - before:,} new; {len(merged):,} total in {OUT}", flush=True)
    if failed_batches:
        print(f"WARNING: {failed_batches} batch(es) gave up after retries — "
              f"re-run to fill gaps", file=sys.stderr, flush=True)


if __name__ == "__main__":
    main()
