"""Build an Ensembl-gene-id -> gene symbol / synonyms / full name map for
human, so the LLM judge (negaverse/streams/literature.py) can reason about
HuRI's opaque ENSG node ids by name instead of being blind to them.

HuRI graph nodes are ENSG ids; the judge needs human-readable identity, not
just a bare symbol — a symbol alone ("TP53") is often already known to the
model, but the full protein name ("Cellular tumor antigen p53") and known
synonyms ("P53", "TRP53") give it more to recognize the protein by and reason
about its function. We already have
local-docs/mappings/uniprot_ensg_human.tsv (UniProt accession -> ENSG, from
build_known_positive_sources.py). This script:

  1. reads every UniProt accession in that map,
  2. fetches each accession's primary gene symbol, gene synonyms, and
     recommended full protein name from UniProt REST in batches, with
     per-batch retry + exponential backoff (a failed batch is retried, not
     silently dropped — the earlier inline fetch dropped failures via a bare
     `else: continue`, which is why coverage stalled at 629/8245),
  3. inverts UniProt->info composed with UniProt->ENSG into ENSG->info,
  4. merges (not clobbers) into local-docs/mappings/ensg_symbol.tsv.

    PYTHONPATH=. python scripts/build_ensg_symbol_map.py [--limit N] [--batch 80]

Writes (gitignored, "prefer curated, don't fabricate" spirit as the rest of
the mapping scripts: an accession with no gene name is simply absent):
    local-docs/mappings/ensg_symbol.tsv
        ensg<TAB>symbol<TAB>synonyms(comma-separated, may be empty)<TAB>full_name
"""
from __future__ import annotations

import argparse
import json
import ssl
import sys
import time
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path

try:
    import certifi
    _SSL = ssl.create_default_context(cafile=certifi.where())
except Exception:
    _SSL = ssl._create_unverified_context()

MAP = Path("local-docs/mappings/uniprot_ensg_human.tsv")
OUT = Path("local-docs/mappings/ensg_symbol.tsv")
ENDPOINT = "https://rest.uniprot.org/uniprotkb/search"


@dataclass
class GeneInfo:
    symbol: str
    synonyms: list[str]
    full_name: str = ""


def _load_uniprot_ensg() -> dict[str, list[str]]:
    """UniProt accession -> [ENSG, ...]."""
    m: dict[str, list[str]] = {}
    for line in MAP.read_text().splitlines():
        if line and not line.startswith("#"):
            parts = line.split("\t")
            if len(parts) >= 2:
                m[parts[0]] = parts[1].split(",")
    return m


def _fetch_info(accs: list[str]) -> dict[str, GeneInfo]:
    """accessions -> {accession: GeneInfo}. Needs the primary gene symbol to
    be useful at all; synonyms/full_name are added when present, not required."""
    or_query = " OR ".join(f"accession:{a}" for a in accs)
    q = urllib.parse.urlencode({
        "query": or_query,
        "fields": "accession,gene_primary,gene_synonym,protein_name",
        "format": "json", "size": 500})
    req = urllib.request.Request(f"{ENDPOINT}?{q}", headers={"Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=60, context=_SSL) as fh:
        data = json.load(fh)
    out: dict[str, GeneInfo] = {}
    for entry in data.get("results", []):
        acc = entry.get("primaryAccession")
        if not acc:
            continue
        genes = entry.get("genes") or []
        gene = genes[0] if genes else {}
        symbol = (gene.get("geneName") or {}).get("value")
        if not symbol:
            continue
        synonyms = [s["value"] for s in (gene.get("synonyms") or []) if s.get("value")]
        full_name = ((entry.get("proteinDescription") or {}).get("recommendedName") or {}) \
            .get("fullName", {}).get("value", "")
        out[acc] = GeneInfo(symbol=symbol, synonyms=synonyms, full_name=full_name)
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
    print(f"{len(accs):,} UniProt accessions -> fetching gene symbol/synonyms/full name "
          f"(batch={args.batch}, retries={args.retries})", flush=True)

    uni2info: dict[str, GeneInfo] = {}
    failed_batches = 0
    n_batches = (len(accs) + args.batch - 1) // args.batch
    for bi, i in enumerate(range(0, len(accs), args.batch)):
        chunk = accs[i:i + args.batch]
        ok = False
        for attempt in range(args.retries):
            try:
                uni2info.update(_fetch_info(chunk))
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
            print(f"  ... {bi + 1}/{n_batches} batches, {len(uni2info):,} entries so far", flush=True)

    # compose UniProt->info with UniProt->ENSG  =>  ENSG->info
    ensg2info: dict[str, GeneInfo] = {}
    for acc, info in uni2info.items():
        for ensg in uni2ensg.get(acc, []):
            ensg2info.setdefault(ensg, info)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    merged: dict[str, GeneInfo] = {}
    if OUT.exists():                                    # merge, don't clobber
        for line in OUT.read_text().splitlines():
            if not line.strip() or line.startswith("#"):
                continue
            parts = line.split("\t")
            ensg, sym = parts[0], parts[1]
            syns = parts[2].split(",") if len(parts) > 2 and parts[2] else []
            full = parts[3] if len(parts) > 3 else ""
            merged[ensg] = GeneInfo(symbol=sym, synonyms=syns, full_name=full)
    before = len(merged)
    merged.update(ensg2info)
    with open(OUT, "w") as fh:
        for k in sorted(merged):
            info = merged[k]
            fh.write(f"{k}\t{info.symbol}\t{','.join(info.synonyms)}\t{info.full_name}\n")
    print(f"\nfetched {len(uni2info):,} UniProt entries; {len(ensg2info):,} ENSG mapped; "
          f"{len(merged) - before:,} new; {len(merged):,} total in {OUT}", flush=True)
    if failed_batches:
        print(f"WARNING: {failed_batches} batch(es) gave up after retries — "
              f"re-run to fill gaps", file=sys.stderr, flush=True)


if __name__ == "__main__":
    main()
