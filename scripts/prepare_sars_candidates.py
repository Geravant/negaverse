"""Prepare the SARS-CoV-2 viral-host candidate space (open data).

The viral-host graph's nodes are viral gene *symbols* (nsp1..16, orf*, E/M/N/S)
and host UniProt accessions, with no sequences attached. This fetches sequences
for both sides from UniProtKB and writes, keyed by the graph's node ids:

    local-docs/sars/sequences.tsv   node_id <tab> sequence     (ESM2 build input)
    local-docs/sars/proteins.tsv    node_id <tab> side <tab> source_uniprot

Viral sequences: structural/accessory proteins have their own UniProt accessions;
the nsps are chains of the replicase polyproteins pp1ab (P0DTD1) and pp1a
(P0DTC1, for the short nsp11), sliced by their annotated chain coordinates.
Host sequences: fetched by accession. All from rest.uniprot.org (CC-BY 4.0).

Then build embeddings:
    PYTHONPATH=. python3 scripts/build_esm2_embeddings.py \
        --seqs local-docs/sars/sequences.tsv --out local-docs/sars/esm2_sars.npz

    PYTHONPATH=. python3 scripts/prepare_sars_candidates.py
"""
from __future__ import annotations

import json
import re
import ssl
import time
import urllib.parse
import urllib.request
from pathlib import Path

from negaverse.io import load_sars_cov2_graph

OUT = Path("local-docs/sars")
_SSL = ssl._create_unverified_context()
_UNIPROT = "https://rest.uniprot.org/uniprotkb"
_ACC = re.compile(r"^[OPQ][0-9][A-Z0-9]{3}[0-9]$|^[A-NR-Z][0-9]([A-Z][A-Z0-9]{2}[0-9]){1,2}$")

# Structural + accessory viral proteins → their own UniProt accessions.
VIRAL_ACC = {
    "S": "P0DTC2", "E": "P0DTC4", "M": "P0DTC5", "N": "P0DTC9",
    "orf3a": "P0DTC3", "orf6": "P0DTC6", "orf7a": "P0DTC7", "orf8": "P0DTC8",
    "orf9b": "P0DTD2", "orf10": "A0A663DJA2",
    "orf3b": "P0DTF1", "orf9c": "P0DTD3",   # small/contested ORFs — best-effort
}
POLYPROTEINS = ["P0DTD1", "P0DTC1"]         # pp1ab has nsp1-10,12-16; pp1a adds nsp11


def _get(url: str, timeout: float = 60) -> str:
    with urllib.request.urlopen(url, timeout=timeout, context=_SSL) as r:
        return r.read().decode()


def nsp_sequences() -> dict[int, str]:
    """{nsp_number: sequence} sliced from the replicase polyprotein chains."""
    nsp: dict[int, str] = {}
    for acc in POLYPROTEINS:
        d = json.loads(_get(f"{_UNIPROT}/{acc}.json?fields=sequence,ft_chain"))
        seq = d["sequence"]["value"]
        for f in d.get("features", []):
            if f["type"] != "Chain":
                continue
            desc = f.get("description", "")
            m = re.search(r"nsp\s*(\d+)", desc, re.I) or \
                re.search(r"Non-structural protein\s+(\d+)", desc, re.I)
            if not m:
                continue
            n = int(m.group(1))
            if n in nsp:                       # pp1ab wins; pp1a only fills gaps (nsp11)
                continue
            b, e = f["location"]["start"]["value"], f["location"]["end"]["value"]
            nsp[n] = seq[b - 1:e]
    return nsp


def viral_sequences(symbols: list[str]) -> dict[str, str]:
    nsp = nsp_sequences()
    accs = sorted({VIRAL_ACC[s] for s in symbols if s in VIRAL_ACC})
    acc_seq = fetch_by_accession(accs)
    out: dict[str, str] = {}
    for s in symbols:
        base = s.replace("_C145A", "")         # nsp5_C145A: catalytic mutant → nsp5 seq
        m = re.fullmatch(r"nsp(\d+)", base)
        seq = nsp.get(int(m.group(1))) if m else acc_seq.get(VIRAL_ACC.get(base, ""))
        if seq:
            out[s] = seq
    return out


def fetch_by_accession(accs: list[str], chunk: int = 100) -> dict[str, str]:
    """UniProt canonical FASTA for a list of accessions (chunked)."""
    seqs: dict[str, str] = {}
    for i in range(0, len(accs), chunk):
        part = accs[i:i + chunk]
        q = " OR ".join(f"accession:{a}" for a in part)
        url = f"{_UNIPROT}/search?" + urllib.parse.urlencode(
            {"query": q, "format": "fasta", "size": str(chunk)})
        acc, buf = None, []
        for line in _get(url, timeout=120).splitlines():
            if line.startswith(">"):
                if acc:
                    seqs[acc] = "".join(buf)
                acc = line.split("|")[1] if "|" in line else None
                buf = []
            else:
                buf.append(line.strip())
        if acc:
            seqs[acc] = "".join(buf)
        print(f"  hosts {min(i + chunk, len(accs))}/{len(accs)}")
        time.sleep(0.3)
    return seqs


def fetch_by_gene(syms: list[str]) -> dict[str, str]:
    """Resolve a few host node-ids that are gene *symbols* (loader fallback) to
    the reviewed human sequence, keyed by the symbol so it matches the graph."""
    out: dict[str, str] = {}
    for s in syms:
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", s) or s.upper() == "NA":
            continue                            # skip junk placeholders like 'NA'
        q = f"(gene_exact:{s}) AND (organism_id:9606) AND (reviewed:true)"
        url = f"{_UNIPROT}/search?" + urllib.parse.urlencode(
            {"query": q, "format": "fasta", "size": "1"})
        buf = []
        for line in _get(url, timeout=60).splitlines():
            if not line.startswith(">"):
                buf.append(line.strip())
        if buf:
            out[s] = "".join(buf)
        time.sleep(0.3)
    return out


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    g = load_sars_cov2_graph()
    viral = sorted(n for n in g.g.nodes() if g.node_type.get(n) == "viral")
    host = sorted(n for n in g.g.nodes() if g.node_type.get(n) == "host")
    print(f"graph: {len(viral)} viral, {len(host)} host")

    print("Fetching viral sequences (nsp chains + structural/ORF accessions) ...")
    vseq = viral_sequences(viral)
    print(f"  viral covered: {len(vseq)}/{len(viral)}  missing: {sorted(set(viral) - set(vseq))}")

    print("Fetching host sequences from UniProt ...")
    host_acc = [h for h in host if _ACC.match(h)]
    host_sym = [h for h in host if not _ACC.match(h)]      # loader fallback: raw gene symbols
    hseq = fetch_by_accession(host_acc)
    if host_sym:
        print(f"  resolving {len(host_sym)} host gene-symbol node-ids: {host_sym}")
        hseq.update(fetch_by_gene(host_sym))
    print(f"  host covered: {len(hseq)}/{len(host)}")

    rows = ([(s, "viral", VIRAL_ACC.get(s.replace('_C145A', ''), 'polyprotein_chain'), vseq[s])
             for s in viral if s in vseq]
            + [(h, "host", h, hseq[h]) for h in host if h in hseq])
    with open(OUT / "sequences.tsv", "w") as fh:
        for nid, _side, _src, seq in rows:
            fh.write(f"{nid}\t{seq}\n")
    with open(OUT / "proteins.tsv", "w") as fh:
        fh.write("node_id\tside\tsource_uniprot\tlength\n")
        for nid, side, src, seq in rows:
            fh.write(f"{nid}\t{side}\t{src}\t{len(seq)}\n")

    print(f"\nwrote {len(rows)} proteins -> {OUT}/sequences.tsv, proteins.tsv "
          f"({len(vseq)} viral + {len(hseq)} host)")
    print(f"\nnext: PYTHONPATH=. python3 scripts/build_esm2_embeddings.py "
          f"--seqs {OUT}/sequences.tsv --out {OUT}/esm2_sars.npz")


if __name__ == "__main__":
    main()
