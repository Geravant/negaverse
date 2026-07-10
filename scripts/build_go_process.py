"""Produce GO biological-process annotations so a functional-compatibility rule
can fire — an independent (non-topology, non-localization) biology signal.

Parses the GO Consortium's human GAF (goa_human.gaf.gz) — a stable bulk file,
independent of the flaky UniProt REST API — and writes:
  * ENSG   -> processes   (for the HuRI benchmark graph)
  * UniProt-> processes   (for the SARS-CoV-2 host graph)
into local-docs/annotations/go_bp.tsv, keyed by node id (ENSG and UniProt keys
never collide), so build_annotation_table() picks it up as the `processes` field.

ENSG mapping reuses the cached UniProt->ENSG map from
scripts/build_known_positive_sources.py (local-docs/mappings/uniprot_ensg_human.tsv).

    # get the GAF once (11 MB):
    curl -sL -o local-docs/go/goa_human.gaf.gz \
      https://ftp.ebi.ac.uk/pub/databases/GO/goa/HUMAN/goa_human.gaf.gz
    PYTHONPATH=. python scripts/build_go_process.py

GAF 2.x columns (tab-separated): col2 = UniProt accession, col5 = GO id,
col9 = aspect (P/F/C). We keep aspect == 'P' (biological_process).
"""
from __future__ import annotations

import gzip
from pathlib import Path

GAF = Path("local-docs/go/goa_human.gaf.gz")
MAP = Path("local-docs/mappings/uniprot_ensg_human.tsv")
OUT = Path("local-docs/annotations/go_bp.tsv")


def _load_uni2ensg() -> dict[str, set[str]]:
    m: dict[str, set[str]] = {}
    if not MAP.exists():
        print(f"  (no {MAP}; ENSG rows skipped — run build_known_positive_sources.py)")
        return m
    for line in MAP.read_text().splitlines():
        if line and not line.startswith("#"):
            u, e = line.split("\t")[:2]
            m.setdefault(u, set()).update(e.split(","))
    return m


def _iter_gaf(path: Path):
    with gzip.open(path, "rt", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            if line.startswith("!"):
                continue
            c = line.rstrip("\n").split("\t")
            if len(c) < 9 or c[8] != "P":          # aspect P = biological_process
                continue
            acc = c[1].split("-")[0]               # strip isoform suffix
            go = c[4]                              # GO:0008150
            if acc and go:
                yield acc, go


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
    path.write_text("# node<TAB>comma-separated GO biological_process ids\n"
                    + "".join(f"{k}\t{','.join(sorted(cur[k]))}\n" for k in sorted(cur)))
    return len(new), len(cur)


def main():
    if not GAF.exists():
        raise SystemExit(f"missing {GAF}; download it first (see this file's docstring)")
    from negaverse.io import load_huri_graph
    huri = set(load_huri_graph().g.nodes())
    uni2ensg = _load_uni2ensg()
    print(f"HuRI ENSG nodes: {len(huri):,}; UniProt->ENSG map: {len(uni2ensg):,} accessions")

    uni_proc: dict[str, set] = {}
    for acc, go in _iter_gaf(GAF):
        uni_proc.setdefault(acc, set()).add(go)
    print(f"  parsed GAF: {len(uni_proc):,} human accessions with a biological_process term")

    out: dict[str, set] = {}
    n_ensg = 0
    for acc, procs in uni_proc.items():
        out.setdefault(acc, set()).update(procs)             # UniProt key -> SARS host graph
        for g in uni2ensg.get(acc, ()):                      # ENSG key -> HuRI benchmark
            if g in huri:
                out.setdefault(g, set()).update(procs)
                n_ensg += 1

    n, tot = _merge(OUT, out)
    print(f"\nmerged {n} rows into {OUT} (total {tot}); {n_ensg} HuRI-gene rows written")


if __name__ == "__main__":
    main()
