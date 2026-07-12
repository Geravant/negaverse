"""Build external known-positive PPI source files for KnownPositiveVeto.

Turns raw BioGRID + IntAct human downloads, and STRING's own already-downloaded
data, into the 2-column pair files that `rules/sources.yaml` declares, so the
veto stops treating those sources as "missing" and actually removes documented
interactions from the negative pool.

Two ID spaces are emitted per source, because our two PPI graphs use different
node ids:
  * UniProt space  -> engages the SARS-CoV-2 graph (host nodes are UniProt)
  * Ensembl-gene   -> engages the HuRI benchmark (nodes are ENSG)

Inputs (place first, all gitignored under local-docs/):
  * local-docs/biogrid/BIOGRID-ORGANISM-LATEST.tab3.zip   (all-organism tab3 zip)
  * local-docs/intact/human.zip                            (IntAct species/human.zip)
  * local-docs/string/9606.protein.links.detailed.v12.0.txt.gz +
    local-docs/string/9606.protein.aliases.v12.0.txt.gz     (STRING v12.0, human)

Outputs:
  * local-docs/biogrid/biogrid_human_pairs.tsv         (uniprot)
  * local-docs/intact/intact_human_pairs.tsv           (uniprot)
  * local-docs/string/string_experimental_high_confidence_pairs.tsv (uniprot)
  * local-docs/biogrid/biogrid_human_pairs_ensembl.tsv (ensembl)
  * local-docs/intact/intact_human_pairs_ensembl.tsv   (ensembl)
  * local-docs/string/string_experimental_high_confidence_pairs_ensembl.tsv (ensembl)
  * local-docs/mappings/uniprot_ensg_human.tsv         (cached UniProt->ENSG map)

    PYTHONPATH=. python scripts/build_known_positive_sources.py

Ensembl mapping needs the network (UniProt REST, paginated ~40 pages); it is
cached, so re-runs are instant. Pass --no-ensembl to skip it (UniProt files only).
STRING's own files are already local (no network needed for that source).
"""
from __future__ import annotations

import argparse
import gzip
import io
import json
import re
import ssl
import subprocess
import sys
import time
import urllib.parse
import urllib.request
import zipfile
from pathlib import Path

try:
    import certifi
    _SSL = ssl.create_default_context(cafile=certifi.where())
except Exception:
    _SSL = ssl._create_unverified_context()

ROOT = Path(__file__).resolve().parent.parent
BIOGRID_ZIP = ROOT / "local-docs/biogrid/BIOGRID-ORGANISM-LATEST.tab3.zip"
INTACT_ZIP = ROOT / "local-docs/intact/human.zip"
STRING_ALIASES = ROOT / "local-docs/string/9606.protein.aliases.v12.0.txt.gz"
STRING_LINKS = ROOT / "local-docs/string/9606.protein.links.detailed.v12.0.txt.gz"
BIOGRID_OUT = ROOT / "local-docs/biogrid/biogrid_human_pairs.tsv"
INTACT_OUT = ROOT / "local-docs/intact/intact_human_pairs.tsv"
STRING_OUT = ROOT / "local-docs/string/string_experimental_high_confidence_pairs.tsv"
STRING_EXPERIMENTAL_THRESHOLD = 900   # raw STRING units (0-1000) == 0.9 rescaled
MAP_CACHE = ROOT / "local-docs/mappings/uniprot_ensg_human.tsv"
SEARCH = "https://rest.uniprot.org/uniprotkb/search"

# UniProt accession syntax (canonical, no isoform suffix)
_UP = re.compile(r"^[OPQ][0-9][A-Z0-9]{3}[0-9]$|^[A-NR-Z][0-9]([A-Z][A-Z0-9]{2}[0-9]){1,2}$")


def _clean_uniprot(tok: str) -> str | None:
    """Normalise an accession token: strip isoform suffix, validate syntax."""
    tok = tok.strip().split("-")[0]  # P12345-2 -> P12345
    return tok if _UP.match(tok) else None


def _write_pairs(path: Path, pairs: set[frozenset], header: str) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as fh:
        fh.write(f"# {header}\n")
        for p in sorted(map(sorted, pairs)):
            fh.write(f"{p[0]}\t{p[1]}\n")
    return len(pairs)


# --------------------------------------------------------------------------- #
# BioGRID: tab3, human-human, physical only. Cols (1-indexed):
#   13 Experimental System Type | 16/17 Organism ID | 24/27 SWISS-PROT A/B
# --------------------------------------------------------------------------- #
def build_biogrid() -> set[frozenset]:
    if not BIOGRID_ZIP.exists():
        print(f"  [biogrid] SKIP — {BIOGRID_ZIP} not found")
        return set()
    with zipfile.ZipFile(BIOGRID_ZIP) as z:
        inner = [n for n in z.namelist() if re.search(r"Homo_sapiens-.*\.tab3\.txt$", n)]
        if not inner:
            print("  [biogrid] SKIP — no Homo_sapiens tab3 inside zip")
            return set()
        name = inner[0]
        pairs: set[frozenset] = set()
        rows = kept = 0
        with z.open(name) as raw:
            for bline in io.TextIOWrapper(raw, encoding="utf-8", errors="replace"):
                if bline.startswith("#"):
                    continue
                c = bline.rstrip("\n").split("\t")
                if len(c) < 27:
                    continue
                rows += 1
                if c[12] != "physical":
                    continue
                if c[15] != "9606" or c[16] != "9606":
                    continue
                a, b = _clean_uniprot(c[23].split("|")[0]), _clean_uniprot(c[26].split("|")[0])
                if a and b and a != b:
                    pairs.add(frozenset((a, b)))
                    kept += 1
        print(f"  [biogrid] {name}: {rows:,} rows -> {kept:,} physical human pairs, "
              f"{len(pairs):,} unique")
    _write_pairs(BIOGRID_OUT, pairs,
                 f"BioGRID {name} — human-human physical, UniProt (SWISS-PROT). "
                 f"Built by scripts/build_known_positive_sources.py")
    return pairs


# --------------------------------------------------------------------------- #
# IntAct: PSI-MITAB 2.7, streamed. Cols (1-indexed):
#   1/2 ID(s) A/B (uniprotkb:P12345) | 10/11 taxid (taxid:9606(...))
# All IntAct entries are curated molecular interactions (no genetic epistasis),
# so any human-human uniprotkb pair counts as a documented positive.
# --------------------------------------------------------------------------- #
def build_intact() -> set[frozenset]:
    if not INTACT_ZIP.exists():
        print(f"  [intact] SKIP — {INTACT_ZIP} not found")
        return set()
    with zipfile.ZipFile(INTACT_ZIP) as z:
        inner = [n for n in z.namelist() if n.lower().endswith(".txt")]
        # prefer a pure 'human.txt' if present, else the largest member
        pure = [n for n in inner if Path(n).name.lower() == "human.txt"]
        name = pure[0] if pure else max(inner, key=lambda n: z.getinfo(n).file_size)
        pairs: set[frozenset] = set()
        rows = kept = 0
        with z.open(name) as raw:
            head = True
            for bline in io.TextIOWrapper(raw, encoding="utf-8", errors="replace"):
                if head:  # skip the '#ID(s)...' header
                    head = False
                    continue
                c = bline.rstrip("\n").split("\t")
                if len(c) < 11:
                    continue
                rows += 1
                if not (c[0].startswith("uniprotkb:") and c[1].startswith("uniprotkb:")):
                    continue
                if "9606" not in c[9] or "9606" not in c[10]:
                    continue
                a = _clean_uniprot(c[0].split(":", 1)[1])
                b = _clean_uniprot(c[1].split(":", 1)[1])
                if a and b and a != b:
                    pairs.add(frozenset((a, b)))
                    kept += 1
        print(f"  [intact] {name}: {rows:,} rows -> {kept:,} human uniprot pairs, "
              f"{len(pairs):,} unique")
    _write_pairs(INTACT_OUT, pairs,
                 f"IntAct {name} — human-human molecular interactions, UniProt. "
                 f"Built by scripts/build_known_positive_sources.py")
    return pairs


# --------------------------------------------------------------------------- #
# STRING v12.0: experimental channel > 0.9 -> known positive (direct wet-lab
# evidence, a plain "documented interaction" fact — see rules/SOURCES.md for
# why this is a known-positive source rather than a rule-engine veto rule).
# Cols (1-indexed, whitespace-separated): protein1 protein2 neighborhood
# fusion cooccurence coexpression experimental database textmining
# combined_score. Both already-downloaded local files, no network needed.
# --------------------------------------------------------------------------- #
def build_string() -> set[frozenset]:
    if not STRING_LINKS.exists() or not STRING_ALIASES.exists():
        print(f"  [string] SKIP — {STRING_LINKS} / {STRING_ALIASES} not found")
        return set()

    ensp_to_uni: dict[str, str] = {}
    with gzip.open(STRING_ALIASES, "rt") as fh:
        next(fh)  # header
        for line in fh:
            ensp, alias, source = line.rstrip("\n").split("\t")
            if "UniProt_AC" in source and ensp not in ensp_to_uni:
                ensp_to_uni[ensp] = alias
    print(f"  [string] mapped {len(ensp_to_uni):,} STRING protein ids to UniProt")

    pairs: set[frozenset] = set()
    rows = kept = 0
    with gzip.open(STRING_LINKS, "rt") as fh:
        header = next(fh).split()
        col = header.index("experimental")
        for line in fh:
            parts = line.split()
            rows += 1
            if int(parts[col]) <= STRING_EXPERIMENTAL_THRESHOLD:
                continue
            a, b = ensp_to_uni.get(parts[0]), ensp_to_uni.get(parts[1])
            if a and b and a != b:
                pairs.add(frozenset((a, b)))
                kept += 1
    print(f"  [string] {rows:,} rows -> {kept:,} pairs with experimental > 0.9, "
          f"{len(pairs):,} unique")
    _write_pairs(STRING_OUT, pairs,
                 f"STRING v12.0 experimental channel > 0.9 (raw > {STRING_EXPERIMENTAL_THRESHOLD}) "
                 f"— human-human, UniProt. Built by scripts/build_known_positive_sources.py")
    return pairs


# --------------------------------------------------------------------------- #
# UniProt -> Ensembl-gene map (human reviewed proteome), cached to disk.
# --------------------------------------------------------------------------- #
def _get(url):
    req = urllib.request.Request(url, headers={"Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=60, context=_SSL) as fh:
        link = fh.headers.get("Link", "")
        data = json.load(fh)
    m = re.search(r'<([^>]+)>;\s*rel="next"', link)
    return data, (m.group(1) if m else None)


def load_or_build_uni2ensg() -> dict[str, set[str]]:
    if MAP_CACHE.exists():
        m: dict[str, set[str]] = {}
        for line in MAP_CACHE.read_text().splitlines():
            if line and not line.startswith("#"):
                u, e = line.split("\t")[:2]
                m.setdefault(u, set()).update(e.split(","))
        print(f"  [map] loaded cached UniProt->ENSG: {len(m):,} accessions")
        return m
    print("  [map] scanning human reviewed proteome for UniProt->ENSG ...")
    url = SEARCH + "?" + urllib.parse.urlencode({
        "query": "organism_id:9606 AND reviewed:true",
        "fields": "accession,xref_ensembl", "format": "json", "size": "500"})
    m = {}
    page = 0
    while url:
        for attempt in range(4):
            try:
                data, url = _get(url)
                break
            except Exception as e:
                print(f"    page {page}: retry {attempt+1} ({e})")
                time.sleep(2 * (attempt + 1))
        else:
            break
        for e in data.get("results", []):
            acc = e.get("primaryAccession")
            ensgs = set()
            for x in e.get("uniProtKBCrossReferences", []):
                if x.get("database") == "Ensembl":
                    for p in x.get("properties", []):
                        if p.get("key") == "GeneId" and p.get("value"):
                            ensgs.add(p["value"].split(".")[0])
            if acc and ensgs:
                m[acc] = ensgs
        page += 1
        if page % 10 == 0:
            print(f"    page {page} ({len(m):,} mapped) ...")
    MAP_CACHE.parent.mkdir(parents=True, exist_ok=True)
    MAP_CACHE.write_text(
        "# UniProt primary accession -> Ensembl gene id(s), human reviewed proteome\n"
        + "".join(f"{u}\t{','.join(sorted(v))}\n" for u, v in sorted(m.items())))
    print(f"  [map] built + cached {len(m):,} accessions -> {MAP_CACHE}")
    return m


def to_ensembl(uni_pairs: set[frozenset], uni2ensg: dict[str, set[str]],
               huri_nodes: set[str], out: Path, label: str) -> int:
    ens: set[frozenset] = set()
    for pr in uni_pairs:
        a, b = tuple(pr)
        for ea in uni2ensg.get(a, ()):
            if ea not in huri_nodes:
                continue
            for eb in uni2ensg.get(b, ()):
                if eb in huri_nodes and ea != eb:
                    ens.add(frozenset((ea, eb)))
    _write_pairs(out, ens, f"{label} — mapped to Ensembl gene ids, restricted to HuRI nodes.")
    print(f"  [ensembl] {label}: {len(ens):,} pairs within HuRI's node set -> {out.name}")
    return len(ens)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-ensembl", action="store_true",
                    help="skip the HuRI Ensembl mapping (UniProt files only)")
    args = ap.parse_args()

    print("Building UniProt-space pair files (engages the SARS-CoV-2 graph) ...")
    bg = build_biogrid()
    ia = build_intact()
    st = build_string()

    if args.no_ensembl or (not bg and not ia and not st):
        return

    print("\nBuilding Ensembl-space pair files (engages the HuRI benchmark) ...")
    from negaverse.io import load_huri_graph
    huri = set(load_huri_graph().g.nodes())
    print(f"  HuRI nodes: {len(huri):,}")
    uni2ensg = load_or_build_uni2ensg()
    if bg:
        to_ensembl(bg, uni2ensg, huri, BIOGRID_OUT.with_name("biogrid_human_pairs_ensembl.tsv"),
                   "BioGRID human physical")
    if ia:
        to_ensembl(ia, uni2ensg, huri, INTACT_OUT.with_name("intact_human_pairs_ensembl.tsv"),
                   "IntAct human molecular")
    if st:
        to_ensembl(st, uni2ensg, huri, STRING_OUT.with_name("string_experimental_high_confidence_pairs_ensembl.tsv"),
                   "STRING experimental > 0.9")


if __name__ == "__main__":
    main()
