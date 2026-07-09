"""Load Negatome 2.0 gold non-interacting protein pairs (ARCHITECTURE.md §4 L4, P3).

Files (UniProt-keyed, tab-separated) live in local-docs/negatome2/:
  combined_stringent.txt   6136 pairs  (manual + PDB, minus IntAct)  <- default
  manual_stringent.txt     1990 pairs  (+ PubMed id, detection method)
  pdb_stringent.txt        4161 pairs  (+ PDB code)

These are mammalian human-human non-interactions, so they anchor/evaluate a
human-human PPI graph — NOT the viral-host SARS-CoV-2 graph (different space).
"""
from __future__ import annotations

from pathlib import Path

DEFAULT_PATH = "local-docs/negatome2/combined_stringent.txt"
DEFAULT_MAP_PATH = "local-docs/mappings/uniprot_to_ensembl.tsv"


def load_negatome_pairs(path: str | Path = DEFAULT_PATH) -> set[frozenset]:
    """Return gold non-interacting pairs as a set of frozenset({uniprot_a, uniprot_b})."""
    pairs: set[frozenset] = set()
    with open(path, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            cols = line.split("\t")
            if len(cols) < 2:
                continue
            a, b = cols[0].strip(), cols[1].strip()
            if a and b and a != b:
                pairs.add(frozenset((a, b)))
    return pairs


def load_uniprot_ensembl_map(path: str | Path = DEFAULT_MAP_PATH) -> dict[str, set[str]]:
    """UniProt accession -> set of Ensembl gene IDs (ENSG). Built by
    scripts/build_uniprot_ensembl_map.py (see README)."""
    mapping: dict[str, set[str]] = {}
    with open(path, encoding="utf-8") as fh:
        next(fh, None)  # header
        for line in fh:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 2 or not parts[1]:
                continue
            mapping[parts[0]] = {g for g in parts[1].split(",") if g}
    return mapping


def load_negatome_in_ensembl_space(
    huri_nodes: set[str],
    negatome_path: str | Path = DEFAULT_PATH,
    map_path: str | Path = DEFAULT_MAP_PATH,
) -> set[frozenset]:
    """Negatome gold non-interactions mapped into Ensembl-gene space and
    restricted to the given HuRI node set — i.e. gold negatives usable as
    *in-space* test negatives for the HuRI benchmark.

    A UniProt id may map to several ENSG; a pair is emitted for every ENSG×ENSG
    combination whose both endpoints are HuRI nodes (self-pairs dropped).
    """
    pairs = load_negatome_pairs(negatome_path)
    umap = load_uniprot_ensembl_map(map_path)
    out: set[frozenset] = set()
    for pr in pairs:
        a, b = tuple(pr)
        ga = umap.get(a, set()) & huri_nodes
        gb = umap.get(b, set()) & huri_nodes
        for x in ga:
            for y in gb:
                if x != y:
                    out.add(frozenset((x, y)))
    return out
