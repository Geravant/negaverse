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
