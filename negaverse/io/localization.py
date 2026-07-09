"""Subcellular-localization annotations for the co-localization filter.

Format (TSV, one node per line):

    node<TAB>compartment1,compartment2,...

Compartments are free strings (GO cellular-component terms, or coarse labels like
"cytoplasm"/"nucleus"); the filter only cares about set overlap, so any consistent
vocabulary works. Build one with scripts/fetch_go_localization.py, or hand-write a
small map. Keyed to whatever node IDs your graph uses (Ensembl for HuRI, UniProt
for the SARS graph).
"""
from __future__ import annotations

from pathlib import Path

DEFAULT_PATH = "local-docs/localization/go_cc.tsv"


def load_localization_tsv(path: str | Path = DEFAULT_PATH) -> dict[str, set[str]]:
    ann: dict[str, set[str]] = {}
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.rstrip("\n")
            if not line or line.startswith("#"):
                continue
            parts = line.split("\t")
            if len(parts) < 2 or not parts[1].strip():
                continue
            node = parts[0].strip()
            comps = {c.strip().lower() for c in parts[1].split(",") if c.strip()}
            if node and comps:
                ann[node] = comps
    return ann
