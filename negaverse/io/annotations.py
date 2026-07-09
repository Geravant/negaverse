"""Per-entity annotation records for the rule engine.

A record is `dict[node -> dict[field -> value]]`; a rule's `when` reads fields
(`a.compartments`, `ligand.logp`, …). Annotations are merged from whatever
sources exist; a field absent for a node makes any rule that reads it abstain for
that node. Add a new annotation type = load it here under a new field name — no
rule-engine changes.

Currently wired: `compartments` (set of GO cellular-component terms) from the
localization TSV. Extend by loading hydrophobicity / pocket volume / logP etc.
"""
from __future__ import annotations

from pathlib import Path

from .localization import DEFAULT_PATH as LOC_PATH, load_localization_tsv


def build_annotation_table(localization_path: str | Path = LOC_PATH) -> dict[str, dict]:
    table: dict[str, dict] = {}
    try:
        for node, comps in load_localization_tsv(localization_path).items():
            table.setdefault(node, {})["compartments"] = comps
    except FileNotFoundError:
        pass                      # no localization data -> location rules abstain
    return table
