"""Per-entity annotation records for the rule engine.

A record is `dict[node -> dict[field -> value]]`; a rule's `when` reads fields
(`a.compartments`, `a.surface_hydrophobicity`, `ligand.logp`, …). Annotations are
merged from whatever sources exist; a field absent for a node makes any rule that
reads it abstain for that node. Add a new annotation type = load it here under a
new field name — no rule-engine changes.

Two kinds are wired:
  * set-valued `compartments` (GO cellular-component terms) from a TSV.
  * scalar fields (`surface_hydrophobicity`, `pocket_volume`,
    `pocket_hydrophobicity`, `pocket_polarity`) from `node<TAB>value` TSVs under
    local-docs/annotations/ — compute with scripts/compute_surface_hydrophobicity.py
    (two-tier: DSSP+AlphaFold structure, sequence fallback) and
    scripts/compute_pocket_descriptors.py (fpocket; protein-side pocket
    descriptors only, structure required — no sequence fallback exists for these).

Graph-derived fields (neighbors / degree / graph_two_m) are NOT loaded here — the
rule filters add them from the live graph at fit time (see streams/rules.py), so
topology rules work without a data file.

Pairwise fields (e.g. `string_score_with_b`) are a third kind, loaded
separately via `build_pair_annotation_table()` below — their value depends
on *both* entities in a pair, not one node alone, so they can't live in the
`dict[node -> {field: value}]` shape above. See that function's docstring
for the file format; `streams/rules.py::_RuleFilterBase` merges the right
pair's values onto the `a`-side record at score() time, fresh per (u, v)
call.
"""
from __future__ import annotations

from pathlib import Path

from .localization import DEFAULT_PATH as LOC_PATH, load_localization_tsv

ANNOT_DIR = "local-docs/annotations"
# field name -> default TSV path (node<TAB>float). Extend as sources are added.
_SCALAR_FIELDS = {
    "surface_hydrophobicity": f"{ANNOT_DIR}/hydrophobicity.tsv",
    "pocket_volume": f"{ANNOT_DIR}/pocket_volume.tsv",
    "pocket_hydrophobicity": f"{ANNOT_DIR}/pocket_hydrophobicity.tsv",
    "pocket_polarity": f"{ANNOT_DIR}/pocket_polarity.tsv",
}
# field name -> default TSV path (node_a<TAB>node_b<TAB>float, order-independent).
# Extend as pairwise sources are added; see build_pair_annotation_table().
_PAIR_FIELDS: dict[str, str] = {}


def _load_scalar_tsv(path: str | Path) -> dict[str, float]:
    out: dict[str, float] = {}
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            try:
                out[parts[0]] = float(parts[1])
            except ValueError:
                continue
    return out


def _load_pair_tsv(path: str | Path) -> dict[frozenset[str], float]:
    """node_a<TAB>node_b<TAB>value. Order-independent: pair properties like a
    STRING score are symmetric, so the key is a frozenset of the two node
    IDs, not an ordered tuple."""
    out: dict[frozenset[str], float] = {}
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split("\t")
            if len(parts) < 3:
                continue
            try:
                out[frozenset((parts[0], parts[1]))] = float(parts[2])
            except ValueError:
                continue
    return out


def build_annotation_table(localization_path: str | Path = LOC_PATH,
                           scalar_fields: dict[str, str] | None = None) -> dict[str, dict]:
    table: dict[str, dict] = {}
    try:
        for node, comps in load_localization_tsv(localization_path).items():
            table.setdefault(node, {})["compartments"] = comps
    except FileNotFoundError:
        pass                      # no localization data -> location rules abstain
    for field, path in (scalar_fields or _SCALAR_FIELDS).items():
        try:
            for node, val in _load_scalar_tsv(path).items():
                table.setdefault(node, {})[field] = val
        except FileNotFoundError:
            pass                  # field not sourced -> rules that read it abstain
    return table


def build_pair_annotation_table(
        pair_fields: dict[str, str] | None = None) -> dict[str, dict[frozenset[str], float]]:
    """field name -> {frozenset({node_a, node_b}): value}. Missing file -> that
    field simply absent (same silent-abstain convention as build_annotation_table)."""
    table: dict[str, dict[frozenset[str], float]] = {}
    for field, path in (pair_fields or _PAIR_FIELDS).items():
        try:
            table[field] = _load_pair_tsv(path)
        except FileNotFoundError:
            pass
    return table
