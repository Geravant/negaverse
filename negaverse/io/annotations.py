"""Per-entity annotation records for the rule engine.

A record is `dict[node -> dict[field -> value]]`; a rule's `when` reads fields
(`a.compartments`, `a.surface_hydrophobicity`, `ligand.logp`, …). Annotations are
merged from whatever sources exist; a field absent for a node makes any rule that
reads it abstain for that node. Add a new annotation type = load it here under a
new field name — no rule-engine changes.

Two kinds are wired:
  * set-valued `compartments` (GO cellular-component terms) from a TSV.
  * scalar fields (e.g. `surface_hydrophobicity`) from `node<TAB>value` TSVs under
    local-docs/annotations/ (compute one with scripts/compute_hydrophobicity.py).

Graph-derived fields (neighbors / degree / graph_two_m) are NOT loaded here — the
rule filters add them from the live graph at fit time (see streams/rules.py), so
topology rules work without a data file.
"""
from __future__ import annotations

from pathlib import Path

from .localization import DEFAULT_PATH as LOC_PATH, load_localization_tsv

ANNOT_DIR = "local-docs/annotations"
# field name -> default TSV path (node<TAB>float). Extend as sources are added.
_SCALAR_FIELDS = {
    "surface_hydrophobicity": f"{ANNOT_DIR}/hydrophobicity.tsv",
}
# set-valued fields (node<TAB>comma-separated terms), same shape as localization.
_SET_FIELDS = {
    "processes": f"{ANNOT_DIR}/go_bp.tsv",       # GO biological_process (build_go_process.py)
}


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


def build_annotation_table(localization_path: str | Path = LOC_PATH,
                           scalar_fields: dict[str, str] | None = None,
                           set_fields: dict[str, str] | None = None) -> dict[str, dict]:
    table: dict[str, dict] = {}
    try:
        for node, comps in load_localization_tsv(localization_path).items():
            table.setdefault(node, {})["compartments"] = comps
    except FileNotFoundError:
        pass                      # no localization data -> location rules abstain
    # other set-valued fields (GO biological_process, …) — same TSV shape
    for field, path in (set_fields or _SET_FIELDS).items():
        try:
            for node, terms in load_localization_tsv(path).items():
                table.setdefault(node, {})[field] = terms
        except FileNotFoundError:
            pass                  # field not sourced -> rules that read it abstain
    for field, path in (scalar_fields or _SCALAR_FIELDS).items():
        try:
            for node, val in _load_scalar_tsv(path).items():
                table.setdefault(node, {})[field] = val
        except FileNotFoundError:
            pass                  # field not sourced -> rules that read it abstain
    return table
