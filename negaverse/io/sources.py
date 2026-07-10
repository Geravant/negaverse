"""Load external known-positive interaction sources (rules/sources.yaml).

The graph loaded for a run is one experiment's evidence, not the full universe of
known interactions. A candidate can be a documented positive in IntAct/BioGRID/…
without being an edge in that graph. This loader reads the source manifest and
returns the union of pairs, which KnownPositiveVeto removes before scoring. See
rules/SOURCES.md for the manifest contract.

Each source `path` is a 2-column pairs file (tab/space separated, `#` comments).
Missing files are skipped (with a report), so the manifest can list sources ahead
of the data being placed.
"""
from __future__ import annotations

from pathlib import Path

import yaml

DEFAULT_SOURCES = "rules/sources.yaml"


def _read_pairs(path: Path, restrict_to: set[str] | None) -> tuple[set[frozenset], int]:
    pairs: set[frozenset] = set()
    with open(path, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            cols = line.replace("\t", " ").split()
            if len(cols) < 2:
                continue
            a, b = cols[0], cols[1]
            if a == b:
                continue
            if restrict_to is not None and (a not in restrict_to or b not in restrict_to):
                continue
            pairs.add(frozenset((a, b)))
    return pairs, len(pairs)


def load_positive_sources(sources_path: str | Path = DEFAULT_SOURCES,
                          restrict_to: set[str] | None = None,
                          modality: str | None = None) -> tuple[set[frozenset], dict]:
    """Union of documented-positive pairs across the manifest.

    restrict_to: keep only pairs whose *both* ids are in this set (e.g. the graph's
      node ids) — so a PLI source's ligand ids simply drop out on a PPI graph.
    modality: if given, only load sources of that modality.
    Returns (pairs, report) where report = {"loaded": {name: n}, "missing": [name]}.
    """
    path = Path(sources_path)
    pairs: set[frozenset] = set()
    report: dict = {"loaded": {}, "missing": []}
    if not path.exists():
        return pairs, report
    for e in (yaml.safe_load(path.read_text()) or []):
        if modality and e.get("modality") != modality:
            continue
        fp = Path(e.get("path", ""))
        if not fp.exists():
            report["missing"].append(e.get("name", str(fp)))
            continue
        got, n = _read_pairs(fp, restrict_to)
        pairs |= got
        report["loaded"][e.get("name", str(fp))] = n
    return pairs, report
