"""Load a human protein–protein interaction graph (the validation-track positives).

HuRI (Luck et al. 2020, http://www.interactome-atlas.org) is a systematic human
binary-interaction map — a clean, homogeneous PPI positive set. Edges are
Ensembl gene IDs. Download once to local-docs/ (gitignored); see README.

Note: HuRI is Ensembl-keyed while Negatome is UniProt-keyed, so in-space gold
recall against Negatome needs an ID map (Ensembl<->UniProt) — deferred.
"""
from __future__ import annotations

from pathlib import Path

from ..graph import TypedInteractionGraph

DEFAULT_PATH = "local-docs/huri/HuRI.tsv"


def load_huri_graph(path: str | Path = DEFAULT_PATH) -> TypedInteractionGraph:
    edges: list[tuple[str, str]] = []
    node_type: dict[str, str] = {}
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            parts = line.split()
            if len(parts) < 2:
                continue
            a, b = parts[0].strip(), parts[1].strip()
            if not a or not b or a == b:
                continue
            node_type[a] = "protein"
            node_type[b] = "protein"
            edges.append((a, b))
    return TypedInteractionGraph.from_edges(
        edges, node_type, admissible_types=[("protein", "protein")],
        name="huri", meta={"source": str(path), "edges": len(edges)},
    )
