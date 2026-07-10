"""Load a bipartite protein–ligand interaction graph (the PLI positives).

PLINDER (https://plinder.sh, Durairaj et al. 2024) is a gold-standard protein–
ligand interaction resource. `scripts/build_plinder_pli.py` distils its annotation
table into `local-docs/plinder/pli_edges.tsv` (UniProt protein <TAB> CCD ligand
binders) plus the ligand/pocket annotation TSVs the rule engine reads.

Nodes are typed protein/ligand and the only admissible pair is (protein, ligand),
so candidate generation produces protein–ligand non-edges (never protein–protein
or ligand–ligand). This is the same bipartite shape as the SARS-CoV-2 viral↔host
graph, which already runs end-to-end.
"""
from __future__ import annotations

from pathlib import Path

from ..graph import TypedInteractionGraph

DEFAULT_PATH = "local-docs/plinder/pli_edges.tsv"


def load_plinder_graph(path: str | Path = DEFAULT_PATH,
                       max_edges: int | None = None,
                       seed: int = 0) -> TypedInteractionGraph:
    """protein(UniProt)–ligand(CCD) bipartite graph. `max_edges` subsamples edges
    (deterministically) for a faster run; None = all."""
    edges: list[tuple[str, str]] = []
    node_type: dict[str, str] = {}
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split("\t") if "\t" in line else line.split()
            if len(parts) < 2:
                continue
            prot, lig = parts[0].strip(), parts[1].strip()
            if not prot or not lig:
                continue
            node_type[prot] = "protein"
            node_type[lig] = "ligand"
            edges.append((prot, lig))
    if max_edges and len(edges) > max_edges:
        import numpy as np
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(edges), size=max_edges, replace=False)
        edges = [edges[i] for i in idx]
        node_type = {n: t for e in edges for n, t in
                     ((e[0], "protein"), (e[1], "ligand"))}
    return TypedInteractionGraph.from_edges(
        edges, node_type, admissible_types=[("protein", "ligand")],
        name="plinder", meta={"source": str(path), "edges": len(edges)},
    )
