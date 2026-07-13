"""Generic positives-file loader for `negaverse run --input <path>`.

Tab- or whitespace-separated, no header, one positive pair per line:

    u   v                  -> both endpoints typed "protein" (e.g. HuRI)
    u   v   u_type  v_type  -> explicit per-endpoint types (e.g. viral/host)

The candidate (admissible) space — which node-type pairs may generate
negatives — is inferred rather than configured: cross-type pairs (u_type !=
v_type) if any are present, since those are the "new pairs to screen" a
heterotypic graph is built for; same-type pairs are kept as graph structure
(topology signal) but are not themselves proposed as candidates. On a
homogeneous file (HuRI-style, everything "protein") there is only one type
pair, so it is used as-is.
"""
from __future__ import annotations

from pathlib import Path

from ..graph import TypedInteractionGraph


def load_generic_graph(path: str | Path, name: str | None = None) -> TypedInteractionGraph:
    edges: list[tuple[str, str]] = []
    node_type: dict[str, str] = {}
    pair_types: set[frozenset] = set()
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 2:
                parts = line.split()
            if len(parts) < 2:
                continue
            a, b = parts[0].strip(), parts[1].strip()
            if not a or not b or a == b:
                continue
            ta = parts[2].strip() if len(parts) > 2 and parts[2].strip() else "protein"
            tb = parts[3].strip() if len(parts) > 3 and parts[3].strip() else "protein"
            node_type[a] = ta
            node_type[b] = tb
            edges.append((a, b))
            pair_types.add(frozenset((ta, tb)))

    heterotypic = {p for p in pair_types if len(p) == 2}
    admissible = heterotypic or pair_types

    return TypedInteractionGraph.from_edges(
        edges, node_type, admissible_types=admissible,
        name=name or Path(path).stem,
        meta={"source": str(path), "edges": len(edges)},
    )
