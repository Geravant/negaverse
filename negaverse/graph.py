"""Typed interaction graph — the universal core abstraction (ARCHITECTURE.md §3).

Nodes are typed entities (Protein, viral/host, Ligand, ...); edges are observed
positive interactions. The engine is agnostic to whether the graph is bipartite
(host-pathogen, protein-ligand) or homogeneous (human-human PPI); an
`admissible` predicate declares which (type, type) pairs an edge may connect,
so candidate generation only ever proposes negatives in the right space.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Iterable, Optional

import networkx as nx


@dataclass
class TypedInteractionGraph:
    g: nx.Graph
    node_type: dict[str, str]
    # which unordered (type, type) pairs a real edge can connect; None = any pair
    admissible_types: Optional[set[frozenset]] = None
    name: str = "graph"
    meta: dict = field(default_factory=dict)

    @classmethod
    def from_edges(
        cls,
        edges: Iterable[tuple[str, str]],
        node_type: dict[str, str],
        admissible_types: Optional[Iterable[Iterable[str]]] = None,
        name: str = "graph",
        meta: Optional[dict] = None,
    ) -> "TypedInteractionGraph":
        g = nx.Graph()
        g.add_nodes_from(node_type.keys())
        g.add_edges_from(edges)
        adm = None
        if admissible_types is not None:
            adm = {frozenset(pair) for pair in admissible_types}
        return cls(g=g, node_type=node_type, admissible_types=adm,
                   name=name, meta=meta or {})

    def nodes_of_type(self, t: str) -> list[str]:
        return [n for n, nt in self.node_type.items() if nt == t]

    def degree(self, n: str) -> int:
        return self.g.degree(n)

    def is_positive(self, u: str, v: str) -> bool:
        return self.g.has_edge(u, v)

    def admissible(self, u: str, v: str) -> bool:
        if self.admissible_types is None:
            return True
        return frozenset((self.node_type[u], self.node_type[v])) in self.admissible_types

    @property
    def n_edges(self) -> int:
        return self.g.number_of_edges()

    def summary(self) -> dict:
        from collections import Counter
        return {
            "name": self.name,
            "nodes": self.g.number_of_nodes(),
            "edges": self.g.number_of_edges(),
            "node_types": dict(Counter(self.node_type.values())),
        }
