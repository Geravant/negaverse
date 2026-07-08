"""Layer 1 — candidate generation (ARCHITECTURE.md §4).

Enumerate non-edges of the positive graph that live in an admissible type-space.
Full enumeration is K(K-1)/2 - P and usually intractable, so above a threshold
we fall back to sampling a working pool. For the SARS-CoV-2 bipartite graph the
complement is small (~10^4) and fully enumerated.
"""
from __future__ import annotations

from typing import Iterator, Optional

import numpy as np

from .graph import TypedInteractionGraph


def _admissible_pairs(graph: TypedInteractionGraph) -> Iterator[tuple[str, str]]:
    """All node pairs allowed by the type schema, positives included."""
    if graph.admissible_types is None:
        nodes = list(graph.g.nodes())
        for i in range(len(nodes)):
            for j in range(i + 1, len(nodes)):
                yield nodes[i], nodes[j]
        return
    # bipartite / typed: iterate only across admissible type pairs
    by_type: dict[str, list[str]] = {}
    for n, t in graph.node_type.items():
        by_type.setdefault(t, []).append(n)
    seen: set[frozenset] = set()
    for pair in graph.admissible_types:
        pl = sorted(pair)
        if len(pl) == 1:  # homogeneous within one type
            ns = by_type.get(pl[0], [])
            for i in range(len(ns)):
                for j in range(i + 1, len(ns)):
                    yield ns[i], ns[j]
        else:
            a, b = pl
            for na in by_type.get(a, []):
                for nb in by_type.get(b, []):
                    yield na, nb


def generate_candidates(
    graph: TypedInteractionGraph,
    max_pool: int = 200_000,
    seed: int = 0,
) -> list[tuple[str, str]]:
    """Return admissible non-edges. Enumerate if the complement is small,
    else uniformly sample a working pool of size ~max_pool."""
    rng = np.random.default_rng(seed)
    complement = [(u, v) for u, v in _admissible_pairs(graph)
                  if not graph.is_positive(u, v)]
    if len(complement) <= max_pool:
        return complement
    idx = rng.choice(len(complement), size=max_pool, replace=False)
    return [complement[i] for i in idx]
