"""Layer 1 — candidate generation (ARCHITECTURE.md §4).

Enumerate non-edges of the positive graph that live in an admissible type-space.
The full complement is K(K-1)/2 - P and usually intractable, so above a
threshold we rejection-sample a bounded working pool directly instead of
materialising the whole complement (a homogeneous human PPI graph has ~10^7-10^8
non-edges — enumeration would exhaust memory).
"""
from __future__ import annotations

from typing import Iterator

import numpy as np

from .graph import TypedInteractionGraph

# above this many admissible pairs, sample instead of enumerate
_ENUMERATE_MAX = 1_000_000


def _by_type(graph: TypedInteractionGraph) -> dict[str, list[str]]:
    bt: dict[str, list[str]] = {}
    for n, t in graph.node_type.items():
        bt.setdefault(t, []).append(n)
    return bt


def _type_pairs(graph: TypedInteractionGraph) -> list[tuple[str, ...]]:
    """Admissible (type,)/(type,type) groups. None => one homogeneous group."""
    if graph.admissible_types is None:
        return [("__all__",)]
    return [tuple(sorted(fs)) for fs in graph.admissible_types]


def _admissible_size(graph: TypedInteractionGraph, by_type: dict[str, list[str]]) -> int:
    total = 0
    for pl in _type_pairs(graph):
        if pl == ("__all__",):
            n = graph.g.number_of_nodes()
            total += n * (n - 1) // 2
        elif len(pl) == 1:
            n = len(by_type.get(pl[0], []))
            total += n * (n - 1) // 2
        else:
            total += len(by_type.get(pl[0], [])) * len(by_type.get(pl[1], []))
    return total


def _admissible_pairs(graph: TypedInteractionGraph,
                      by_type: dict[str, list[str]]) -> Iterator[tuple[str, str]]:
    for pl in _type_pairs(graph):
        if pl == ("__all__",):
            ns = list(graph.g.nodes())
            for i in range(len(ns)):
                for j in range(i + 1, len(ns)):
                    yield ns[i], ns[j]
        elif len(pl) == 1:
            ns = by_type.get(pl[0], [])
            for i in range(len(ns)):
                for j in range(i + 1, len(ns)):
                    yield ns[i], ns[j]
        else:
            for a in by_type.get(pl[0], []):
                for b in by_type.get(pl[1], []):
                    yield a, b


def _sample_candidates(graph, by_type, type_pairs, max_pool, rng) -> list[tuple[str, str]]:
    if graph.admissible_types is None:
        by_type = {**by_type, "__all__": list(graph.g.nodes())}
    got: set[frozenset] = set()
    out: list[tuple[str, str]] = []
    cap = max_pool * 20 + 1000
    attempts = 0
    while len(out) < max_pool and attempts < cap:
        attempts += 1
        pl = type_pairs[rng.integers(len(type_pairs))]
        if len(pl) == 1:
            ns = by_type.get(pl[0], [])
            if len(ns) < 2:
                continue
            a, b = ns[rng.integers(len(ns))], ns[rng.integers(len(ns))]
        else:
            A, B = by_type.get(pl[0], []), by_type.get(pl[1], [])
            if not A or not B:
                continue
            a, b = A[rng.integers(len(A))], B[rng.integers(len(B))]
        if a == b or graph.is_positive(a, b):
            continue
        key = frozenset((a, b))
        if key in got:
            continue
        got.add(key)
        out.append((a, b))
    return out


def generate_candidates(
    graph: TypedInteractionGraph,
    max_pool: int = 200_000,
    seed: int = 0,
) -> list[tuple[str, str]]:
    """Return admissible non-edges: enumerate when the space is small, else
    rejection-sample a working pool of size ~max_pool."""
    rng = np.random.default_rng(seed)
    by_type = _by_type(graph)
    type_pairs = _type_pairs(graph)
    if _admissible_size(graph, by_type) <= max(_ENUMERATE_MAX, max_pool * 2):
        complement = [(u, v) for u, v in _admissible_pairs(graph, by_type)
                      if not graph.is_positive(u, v)]
        if len(complement) <= max_pool:
            return complement
        idx = rng.choice(len(complement), size=max_pool, replace=False)
        return [complement[i] for i in idx]
    return _sample_candidates(graph, by_type, type_pairs, max_pool, rng)
