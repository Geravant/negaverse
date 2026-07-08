"""Structured-filter stream (ARCHITECTURE.md §5.1).

Deterministic, explainable signals from structured biology. In this prototype:

  * hard veto  — a pair present as a positive in the graph (or in an injected
    set of externally-known positives) is never emitted as a negative. This is
    the Layer-2 false-negative removal that makes the whole tool defensible.
  * plausibility prior — an implausible pair is a *safe* negative; a plausible
    one is *risky*. We proxy plausibility by target promiscuity: a host protein
    that many baits bind is a plausible binder, so a non-edge to it is a riskier
    negative (lower confidence). Localization / co-expression / GO signals slot
    in here later behind the same interface.
"""
from __future__ import annotations

from typing import Optional

from ..graph import TypedInteractionGraph
from ..schema import StreamScore
from .base import Stream


class StructuredStream(Stream):
    name = "structured"

    def __init__(self, known_positives: Optional[set[frozenset]] = None):
        # extra positives from external DBs (union-of-sources exclusion, §4 L2)
        self.known_positives = known_positives or set()
        self._max_deg: dict[str, int] = {}

    def fit(self, graph: TypedInteractionGraph) -> None:
        # normalise promiscuity per node type so a hub is judged against its peers
        self._max_deg = {}
        for n, t in graph.node_type.items():
            self._max_deg[t] = max(self._max_deg.get(t, 1), graph.degree(n))

    def score(self, graph: TypedInteractionGraph, u: str, v: str) -> StreamScore:
        if graph.is_positive(u, v) or frozenset((u, v)) in self.known_positives:
            return StreamScore(self.name, value=None, veto=True,
                               evidence={"reason": "known_positive"})
        # plausibility prior from target promiscuity (higher promiscuity -> lower
        # confidence it is a true negative)
        tv = graph.node_type[v]
        tu = graph.node_type[u]
        promis = max(graph.degree(v) / self._max_deg.get(tv, 1),
                     graph.degree(u) / self._max_deg.get(tu, 1))
        value = 1.0 - promis
        return StreamScore(
            self.name, value=round(value, 4), veto=False,
            evidence={"deg_u": graph.degree(u), "deg_v": graph.degree(v),
                      "promiscuity": round(promis, 4)},
        )
