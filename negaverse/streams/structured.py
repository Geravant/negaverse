"""Structured filters (ARCHITECTURE.md §5.1 / IMPLEMENTATION-PLAN §1).

Two filters at two hourglass stages:

  * KnownPositiveVeto (VETO) — front-of-hourglass hard filter. A pair present as
    a positive in the graph, or in an injected set of externally-known positives
    (the Phase-1 database-screening seed), is dropped and never emitted.
  * PlausibilityFilter (GRADED) — a cheap plausibility prior: an implausible pair
    is a *safe* negative, a plausible one is *risky*. Proxied by target
    promiscuity (a protein many things bind is a plausible binder). Real
    localization / GO / rule-driven signals slot in behind the same interface.
"""
from __future__ import annotations

from typing import Optional

from ..graph import TypedInteractionGraph
from ..schema import StreamScore
from .base import Filter, Stage
from .registry import register


@register
class KnownPositiveVeto(Filter):
    name = "known_positive_veto"
    stage = Stage.VETO

    def __init__(self, known_positives: Optional[set[frozenset]] = None):
        # extra positives from external DBs (union-of-sources exclusion)
        self.known_positives = known_positives or set()

    def score(self, graph: TypedInteractionGraph, u: str, v: str) -> StreamScore:
        if graph.is_positive(u, v) or frozenset((u, v)) in self.known_positives:
            return StreamScore(self.name, value=None, veto=True,
                               evidence={"reason": "known_positive"})
        return StreamScore(self.name, value=None, evidence={"reason": "not_known_positive"})


@register
class StructuredStream(Filter):
    name = "structured"
    stage = Stage.GRADED

    def __init__(self) -> None:
        self._max_deg: dict[str, int] = {}

    def fit(self, graph: TypedInteractionGraph) -> None:
        # normalise promiscuity per node type so a hub is judged against its peers
        self._max_deg = {}
        for n, t in graph.node_type.items():
            self._max_deg[t] = max(self._max_deg.get(t, 1), graph.degree(n))

    def score(self, graph: TypedInteractionGraph, u: str, v: str) -> StreamScore:
        tv, tu = graph.node_type[v], graph.node_type[u]
        promis = max(graph.degree(v) / self._max_deg.get(tv, 1),
                     graph.degree(u) / self._max_deg.get(tu, 1))
        value = 1.0 - promis
        return StreamScore(
            self.name, value=round(value, 4),
            evidence={"deg_u": graph.degree(u), "deg_v": graph.degree(v),
                      "promiscuity": round(promis, 4)},
        )
