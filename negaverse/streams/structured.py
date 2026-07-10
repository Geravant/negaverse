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

    def __init__(self, known_positives: Optional[set[frozenset]] = None,
                 sources_path: str = "rules/sources.yaml", load_sources: bool = True):
        # extra positives injected directly, plus the manifest of external DBs
        self.known_positives: set[frozenset] = set(known_positives or set())
        self._sources_path = sources_path
        self._load_sources = load_sources
        self.sources_report: dict = {}

    def fit(self, graph: TypedInteractionGraph) -> None:
        # union-of-sources exclusion: veto candidates documented as positives in
        # IntAct/BioGRID/… even if absent from this graph. Restricted to graph
        # nodes (so a PLI source can't match a PPI graph) and graceful when the
        # manifest / its files aren't present.
        if not self._load_sources:
            return
        from ..io.sources import load_positive_sources
        extra, self.sources_report = load_positive_sources(
            self._sources_path, restrict_to=set(graph.g.nodes()))
        self.known_positives |= extra

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
