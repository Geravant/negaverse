"""Graph / geometric-embedding stream (ARCHITECTURE.md §5.3).

A topological view the other streams can't give: a link-prediction score for
(u, v). High score => the pair looks like the positives => *likely false
negative* (risky); low score => *safe* negative; mid/near-boundary => the
informative hard negatives P2 wants.

Prototype signal is a structural link-prediction score computed directly from
the graph (no training): for candidate (u, v), how strongly does u attach to
nodes that resemble v, where resemblance is the Jaccard overlap of their
neighbourhoods. This is the cheap stand-in the walking skeleton uses; node2vec /
GNN / PLM-distance embeddings replace `topo(...)` behind this same interface.
"""
from __future__ import annotations

from ..graph import TypedInteractionGraph
from ..schema import StreamScore
from .base import Filter, Stage
from .registry import register


@register
class EmbeddingStream(Filter):
    name = "embedding"
    stage = Stage.GRADED

    def __init__(self) -> None:
        self._nbr: dict[str, set[str]] = {}

    def fit(self, graph: TypedInteractionGraph) -> None:
        self._nbr = {n: set(graph.g.neighbors(n)) for n in graph.g.nodes()}

    def _jaccard(self, a: str, b: str) -> float:
        na, nb = self._nbr.get(a, set()), self._nbr.get(b, set())
        if not na or not nb:
            return 0.0
        inter = len(na & nb)
        if inter == 0:
            return 0.0
        return inter / len(na | nb)

    def topo(self, u: str, v: str) -> tuple[float, int]:
        """Mean resemblance between v and the nodes u already binds. In [0,1]."""
        refs = self._nbr.get(u, set())
        if not refs:
            return 0.0, 0
        s = sum(self._jaccard(p, v) for p in refs)
        return s / len(refs), len(refs)

    def score(self, graph: TypedInteractionGraph, u: str, v: str) -> StreamScore:
        # score is symmetric-ish; take the stronger of the two directions so a
        # hub on either side still surfaces a likely interaction
        t_uv, n_uv = self.topo(u, v)
        t_vu, n_vu = self.topo(v, u)
        topo = max(t_uv, t_vu)
        if n_uv == 0 and n_vu == 0:
            return StreamScore(self.name, value=None,
                               evidence={"status": "no_neighbours"})
        return StreamScore(
            self.name, value=round(1.0 - topo, 4),
            evidence={"topo": round(topo, 4)},
        )
