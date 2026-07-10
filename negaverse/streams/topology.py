"""Topology filter — link-prediction risk from graph structure (Phase 1).

Replaces the earlier Jaccard-only "embedding" stream. The signal answers: does
(u, v) look like a true interaction given the graph? A high link-likelihood =>
likely *false negative* (risky); a low one => *safe* negative; the near-boundary
middle is the informative hard negative (P2).

Three complementary structural indices, deliberately chosen so the hardness
signal is **not** the same feature the downstream benchmark uses (Jaccard):

  * **L3** — degree-normalised length-3 paths, Kovács et al. 2019 (*Network-based
    prediction of protein interactions*, Nat. Commun.). The key insight for PPI:
    interacting proteins often do **not** share partners (so common-neighbour /
    triadic closure under-predicts), but are linked by length-3 paths reflecting
    complementarity. `L3(u,v) = Σ_{x∈N(u), y∈N(v), xy∈E} 1/√(k_x·k_y)`.
  * **RA** — resource-allocation index `Σ_{x∈N(u)∩N(v)} 1/k_x` (Zhou et al.), the
    degree-de-biased common-neighbour term. Captures the triadic signal L3 skips.
  * **config-model baseline** — expected edges `(k_u·k_v)/2m` under the
    configuration model, reported as evidence so a purely degree-driven "hit" can
    be told apart from real structural proximity.

L3 and RA are unbounded counts, so they're squashed through a saturating map
`x/(x+scale)` whose `scale` is calibrated to the median score of *real* edges at
fit time — making `risk`/`value` comparable across graphs of different density.

No-overlap pairs (no common neighbour and no L3 path) are not dropped: they get a
floor risk, a high `value`, and an `easy_negative` flag + `no_overlap` bucket, so
the "no overlap => easier negative" assumption is explicit and benchmarkable
(per the locked spec) rather than hard-coded away.
"""
from __future__ import annotations

import math

import numpy as np

from ..graph import TypedInteractionGraph
from ..schema import StreamScore
from .base import Filter, Stage
from .registry import register

_L3_WEIGHT = 0.7        # L3 is the primary PPI signal (complementarity)
_RA_WEIGHT = 0.3        # RA is the secondary triadic signal
_FLOOR_VALUE = 0.98     # confidence for a no-overlap (easy) negative


@register
class TopologyFilter(Filter):
    name = "topology"
    stage = Stage.GRADED

    def __init__(self) -> None:
        self._nbr: dict[str, set[str]] = {}
        self._deg: dict[str, int] = {}
        self._two_m: int = 1
        self._l3_scale: float = 1.0
        self._ra_scale: float = 1.0

    # --- fit -------------------------------------------------------------
    def fit(self, graph: TypedInteractionGraph) -> None:
        g = graph.g
        self._nbr = {n: set(g.neighbors(n)) for n in g.nodes()}
        self._deg = {n: g.degree(n) for n in g.nodes()}
        self._two_m = 2 * g.number_of_edges() or 1
        self._calibrate(list(g.edges()))

    def _calibrate(self, edges: list[tuple[str, str]]) -> None:
        """Set saturating scales to the median L3/RA of real edges, so a
        positive-like candidate maps to a low `value` on any graph."""
        if not edges:
            return
        rng = np.random.default_rng(0)
        k = min(len(edges), 2000)
        idx = rng.choice(len(edges), size=k, replace=False)
        l3s, ras = [], []
        for i in idx:
            u, v = edges[i]
            l3 = self._l3(u, v, exclude_direct=True)
            _, ra = self._cn_ra(u, v)
            if l3 > 0:
                l3s.append(l3)
            if ra > 0:
                ras.append(ra)
        if l3s:
            self._l3_scale = float(np.median(l3s))
        if ras:
            self._ra_scale = float(np.median(ras))

    # --- structural indices ---------------------------------------------
    def _cn_ra(self, u: str, v: str) -> tuple[int, float]:
        nu, nv = self._nbr.get(u, set()), self._nbr.get(v, set())
        cn = nu & nv
        if not cn:
            return 0, 0.0
        ra = sum(1.0 / self._deg[x] for x in cn)
        return len(cn), ra

    def _l3(self, u: str, v: str, exclude_direct: bool = False) -> float:
        """Degree-normalised length-3 path score (Kovács et al. 2019)."""
        nu, nv = self._nbr.get(u), self._nbr.get(v)
        if not nu or not nv:
            return 0.0
        deg = self._deg
        total = 0.0
        for x in nu:
            if exclude_direct and x == v:
                continue
            nx = self._nbr[x]
            # edges from x into N(v): iterate the smaller set
            inter = (nx & nv) if len(nx) <= len(nv) else (nv & nx)
            if not inter:
                continue
            inv_kx = 1.0 / math.sqrt(deg[x])
            for y in inter:
                if exclude_direct and y == u:
                    continue
                total += inv_kx / math.sqrt(deg[y])
        return total

    # --- score -----------------------------------------------------------
    def score(self, graph: TypedInteractionGraph, u: str, v: str) -> StreamScore:
        if not self._nbr.get(u) or not self._nbr.get(v):
            return StreamScore(self.name, value=None,
                               evidence={"status": "no_neighbours"})
        cn, ra = self._cn_ra(u, v)
        l3 = self._l3(u, v)
        expected = self._deg[u] * self._deg[v] / self._two_m

        if cn == 0 and l3 == 0.0:
            return StreamScore(
                self.name, value=_FLOOR_VALUE, flags=["easy_negative"],
                evidence={"cn": 0, "l3": 0.0, "ra": 0.0,
                          "expected_config": round(expected, 4),
                          "risk": round(1.0 - _FLOOR_VALUE, 4),
                          # definitive absence of a local path => confident call
                          "confidence": 0.9,
                          "bucket": "no_overlap"},
            )

        l3n = l3 / (l3 + self._l3_scale)
        ran = ra / (ra + self._ra_scale) if ra > 0 else 0.0
        risk = _L3_WEIGHT * l3n + _RA_WEIGHT * ran
        return StreamScore(
            self.name, value=round(1.0 - risk, 4),
            evidence={"cn": cn, "l3": round(l3, 4), "ra": round(ra, 4),
                      "l3_norm": round(l3n, 4), "ra_norm": round(ran, 4),
                      "expected_config": round(expected, 4),
                      "risk": round(risk, 4),
                      # reported confidence = how much structural signal backs the
                      # call (little overlap => shaky; strong L3/RA => decisive)
                      "confidence": round(min(1.0, l3n + ran), 4)},
        )
