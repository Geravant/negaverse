"""PLI-native hardness: pocket–ligand fit (the bipartite replacement for topology).

On a protein–ligand bipartite graph topology is dead (shared-neighbour = 0), so the
train/eval hardness signal can't come from the graph. Instead, "hard" = the pair
LOOKS like a binder: the ligand's size fits the pocket the way real binders do.

`PliPocketFitFilter` calibrates the real-binder distribution of
log(ligand heavy-atoms / pocket residues) over the graph's edges, then scores a
candidate by how typical its size ratio is (a Gaussian binder-likeness in [0,1]).
High likeness => looks like a binder => hard negative; it also contributes the
inverse to the true-negative confidence fused across streams.
"""
from __future__ import annotations

import math

from ..graph import TypedInteractionGraph
from ..schema import StreamScore
from .base import Filter, Stage
from .registry import register


@register
class PliPocketFitFilter(Filter):
    name = "pli_pocket_fit"
    stage = Stage.GRADED
    modalities = frozenset({"pli"})
    provides_hardness = True                      # its binder-likeness drives the split

    def __init__(self, annotations: dict | None = None) -> None:
        self._ann_arg = annotations
        self._ann: dict[str, dict] = {}
        self._mu = 0.0
        self._sigma = 1.0

    def _pair(self, graph, u, v):
        """Return (protein_rec, ligand_rec) or None if this isn't a protein–ligand pair."""
        tu, tv = graph.node_type.get(u), graph.node_type.get(v)
        if tu == "protein" and tv == "ligand":
            return self._ann.get(u, {}), self._ann.get(v, {})
        if tu == "ligand" and tv == "protein":
            return self._ann.get(v, {}), self._ann.get(u, {})
        return None

    @staticmethod
    def _log_ratio(prot: dict, lig: dict):
        vol, pv = lig.get("volume"), prot.get("pocket_volume")
        if vol is None or pv is None or vol <= 0 or pv <= 0:
            return None
        return math.log(vol / pv)

    def fit(self, graph: TypedInteractionGraph) -> None:
        if self._ann_arg is not None:
            self._ann = self._ann_arg
        else:
            from ..io.annotations import build_annotation_table
            self._ann = build_annotation_table()
        vals = []
        for u, v in graph.g.edges():                 # real binders = calibration set
            pl = self._pair(graph, u, v)
            if pl and (lr := self._log_ratio(*pl)) is not None:
                vals.append(lr)
        if vals:
            self._mu = sum(vals) / len(vals)
            var = sum((x - self._mu) ** 2 for x in vals) / max(len(vals) - 1, 1)
            self._sigma = math.sqrt(var) or 1.0

    def score(self, graph: TypedInteractionGraph, u: str, v: str) -> StreamScore:
        pl = self._pair(graph, u, v)
        if pl is None:
            return StreamScore(self.name, value=None)          # not a protein–ligand pair
        lr = self._log_ratio(*pl)
        if lr is None:
            return StreamScore(self.name, value=None,
                               evidence={"status": "missing_size_annotation"})
        z = (lr - self._mu) / self._sigma
        likeness = math.exp(-0.5 * z * z)            # 1 = typical binder size-ratio = hard
        value = round(1.0 - likeness, 4)             # confidence it's a TRUE negative
        return StreamScore(
            self.name, value=value,
            flags=["binder_like_size"] if likeness >= 0.6 else [],
            evidence={"hardness": round(likeness, 4), "confidence": round(abs(2 * value - 1), 4),
                      "size_z": round(z, 3), "pocket_fit_likeness": round(likeness, 4)},
        )
