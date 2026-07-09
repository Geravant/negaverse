"""Co-localization filter — subcellular-compartment plausibility (Phase 1, PPI).

An **independent** signal, orthogonal to graph topology: two proteins that never
share a subcellular compartment can't physically interact, so a disjoint-
compartment pair is a *safe* negative; a pair sharing compartments *could* interact,
so it's a riskier (harder) negative. Because this signal comes from biology
(GO cellular-component / experimental localization), not from the interaction
graph, it does not share information with the topology-selection indices — which is
exactly what a fair downstream benchmark needs (see docs/BENCHMARK-FINDINGS.md).

Annotation interface (the tunable surface): a plain `dict[node -> set[compartment]]`.
Provide it directly, or point `path=` at a TSV (`node<TAB>comp1,comp2,...`) built by
`scripts/fetch_go_localization.py`. With no annotations the filter abstains on every
pair (value=None) — so it's harmless in the default set until data is supplied,
mirroring the literature filter's key-optional behaviour.
"""
from __future__ import annotations

from pathlib import Path

from ..graph import TypedInteractionGraph
from ..schema import StreamScore
from .base import Filter, Stage
from .registry import register

DEFAULT_LOC_PATH = "local-docs/localization/go_cc.tsv"
_SAFE_DISJOINT = 0.9        # confidence for a different-compartment (safe) negative


@register
class ColocalizationFilter(Filter):
    name = "colocalization"
    stage = Stage.GRADED
    modalities = frozenset({"ppi"})

    def __init__(self, annotations: dict[str, set[str]] | None = None,
                 path: str | Path = DEFAULT_LOC_PATH) -> None:
        self._given = annotations
        self._path = path
        self._ann: dict[str, set[str]] = {}

    def fit(self, graph: TypedInteractionGraph) -> None:
        if self._given is not None:
            self._ann = self._given
            return
        try:
            from ..io.localization import load_localization_tsv
            self._ann = load_localization_tsv(self._path)
        except FileNotFoundError:
            self._ann = {}     # no annotations -> abstain everywhere

    def score(self, graph: TypedInteractionGraph, u: str, v: str) -> StreamScore:
        cu, cv = self._ann.get(u), self._ann.get(v)
        if not cu or not cv:
            return StreamScore(self.name, value=None,
                               evidence={"status": "unannotated"})
        shared = cu & cv
        if not shared:
            return StreamScore(
                self.name, value=_SAFE_DISJOINT, flags=["different_compartment"],
                evidence={"compartments_u": sorted(cu), "compartments_v": sorted(cv),
                          "shared": [], "bucket": "disjoint_compartment"},
            )
        jacc = len(shared) / len(cu | cv)
        # more compartment overlap => more plausibly a real interactor => riskier
        value = round(1.0 - 0.5 * jacc, 4)
        return StreamScore(
            self.name, value=value,
            evidence={"compartments_u": sorted(cu), "compartments_v": sorted(cv),
                      "shared": sorted(shared), "jaccard": round(jacc, 4)},
        )
