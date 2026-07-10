"""The filter interface + hourglass stages (see docs/IMPLEMENTATION-PLAN.md).

Every scoring method is a small independent **Filter** that declares which
hourglass stage it runs in and which modalities it applies to, then returns a
score / veto / flags / evidence for a candidate pair. New filters are added by
subclassing this and registering (see registry.py + docs/ADDING-A-FILTER.md) —
no pipeline edits required.

Stages (the hourglass):
  VETO   — cheap hard filters run first; a veto drops the candidate (funnel).
  GRADED — cheap graded filters run in parallel on survivors; scores are merged.
  GATED  — expensive filters (LLM/literature) run only on the contested tail.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from enum import Enum
from typing import Sequence

from ..graph import TypedInteractionGraph
from ..schema import StreamScore


class Stage(str, Enum):
    VETO = "veto"
    GRADED = "graded"
    GATED = "gated"


class Filter(ABC):
    #: unique, stable name (appears in provenance and config)
    name: str
    #: where this filter runs in the hourglass
    stage: Stage = Stage.GRADED
    #: interaction types this filter applies to
    modalities: frozenset = frozenset({"ppi", "pli"})
    #: whether the pipeline includes this filter in the default selection for its
    #: modality. Set False for heavier / experimental filters that must be opted
    #: into by name (they stay registered and buildable — see build_filters).
    default: bool = True
    #: whether this filter supplies the hardness / near-boundary signal that drives
    #: the train-vs-eval split. The orchestrator reads it from whichever GRADED
    #: filter declares this — so the "what makes a negative hard" knowledge lives in
    #: the filter, not the pipeline. Such a filter should expose the magnitude in
    #: evidence["hardness"] (or evidence["risk"]). At most one per run is used.
    provides_hardness: bool = False

    def fit(self, graph: TypedInteractionGraph) -> None:
        """Optional: precompute over the graph before scoring."""

    @abstractmethod
    def score(self, graph: TypedInteractionGraph, u: str, v: str) -> StreamScore:
        ...

    def score_many(
        self, graph: TypedInteractionGraph, pairs: Sequence[tuple[str, str]]
    ) -> list[StreamScore]:
        return [self.score(graph, u, v) for u, v in pairs]


# Back-compat alias: earlier code referred to filters as "streams".
Stream = Filter
