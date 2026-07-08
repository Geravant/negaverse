"""The scoring-stream interface (ARCHITECTURE.md §5).

Confidence is a fusion of independent views. Keeping streams behind one
interface means each can be validated, ablated, and trusted on its own — and
new streams (graph embeddings, KG link-prediction) drop in without touching the
pipeline. A stream may score, abstain, or hard-veto.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Sequence

from ..graph import TypedInteractionGraph
from ..schema import StreamScore


class Stream(ABC):
    name: str

    def fit(self, graph: TypedInteractionGraph) -> None:
        """Optional: precompute over the graph before scoring (e.g. embeddings)."""

    @abstractmethod
    def score(self, graph: TypedInteractionGraph, u: str, v: str) -> StreamScore:
        ...

    def score_many(
        self, graph: TypedInteractionGraph, pairs: Sequence[tuple[str, str]]
    ) -> list[StreamScore]:
        return [self.score(graph, u, v) for u, v in pairs]
