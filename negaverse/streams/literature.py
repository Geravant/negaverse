"""Literature-reasoning stream (ARCHITECTURE.md §5.2) — STUB.

A modernised Negatome: embed abstracts, retrieve passages about the candidate
pair, and have an LLM extract explicit non-interaction evidence with citations.
Per §8.5 this is gated to the top-K contested / near-boundary pairs at demo
time, not run per-candidate — so in the walking skeleton it abstains on
everything and exists only to hold the seam. `gate()` marks which pairs a real
implementation would spend an LLM call on.
"""
from __future__ import annotations

from ..graph import TypedInteractionGraph
from ..schema import StreamScore
from .base import Stream


class LiteratureStream(Stream):
    name = "literature"

    def score(self, graph: TypedInteractionGraph, u: str, v: str) -> StreamScore:
        return StreamScore(self.name, value=None,
                           evidence={"status": "stub_abstains"})
