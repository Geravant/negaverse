"""LiteratureFilter's persistent verdict cache.

Proves the cache key changes when `self.names` changes — a regression test
for a real bug: adding gene names later (or any other context enrichment)
must invalidate previously-cached verdicts, since the LLM sees a materially
different prompt. Before the fix, `_feature_key` only hashed graph-derived
features (types/degrees), never `self.names`, so a pair judged before names
existed would silently keep serving its pre-enrichment verdict forever.

    python -m pytest tests/test_literature.py
"""
from __future__ import annotations

from negaverse.graph import TypedInteractionGraph
from negaverse.streams import LiteratureFilter


def _graph() -> TypedInteractionGraph:
    return TypedInteractionGraph.from_edges(
        edges=[("P1", "P2")],
        node_type={"P1": "protein", "P2": "protein"})


def test_feature_key_changes_when_names_are_added():
    graph = _graph()
    no_names = LiteratureFilter(enabled=False)
    key_before = no_names._feature_key(graph, "P1", "P2")

    with_names = LiteratureFilter(enabled=False, names={"P1": "TP53", "P2": "MDM2"})
    key_after = with_names._feature_key(graph, "P1", "P2")

    assert key_before != key_after


def test_feature_key_stable_when_nothing_changes():
    graph = _graph()
    a = LiteratureFilter(enabled=False, names={"P1": "TP53"})
    b = LiteratureFilter(enabled=False, names={"P1": "TP53"})
    assert a._feature_key(graph, "P1", "P2") == b._feature_key(graph, "P1", "P2")


def test_feature_key_changes_when_a_single_name_is_updated():
    """Not just "has a name" vs "no name" — a corrected/enriched name (e.g.
    symbol -> symbol + full name + synonyms) must also invalidate the cache."""
    graph = _graph()
    old = LiteratureFilter(enabled=False, names={"P1": "TP53"})
    new = LiteratureFilter(enabled=False, names={"P1": "TP53 — Cellular tumor antigen p53 (aka P53)"})
    assert old._feature_key(graph, "P1", "P2") != new._feature_key(graph, "P1", "P2")
