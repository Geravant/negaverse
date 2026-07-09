"""Co-localization filter: the three behaviours (safe / risky / abstain).

    python -m tests.test_colocalization
"""
from __future__ import annotations

from negaverse.graph import TypedInteractionGraph
from negaverse.streams import ColocalizationFilter


def _graph():
    return TypedInteractionGraph.from_edges(
        [("a", "b")], {n: "protein" for n in "abcd"},
        admissible_types=[("protein", "protein")], name="toy")


def test_disjoint_compartments_are_safe_negatives():
    g = _graph()
    f = ColocalizationFilter(annotations={"a": {"nucleus"}, "b": {"mitochondrion"}})
    f.fit(g)
    s = f.score(g, "a", "b")
    assert s.value == 0.9 and "different_compartment" in s.flags


def test_shared_compartment_is_riskier():
    g = _graph()
    f = ColocalizationFilter(annotations={"a": {"cytoplasm"}, "b": {"cytoplasm"}})
    f.fit(g)
    s = f.score(g, "a", "b")
    # identical single compartment -> jaccard 1 -> value 0.5 (not a safe negative)
    assert s.value == 0.5 and not s.flags


def test_partial_overlap_between_safe_and_risky():
    g = _graph()
    f = ColocalizationFilter(annotations={"a": {"cytoplasm", "nucleus"},
                                          "b": {"cytoplasm", "er"}})
    f.fit(g)
    s = f.score(g, "a", "b")
    assert 0.5 < s.value < 0.9        # jaccard 1/3 -> ~0.833

def test_missing_annotation_abstains():
    g = _graph()
    f = ColocalizationFilter(annotations={"a": {"nucleus"}})   # b unannotated
    f.fit(g)
    s = f.score(g, "a", "b")
    assert s.value is None and s.abstains


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"PASS  {fn.__name__}")
    print(f"\n{len(fns)} checks passed")
