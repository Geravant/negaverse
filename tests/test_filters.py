"""Phase 0: the filter plugin layer + hourglass staging.

Proves a new filter flows through the pipeline with no pipeline edits, and that
the three stages behave (veto drops, graded merges, gated runs on the tail).

    python -m tests.test_filters
"""
from __future__ import annotations

from negaverse import run_pipeline, PipelineConfig
from negaverse.graph import TypedInteractionGraph
from negaverse.rule_engine import Rule
from negaverse.schema import StreamScore
from negaverse.streams import Filter, Stage, register, build_filters, registered
from negaverse.streams import RuleGradedFilter, RuleVetoFilter
from negaverse.io import load_sars_cov2_graph


def test_builtin_stages_registered():
    reg = registered()
    assert reg["known_positive_veto"].stage == Stage.VETO
    assert reg["structured"].stage == Stage.GRADED
    assert reg["topology"].stage == Stage.GRADED
    assert reg["literature"].stage == Stage.GATED


def test_veto_filter_drops_candidates():
    """A VETO filter that rejects everything empties the scored pool."""
    class RejectAll(Filter):
        name = "reject_all_test"
        stage = Stage.VETO

        def score(self, graph, u, v):
            return StreamScore(self.name, value=None, veto=True)

    graph = load_sars_cov2_graph()
    result = run_pipeline(graph, PipelineConfig(n_eval=50, n_train=50),
                          filters=[RejectAll()])
    assert result.stats["scored_pool"] == 0
    assert result.records == []


def test_new_graded_filter_flows_through_without_pipeline_edits():
    """Register a custom GRADED filter; it appears in every record's sub-scores."""
    @register
    class ConstFilter(Filter):
        name = "const_test"
        stage = Stage.GRADED
        modalities = frozenset({"ppi"})

        def score(self, graph, u, v):
            return StreamScore(self.name, value=0.5)

    graph = load_sars_cov2_graph()
    # explicit selection so we don't depend on other registered filters
    result = run_pipeline(graph, PipelineConfig(
        n_eval=50, n_train=50, match_on_type="viral",
        filters=["known_positive_veto", "const_test", "topology"]))
    assert result.records
    r = result.records[0]
    assert "const_test" in r.streams and r.streams["const_test"] == 0.5
    assert "const_test" in result.stats["filters"]["graded"]


def _string_low_confidence_rule() -> Rule:
    return Rule(id="string_low_confidence_non_interaction_test", modality="ppi",
                applies_to=("protein", "protein"),
                when="a.string_score_with_b < 0.1",
                effect="safer_negative", weight=0.5).compile()


def test_pairwise_annotation_fires_for_the_right_pair():
    """A pair-keyed field (string_score_with_b) should fire the
    rule for a pair whose score clears the threshold, and abstain for a pair
    that doesn't — proving build_pair_annotation_table's values actually reach
    the rule engine as `a.<field>`."""
    graph = TypedInteractionGraph.from_edges(
        edges=[("P1", "P2")],
        node_type={"P1": "protein", "P2": "protein", "P3": "protein"})
    pair_ann = {"string_score_with_b": {
        frozenset({"P1", "P2"}): 0.05,   # below 0.1 -> rule fires
        frozenset({"P1", "P3"}): 0.9,    # above 0.1 -> rule doesn't fire
    }}
    f = RuleGradedFilter(rules=[_string_low_confidence_rule()], annotations={}, pair_annotations=pair_ann)
    f.fit(graph)

    fired = f.score(graph, "P1", "P2")
    assert fired.value is not None and fired.value > 0.5

    abstained = f.score(graph, "P1", "P3")
    assert abstained.value is None


def test_pairwise_annotation_does_not_leak_across_partners():
    """Scoring (P1, P3) right after (P1, P2) must not corrupt P1's cached
    per-node record — each call's injected pair value must be scoped to that
    exact call, not mutate the shared annotation cache for node P1."""
    graph = TypedInteractionGraph.from_edges(
        edges=[("P1", "P2"), ("P1", "P3")],
        node_type={"P1": "protein", "P2": "protein", "P3": "protein"})
    pair_ann = {"string_score_with_b": {
        frozenset({"P1", "P2"}): 0.05,
        frozenset({"P1", "P3"}): 0.9,
    }}
    f = RuleGradedFilter(rules=[_string_low_confidence_rule()], annotations={}, pair_annotations=pair_ann)
    f.fit(graph)

    f.score(graph, "P1", "P2")            # fires; if this mutated self._ann["P1"]...
    result = f.score(graph, "P1", "P3")   # ...this would incorrectly also fire
    assert result.value is None

    # and P1-P2 must still fire correctly afterward, unaffected by the P1-P3 call
    still_fires = f.score(graph, "P1", "P2")
    assert still_fires.value is not None and still_fires.value > 0.5


def _pairwise_field_veto_rule() -> Rule:
    """A hypothetical veto rule over a pairwise field — no shipped rule in
    ppi.yaml currently uses `effect: veto` with a pairwise field (STRING's
    high-confidence experimental evidence is instead handled as a
    known-positive source, rules/sources.yaml, not a rule-engine rule — see
    that file and rules/SOURCES.md), so this exists purely to prove the
    engine mechanism works, independent of what ships today."""
    return Rule(id="pairwise_veto_test", modality="ppi",
                applies_to=("protein", "protein"),
                when="a.string_experimental_score_with_b > 0.9",
                effect="veto", weight=0.0).compile()


def test_veto_rule_reads_pairwise_annotation():
    """RuleVetoFilter shares _RuleFilterBase with RuleGradedFilter, so a
    pair-keyed field (string_experimental_score_with_b) should reach a veto
    rule's `when` the same way it reaches a graded rule's — proving the
    pairwise-injection mechanism isn't graded-filter-specific."""
    graph = TypedInteractionGraph.from_edges(
        edges=[("P1", "P2")],
        node_type={"P1": "protein", "P2": "protein", "P3": "protein"})
    pair_ann = {"string_experimental_score_with_b": {
        frozenset({"P1", "P2"}): 0.95,   # above 0.9 -> veto fires
        frozenset({"P1", "P3"}): 0.2,    # below 0.9 -> no veto
    }}
    f = RuleVetoFilter(rules=[_pairwise_field_veto_rule()],
                       annotations={}, pair_annotations=pair_ann)
    f.fit(graph)

    vetoed = f.score(graph, "P1", "P2")
    assert vetoed.veto is True
    assert "pairwise_veto_test" in vetoed.flags

    not_vetoed = f.score(graph, "P1", "P3")
    assert not_vetoed.veto is False


def test_build_filters_by_modality():
    ppi = [f.name for f in build_filters("ppi",
           names=["known_positive_veto", "structured", "topology", "literature"])]
    assert ppi == ["known_positive_veto", "structured", "topology", "literature"]


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"PASS  {fn.__name__}")
    print(f"\n{len(fns)} checks passed")
