"""Phase 0: the filter plugin layer + hourglass staging.

Proves a new filter flows through the pipeline with no pipeline edits, and that
the three stages behave (veto drops, graded merges, gated runs on the tail).

    python -m tests.test_filters
"""
from __future__ import annotations

from negaverse import run_pipeline, PipelineConfig
from negaverse.schema import StreamScore
from negaverse.streams import Filter, Stage, register, build_filters, registered
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
