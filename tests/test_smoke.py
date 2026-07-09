"""Smoke test for the negaverse walking skeleton.

Runnable two ways:
    pytest tests/
    python -m tests.test_smoke
"""
from __future__ import annotations

from negaverse import run_pipeline, PipelineConfig
from negaverse import eval as ev
from negaverse.io import load_sars_cov2_graph


def _run():
    graph = load_sars_cov2_graph()
    cfg = PipelineConfig(n_eval=200, n_train=200, seed=0, match_on_type="viral")
    return graph, run_pipeline(graph, cfg)


def test_graph_typing():
    graph = load_sars_cov2_graph()
    assert graph.node_type  # non-empty
    virals = graph.nodes_of_type("viral")
    assert 20 <= len(virals) <= 35, f"expected ~27 viral proteins, got {len(virals)}"
    assert graph.n_edges > 300


def test_no_leakage_and_disjoint_split():
    graph, result = _run()
    # Layer 2 invariant: no emitted negative is a known positive
    assert ev.leakage(graph, result.records) == 0
    eval_keys = {(r.u, r.v) for r in result.records if r.mode == "eval"}
    train_keys = {(r.u, r.v) for r in result.records if r.mode == "train"}
    assert eval_keys.isdisjoint(train_keys), "train and eval must not overlap (P1)"


def test_admissible_space():
    graph, result = _run()
    for r in result.records:
        types = {graph.node_type[r.u], graph.node_type[r.v]}
        assert types == {"viral", "host"}, f"non-admissible pair emitted: {r.u},{r.v}"


def test_degree_matching_beats_random():
    graph, result = _run()
    eval_records = [r for r in result.records if r.mode == "eval"]
    dm = ev.degree_match(graph, eval_records, match_type="viral", seed=0)
    # eval negatives should match positive viral-degree better than random does
    assert dm["ks_negaverse_vs_positive"] < dm["ks_random_vs_positive"]


def test_train_harder_than_eval():
    graph, result = _run()
    hs = ev.hardness_split(result.records)
    assert hs["train_mean_hardness"] > hs["eval_mean_hardness"], "train must be harder (P2)"


def test_output_contract():
    graph, result = _run()
    r = result.records[0]
    row = r.as_row()
    for key in ("u", "v", "mode", "confidence", "hardness", "provenance"):
        assert key in row
    assert set(r.streams) == {"structured", "literature", "topology", "rules"}
    assert 0.0 <= r.confidence <= 1.0 and 0.0 <= r.hardness <= 1.0


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"PASS  {fn.__name__}")
    print(f"\n{len(fns)} checks passed")
