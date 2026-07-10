"""End-to-end verification that the system works with the YAML biology rules
(incl. the topology/evolutionary rules added in rules/ppi.yaml).

Two runs:
  A. Integration — the full registry pipeline (which auto-includes the rule
     filters) on the real SARS-CoV-2 graph. Confirms nothing crashes and the
     rule filters are staged; the new rules abstain where their annotations /
     types don't apply.
  B. Firing — a protein-typed host graph with real GO cellular-component
     annotations plus graph-derived fields (neighbors/degree/graph_two_m), so
     Lucy's co-localization AND topology rules actually fire and their verdicts
     reach the emitted records (score + flag in provenance).

    PYTHONPATH=. python scripts/verify_rules_e2e.py
"""
from __future__ import annotations

from negaverse import run_pipeline, PipelineConfig
from negaverse.graph import TypedInteractionGraph
from negaverse.io import load_sars_cov2_graph, load_localization_tsv
from negaverse.streams import (KnownPositiveVeto, StructuredStream, TopologyFilter,
                               RuleGradedFilter, RuleVetoFilter, build_filters)


def run_a_integration():
    print("=" * 70)
    print("RUN A — integration: full registry pipeline on SARS (rules loaded)")
    print("=" * 70)
    g = load_sars_cov2_graph()
    res = run_pipeline(g, PipelineConfig(n_eval=100, n_train=100, match_on_type="viral"))
    f = res.stats["filters"]
    print(f"  filters staged: veto={f['veto']}  graded={f['graded']}  gated={f['gated']}")
    assert "rules" in f["graded"], "rules filter not in graded stage!"
    assert "rule_veto" in f["veto"], "rule_veto filter not in veto stage!"
    assert res.records, "no records emitted!"
    print(f"  emitted {len(res.records)} records; pipeline completed with rules active. OK")


def run_b_firing():
    print("\n" + "=" * 70)
    print("RUN B — firing: co-localization + topology rules on a host graph")
    print("=" * 70)
    sars = load_sars_cov2_graph()
    # homogeneous host graph (hosts re-typed 'protein' so [protein,protein] rules apply)
    hosts = set(sars.nodes_of_type("host"))
    edges = [(u, v) for u, v in sars.g.edges() if u in hosts and v in hosts]
    g = TypedInteractionGraph.from_edges(
        edges, {h: "protein" for h in hosts},
        admissible_types=[("protein", "protein")], name="sars-host")
    print(f"  host graph: {g.g.number_of_nodes()} proteins, {g.g.number_of_edges()} edges")

    # annotations: GO cellular-component + graph-derived topology fields
    loc = load_localization_tsv()
    two_m = 2 * g.g.number_of_edges()
    ann = {}
    for n in g.g.nodes():
        rec = {"neighbors": set(g.g.neighbors(n)), "degree": g.g.degree(n),
               "graph_two_m": two_m}
        if n in loc:
            rec["compartments"] = loc[n]
        ann[n] = rec
    have_comp = sum("compartments" in r for r in ann.values())
    print(f"  annotations: {have_comp} proteins with GO compartments + graph fields on all")

    filters = [KnownPositiveVeto(), StructuredStream(), TopologyFilter(),
               RuleVetoFilter(annotations=ann), RuleGradedFilter(annotations=ann)]
    res = run_pipeline(g, PipelineConfig(modality="ppi", n_eval=150, n_train=150,
                                         max_pool=50_000), filters=filters)

    fired = {}
    with_rule_score = 0
    for r in res.records:
        if r.streams.get("rules") is not None:
            with_rule_score += 1
        for fl in r.flags:
            if fl not in ("near_boundary", "suspected_false_negative", "easy_negative"):
                fired[fl] = fired.get(fl, 0) + 1
    print(f"  emitted {len(res.records)} records; {with_rule_score} carry a `rules` sub-score")
    print(f"  rule flags reaching output: {fired}")
    assert with_rule_score > 0, "no record carries a rules sub-score — rules did not fire!"
    assert fired, "no rule flags propagated to records!"

    # show one firing record end to end
    ex = next(r for r in res.records if r.streams.get("rules") is not None and r.flags)
    print("\n  sample firing record:")
    print(f"    {ex.u} x {ex.v}  mode={ex.mode}  confidence={ex.confidence}")
    print(f"    flags={ex.flags}")
    print(f"    stream_rules={ex.streams.get('rules')}  "
          f"stream_topology={ex.streams.get('topology')}")
    print("  OK — Lucy's rules fire and flow to the emitted records.")


if __name__ == "__main__":
    run_a_integration()
    run_b_firing()
    print("\nALL END-TO-END CHECKS PASSED")
