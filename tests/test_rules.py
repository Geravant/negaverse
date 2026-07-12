"""The declarative rule engine + rule-driven filters.

Proves: the `when` grammar is safe, rules abstain on missing fields / wrong types,
co-localization now runs *from YAML* (rules/ppi.yaml), and a bad expression is
rejected at load.

    python -m tests.test_rules
"""
from __future__ import annotations

from negaverse.graph import TypedInteractionGraph
from negaverse.rule_engine import Rule, RuleError, load_rules, bind_env, MissingField
from negaverse.streams import RuleGradedFilter, RuleVetoFilter


def _protein_graph():
    return TypedInteractionGraph.from_edges(
        [("a", "b")], {n: "protein" for n in "abcd"},
        admissible_types=[("protein", "protein")], name="toy")


# --- engine -------------------------------------------------------------
def test_disjoint_predicate_and_binding():
    r = Rule("t", "ppi", ("protein", "protein"),
             "disjoint(a.compartments, b.compartments)", "safer_negative", 0.8).compile()
    env = bind_env(r, {"compartments": {"nucleus"}}, {"compartments": {"mitochondrion"}})
    assert r.evaluate(env) is True
    env2 = bind_env(r, {"compartments": {"nucleus"}}, {"compartments": {"nucleus"}})
    assert r.evaluate(env2) is False


def test_missing_field_raises_for_abstain():
    r = Rule("t", "ppi", ("protein", "protein"),
             "disjoint(a.compartments, b.compartments)", "safer_negative", 0.8).compile()
    env = bind_env(r, {"compartments": {"nucleus"}}, {})   # b has no compartments
    try:
        r.evaluate(env)
        assert False, "expected MissingField"
    except MissingField:
        pass


def test_arithmetic_and_typename_binding():
    r = Rule("t", "pli", ("protein", "ligand"),
             "ligand.volume > protein.pocket_volume * 1.5", "safer_negative", 0.7).compile()
    env = bind_env(r, {"pocket_volume": 100}, {"volume": 200})
    assert env["protein"]["pocket_volume"] == 100 and env["ligand"]["volume"] == 200
    assert r.evaluate(env) is True


def test_unsafe_expression_rejected_at_load():
    for bad in ["__import__('os').system('x')", "a.x.y", "open('f')"]:
        try:
            Rule("t", "ppi", ("protein", "protein"), bad, "safer_negative", 0.5).compile()
            assert False, f"expected RuleError for {bad!r}"
        except RuleError:
            pass


def test_shipped_rules_load_and_compile():
    rules = {r.id for r in load_rules()}
    assert "colocalization_mismatch" in rules


# --- filter -------------------------------------------------------------
def test_colocalization_runs_from_yaml():
    g = _protein_graph()
    ann = {"a": {"compartments": {"nucleus"}}, "b": {"compartments": {"mitochondrion"}}}
    f = RuleGradedFilter(annotations=ann)
    f.fit(g)                                    # loads rules/ppi.yaml
    s = f.score(g, "a", "b")
    assert s.value == 0.85                       # safer_negative weight 0.7 -> 0.85
    assert "different_compartment" in s.flags
    assert s.evidence["fired"][0]["id"] == "colocalization_mismatch"


def test_rule_abstains_on_missing_annotation():
    g = _protein_graph()
    f = RuleGradedFilter(annotations={"a": {"compartments": {"nucleus"}}})  # b unannotated
    f.fit(g)
    assert f.score(g, "a", "b").value is None


def test_wrong_types_do_not_fire():
    # viral/host typed graph: the [protein,protein] rule shouldn't apply
    g = TypedInteractionGraph.from_edges(
        [("x", "y")], {"x": "viral", "y": "host"}, name="vh")
    ann = {"x": {"compartments": {"nucleus"}}, "y": {"compartments": {"mitochondrion"}}}
    f = RuleGradedFilter(annotations=ann)
    f.fit(g)
    assert f.score(g, "x", "y").value is None


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"PASS  {fn.__name__}")
    print(f"\n{len(fns)} checks passed")
