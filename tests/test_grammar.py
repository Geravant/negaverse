"""Sparse proof fingerprint: branch separation, coverage, contradiction."""
import numpy as np

from negaverse.grammar.features import fingerprint_matrix, fingerprint_names, proof_fingerprint


class _Rule:
    def __init__(self, rid, effect, weight=1.0):
        self.id, self.effect, self.weight = rid, effect, weight


RULES = [_Rule("colo", "safer_negative"), _Rule("hydro", "safer_negative"),
         _Rule("risky", "riskier_negative")]


def test_safe_and_risk_never_merged():
    # a safe rule fired AND high topology risk: both must appear as SEPARATE
    # features, not averaged into one value.
    row = {"struct": 0.9, "topo": 0.8, "risk": 0.9, "hardness": 0.9,
           "rules": {"colo": (0.7, 1.0)}}
    f = proof_fingerprint(row, RULES)
    assert f["safe:rule:colo:fired"] == 1.0 and f["safe:rule:colo:val"] == 1.0
    assert f["risk:topology_risk"] == 0.9 and f["risk:hardness"] == 0.9
    # the safe value and the risk value are distinct keys, both preserved
    assert f["safe:rule:colo:val"] != f["risk:topology_risk"]


def test_coverage_counts_missingness():
    full = proof_fingerprint({"struct": 0.5, "topo": 0.5, "risk": 0.1,
                              "rules": {"colo": (0.7, 1.0), "hydro": (0.8, 1.0)}}, RULES)
    sparse = proof_fingerprint({"struct": None, "topo": None, "risk": 0.1, "rules": {}}, RULES)
    assert full["cov:n_present"] > sparse["cov:n_present"]
    assert sparse["cov:n_safe_rules_missing"] == 2.0        # both safe rules absent
    assert full["cov:frac_safe_rules_fired"] > sparse["cov:frac_safe_rules_fired"]


def test_contradiction_flag():
    contra = proof_fingerprint({"struct": 0.9, "topo": 0.9, "risk": 0.9, "hardness": 0.9,
                                "rules": {"colo": (0.7, 0.9)}}, RULES)
    clean = proof_fingerprint({"struct": 0.9, "topo": 0.9, "risk": 0.05, "hardness": 0.05,
                               "rules": {"colo": (0.7, 0.9)}}, RULES)
    assert contra["contradiction:flag"] == 1.0             # strong safe AND strong risk
    assert clean["contradiction:flag"] == 0.0
    assert contra["contradiction:product"] > clean["contradiction:product"]


def test_matrix_shape_and_stable_order():
    rows = [{"struct": 0.5, "topo": 0.5, "risk": 0.2, "rules": {}},
            {"struct": None, "topo": 0.9, "risk": 0.8, "rules": {"colo": (0.7, 1.0)}}]
    names = fingerprint_names(RULES)
    X, cols = fingerprint_matrix(rows, RULES)
    assert cols == names
    assert X.shape == (2, len(names))
    assert not np.isnan(X).any()


def test_ledger_distinguishes_false_from_missing():
    """The four-state upgrade: a rule with annotations present but condition unmet is
    FALSE; a rule whose annotation is absent is MISSING — never conflated."""
    from negaverse.graph import TypedInteractionGraph
    from negaverse.grammar.ledger import LedgerScorer, TRUE, FALSE, MISSING
    from negaverse.rule_engine import Rule

    # one rule keyed on a 'compartments' field; fire when compartments are disjoint
    rule = Rule(id="loc_test", modality="ppi", applies_to=("protein", "protein"),
                when="disjoint(a.compartments, b.compartments)", effect="safer_negative",
                weight=0.7, rationale="", source="test").compile()
    g = TypedInteractionGraph.from_edges(
        [("a", "x"), ("b", "x")], {n: "protein" for n in ("a", "b", "x")},
        admissible_types=[("protein", "protein")], name="t")
    ann = {"a": {"compartments": {"nucleus"}}, "b": {"compartments": {"cytosol"}},
           "x": {"compartments": {"nucleus"}}}                     # 'm' has NO compartments
    sc = LedgerScorer(rules=[rule], annotations=ann, pair_annotations={}).fit(g)

    def outcome(u, v, name):
        return next(t.outcome for t in sc.ledger(g, u, v) if t.name == name)

    assert outcome("a", "b", "loc_test") == TRUE                   # disjoint compartments
    assert outcome("a", "x", "loc_test") == FALSE                  # both present, overlap → not fired
    # a node with no compartments annotation → MISSING, not FALSE
    g2 = TypedInteractionGraph.from_edges(
        [("a", "b"), ("m", "b")], {n: "protein" for n in ("a", "b", "m")},
        admissible_types=[("protein", "protein")], name="t2")
    sc2 = LedgerScorer(rules=[rule], annotations={"a": {"compartments": {"nucleus"}}},
                       pair_annotations={}).fit(g2)
    assert next(t.outcome for t in sc2.ledger(g2, "a", "m") if t.name == "loc_test") == MISSING


def test_ledger_encodings_grammar_superset_of_flat():
    from negaverse.grammar.ledger import Terminal, TRUE, FALSE, RISK, SAFE
    from negaverse.grammar.features import ledger_flat_features, ledger_grammar_features
    lg = [Terminal("colocalization_mismatch", TRUE, SAFE, 0.7),
          Terminal("RISK_L3", TRUE, RISK, 0.9)]
    flat, gram = ledger_flat_features(lg), ledger_grammar_features(lg)
    assert set(flat).issubset(set(gram))                          # grammar adds columns
    assert gram["comp:contradiction"] == 1.0                      # safe TRUE and risk TRUE
    assert gram["comp:n_safe_true"] == 1.0 and gram["comp:n_risk_true"] == 1.0
