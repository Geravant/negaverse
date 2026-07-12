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
