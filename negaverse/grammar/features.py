"""Sparse proof fingerprint — branch-separated evidence, never one blob.

For a candidate pair we assemble the productions the streams already fired into a
canonical, branch-separated feature vector (the flat analogue of the proof
forest):

  SAFE          per safe-effect rule: fired? + value; structured / topology as
                safe-direction evidence.
  RISK          topology risk + hardness (the positive-likeness signals).
  COVERAGE      how much evidence was actually available (counts + missingness) —
                a negative label backed by two of five possible sources is less
                defensible than one backed by all five.
  CONTRADICTION safe evidence AND risk evidence both firing strongly at once.

The point is separation: the shipped fused score averages SAFE and RISK into one
number, which a safety model cannot undo. The fingerprint keeps them apart so a
classifier can learn e.g. "safe-rule fired AND low risk AND high coverage" as a
distinct, defensible pattern.

Operates on a scored-pool row (the corrected bench's `_score_pool` dict):
`{struct, topo, risk, hardness, rules: {rule_id: (weight, value)}}`. Branch
membership is derived from each rule's `effect` (safer_/riskier_negative), so it
generalises when risk rules are added. `rules` here is the rule metadata list
(objects with `.id`, `.effect`), not the row's fired dict.

Coverage note: the row only records rules that fired *True* (the per-rule filter
returns a value only then), so at this rung "missing" conflates "did not apply"
with "abstained for lack of annotation". The full proof forest will separate
them; the sparse fingerprint does not.
"""
from __future__ import annotations

import numpy as np

_SAFE = "safer_negative"


def proof_fingerprint(row: dict, rules) -> dict:
    """Branch-separated fingerprint dict for one scored-pool row."""
    fired = row.get("rules", {})
    struct, topo = row.get("struct"), row.get("topo")
    risk = float(row.get("risk", 0.0) or 0.0)
    hardness = float(row.get("hardness", 0.0) or 0.0)
    f: dict = {}

    safe_vals, risk_vals = [], [risk, hardness]
    n_safe_fired = n_risk_fired = 0
    for r in rules:
        br = "safe" if r.effect == _SAFE else "risk"
        wv = fired.get(r.id)
        hit = wv is not None
        val = float(wv[1]) if hit else 0.0
        f[f"{br}:rule:{r.id}:fired"] = 1.0 if hit else 0.0
        f[f"{br}:rule:{r.id}:val"] = val
        if hit:
            (safe_vals if br == "safe" else risk_vals).append(val)
            if br == "safe":
                n_safe_fired += 1
            else:
                n_risk_fired += 1

    # structured / topology as safe-direction evidence (value = confidence-it's-a-TN)
    f["safe:structured:present"] = 0.0 if struct is None else 1.0
    f["safe:structured:val"] = 0.0 if struct is None else float(struct)
    f["safe:topology:present"] = 0.0 if topo is None else 1.0
    f["safe:topology:val"] = 0.0 if topo is None else float(topo)
    if struct is not None:
        safe_vals.append(float(struct))
    if topo is not None:
        safe_vals.append(float(topo))

    # risk branch: topology's positive-likeness signals
    f["risk:topology_risk"] = risk
    f["risk:hardness"] = hardness

    # coverage / missingness
    n_safe_rules = sum(1 for r in rules if r.effect == _SAFE)
    n_present = n_safe_fired + n_risk_fired + int(struct is not None) + int(topo is not None)
    f["cov:n_present"] = float(n_present)
    f["cov:n_safe_rules_missing"] = float(n_safe_rules - n_safe_fired)
    f["cov:frac_safe_rules_fired"] = (n_safe_fired / n_safe_rules) if n_safe_rules else 0.0

    # contradiction: strongest safe evidence and strongest risk evidence together
    safe_strength = max(safe_vals) if safe_vals else 0.0
    risk_strength = max(risk_vals) if risk_vals else 0.0
    f["contradiction:product"] = safe_strength * risk_strength
    f["contradiction:flag"] = 1.0 if (safe_strength > 0.6 and risk_strength > 0.6) else 0.0
    return f


def fingerprint_names(rules) -> list[str]:
    """Stable feature order — the keys `proof_fingerprint` emits, deterministically."""
    probe = proof_fingerprint({"rules": {}}, rules)
    return list(probe.keys())


def fingerprint_matrix(rows, rules):
    """(X, names): stack `proof_fingerprint` over rows in a fixed column order."""
    names = fingerprint_names(rules)
    X = np.array([[proof_fingerprint(r, rules).get(n, 0.0) for n in names] for r in rows],
                 dtype=float)
    return X, names
