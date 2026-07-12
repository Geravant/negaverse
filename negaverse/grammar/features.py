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


# --- Exp-2: four-state ledger encodings (flat terminals vs compositional grammar) ---
from .ledger import FALSE, LEDGER_TERMINALS, MISSING, RISK, SAFE, TRUE  # noqa: E402


def ledger_flat_features(ledger, terminals=LEDGER_TERMINALS) -> dict:
    """FLAT expanded terminals: per terminal, one-hot four-state outcome + strength.
    No composition — the ablation the grammar fingerprint must beat."""
    d = {t.name: t for t in ledger}
    f: dict = {}
    for name, _effect in terminals:
        t = d.get(name)
        oc = t.outcome if t else MISSING
        f[f"{name}:is_true"] = 1.0 if oc == TRUE else 0.0
        f[f"{name}:is_false"] = 1.0 if oc == FALSE else 0.0
        f[f"{name}:is_missing"] = 1.0 if oc == MISSING else 0.0
        f[f"{name}:strength"] = float(t.strength) if t else 0.0
    return f


def ledger_grammar_features(ledger, terminals=LEDGER_TERMINALS) -> dict:
    """GRAMMAR fingerprint: flat terminals PLUS compositional structure — coverage /
    missingness by effect, and cross-terminal conjunctions (contradiction, clean-safe)
    that a linear model cannot form from flat terminals on its own."""
    f = dict(ledger_flat_features(ledger, terminals))
    d = {t.name: t for t in ledger}
    safe = [d[n] for n, e in terminals if e == SAFE and n in d]
    risk = [d[n] for n, e in terminals if e == RISK and n in d]
    n_true = lambda ts: sum(1 for t in ts if t.outcome == TRUE)
    n_res = lambda ts: sum(1 for t in ts if t.outcome in (TRUE, FALSE))
    n_miss = lambda ts: sum(1 for t in ts if t.outcome == MISSING)
    f["cov:n_safe_resolved"] = float(n_res(safe))
    f["cov:n_safe_missing"] = float(n_miss(safe))
    f["cov:n_risk_resolved"] = float(n_res(risk))
    f["cov:n_risk_missing"] = float(n_miss(risk))
    f["cov:frac_resolved"] = (n_res(safe) + n_res(risk)) / max(len(terminals), 1)
    st, rt = n_true(safe), n_true(risk)
    f["comp:n_safe_true"] = float(st)
    f["comp:n_risk_true"] = float(rt)
    f["comp:contradiction"] = 1.0 if (st > 0 and rt > 0) else 0.0     # safe AND risk both fire
    safe_str = max((t.strength for t in safe if t.outcome == TRUE), default=0.0)
    risk_str = max((t.strength for t in risk if t.outcome == TRUE), default=0.0)
    f["comp:contradiction_strength"] = safe_str * risk_str
    # clean-safe: safe evidence fired AND relational risk was resolved-FALSE (not missing)
    f["comp:clean_safe"] = 1.0 if (st > 0 and rt == 0 and n_res(risk) > 0) else 0.0
    return f


def ledger_matrix(ledgers, mode="grammar", terminals=LEDGER_TERMINALS):
    """(X, names) over ledgers. mode='flat' → terminals only; 'grammar' → + composition."""
    enc = ledger_grammar_features if mode == "grammar" else ledger_flat_features
    names = list(enc([], terminals).keys())
    X = np.array([[enc(lg, terminals).get(n, 0.0) for n in names] for lg in ledgers], dtype=float)
    return X, names
