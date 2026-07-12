"""Four-state evidence ledger — the representation Exp-1's fingerprint lacked.

Exp-1 (features.proof_fingerprint) conflated "rule evaluated FALSE" with "rule
abstained because an annotation was MISSING". A learned grammar that can't tell
them apart learns *annotation availability* (a data artifact), not biology. Here
every terminal keeps a four-state outcome and provenance:

    outcome  ∈ {TRUE, FALSE, MISSING, NOT_APPLICABLE}
    effect   ∈ {SAFE, RISK, COVERAGE}
    strength ∈ [0,1]   magnitude when resolved
    quality  ∈ [0,1]   evidence quality (source-dependent)
    source              provenance string

Terminals come from two families:
  * RULE terminals — the YAML rules, re-evaluated to distinguish FALSE (annotation
    present, condition not met) from MISSING (annotation absent → MissingField).
    Reuses the rule engine's annotation/pair/graph merging (`_RuleFilterBase`).
  * GRAPH terminals — genuinely relational risk productions computed from the
    training graph: RISK_L3 (∃ length-3 path A–X–Y–B) and RISK_COMMON_NEIGHBOR.
    These are new compositional structure, not a pre-aggregated topology scalar.

Data-gated productions (RISK_SAME_COMPLEX from CORUM, RISK_PPI_MODEL from a
cross-fitted D-SCRIPT/ESM2 score) are declared below but emit MISSING until that
data is loaded — the vocabulary is extensible without faking absent evidence.
"""
from __future__ import annotations

from dataclasses import dataclass

from ..rule_engine import MissingField, RULES_DIR, bind_env, load_rules
from ..streams.rules import _RuleFilterBase

TRUE, FALSE, MISSING, NA = "TRUE", "FALSE", "MISSING", "NOT_APPLICABLE"
SAFE, RISK, COVERAGE = "SAFE", "RISK", "COVERAGE"


@dataclass
class Terminal:
    name: str
    outcome: str
    effect: str
    strength: float = 0.0
    quality: float = 1.0
    source: str = ""


class _LedgerRules(_RuleFilterBase):
    """Reuse the rule engine's annotation + pair + graph merging via _applicable,
    but keep every effect (safe/risk/veto), not just one."""
    name = "ledger_rules"
    _effects = frozenset({"safer_negative", "riskier_negative", "veto"})

    def score(self, graph, u, v):        # unused: the ledger calls _applicable directly
        raise NotImplementedError


class LedgerScorer:
    """Build the four-state evidence ledger for a pair. `fit(graph)` loads rules and
    annotations once; `ledger(graph, u, v)` returns the terminal list."""

    def __init__(self, rules=None, rules_dir=RULES_DIR, annotations=None,
                 pair_annotations=None, l3_cap: int = 400):
        self._rules = _LedgerRules(rules=rules, rules_dir=rules_dir,
                                   annotations=annotations, pair_annotations=pair_annotations)
        self.l3_cap = l3_cap

    def fit(self, graph):
        self._rules.fit(graph)
        return self

    def _rule_terminals(self, graph, u, v):
        out = []
        for rule, r1, r2 in self._rules._applicable(graph, u, v):
            effect = SAFE if rule.effect == "safer_negative" else (
                RISK if rule.effect == "riskier_negative" else "VETO")
            try:
                fired = bool(rule.evaluate(bind_env(rule, r1, r2)))
                outcome = TRUE if fired else FALSE
                strength = float(rule.weight) if fired else 0.0
            except MissingField:
                outcome, strength = MISSING, 0.0
            out.append(Terminal(rule.id, outcome, effect, strength, 1.0, rule.source or ""))
        return out

    def _graph_terminals(self, graph, u, v):
        g = graph.g
        if not (g.has_node(u) and g.has_node(v)):
            return [Terminal("RISK_COMMON_NEIGHBOR", MISSING, RISK, source="graph"),
                    Terminal("RISK_L3", MISSING, RISK, source="graph")]
        Nu, Nv = set(g.neighbors(u)), set(g.neighbors(v))
        if not Nu or not Nv:                       # an isolated endpoint → can't assess relational risk
            return [Terminal("RISK_COMMON_NEIGHBOR", MISSING, RISK, source="graph"),
                    Terminal("RISK_L3", MISSING, RISK, source="graph")]
        cn = len(Nu & Nv)
        cn_strength = cn / (1.0 + cn)              # squashed count in [0,1)
        # L3: edges between a neighbour of u and a neighbour of v (length-3 path)
        small = Nu if len(Nu) <= len(Nv) else Nv
        big = Nv if small is Nu else Nu
        l3 = 0
        for i, x in enumerate(small):
            if i >= self.l3_cap:
                break
            l3 += len(set(g.neighbors(x)) & big)
        l3_strength = l3 / (1.0 + l3)
        return [
            Terminal("RISK_COMMON_NEIGHBOR", TRUE if cn else FALSE, RISK, cn_strength, 1.0, "graph"),
            Terminal("RISK_L3", TRUE if l3 else FALSE, RISK, l3_strength, 1.0, "graph"),
        ]

    def ledger(self, graph, u, v):
        return self._rule_terminals(graph, u, v) + self._graph_terminals(graph, u, v)


#: declared productions with real data today; the vocabulary the fingerprint reads.
LEDGER_TERMINALS = [
    ("colocalization_mismatch", SAFE), ("hydrophobicity_interface", SAFE),
    ("string_low_confidence_non_interaction", SAFE),
    ("RISK_COMMON_NEIGHBOR", RISK), ("RISK_L3", RISK),
]


def default_ledger_scorer(graph):
    rules = [r for r in load_rules() if r.modality == "ppi"]
    return LedgerScorer(rules=rules).fit(graph)
