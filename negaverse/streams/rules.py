"""Rule-driven filters — YAML biology rules become filters, no code (Phase 1).

This is what makes "add a rule = add a YAML entry" literally true. Every rule in
rules/*.yaml is evaluated here; co-localization is now just the
`colocalization_mismatch` rule, not hand-coded Python.

  * RuleGradedFilter (GRADED, name="rules") — every safer/riskier rule that
    applies to a pair (by `applies_to` vs the pair's node types) contributes; the
    firing rules are combined by weight into one graded sub-score.
  * RuleVetoFilter (VETO, name="rule_veto") — any firing `veto` rule drops the pair.

A rule abstains for a pair when its types don't match or an annotation field is
missing, so an unpopulated rule (or missing annotations) is simply silent.
"""
from __future__ import annotations

from pathlib import Path

from ..graph import TypedInteractionGraph
from ..rule_engine import MissingField, Rule, RULES_DIR, bind_env, load_rules
from ..schema import StreamScore
from .base import Filter, Stage
from .registry import register


def _effect_value(effect: str, weight: float) -> float:
    # confidence-it's-a-true-negative: safer pushes above 0.5, riskier below
    if effect == "safer_negative":
        return round(0.5 + 0.5 * weight, 4)
    return round(0.5 - 0.5 * weight, 4)      # riskier_negative


class _RuleFilterBase(Filter):
    _effects: frozenset = frozenset()

    def __init__(self, rules: list[Rule] | None = None,
                 rules_dir: str | Path = RULES_DIR,
                 annotations: dict[str, dict] | None = None) -> None:
        self._rules_arg = rules
        self._rules_dir = rules_dir
        self._ann_arg = annotations
        self._rules: list[Rule] = []
        self._ann: dict[str, dict] = {}

    def fit(self, graph: TypedInteractionGraph) -> None:
        rules = self._rules_arg if self._rules_arg is not None else load_rules(self._rules_dir)
        self._rules = [r for r in rules if r.effect in self._effects]
        if self._ann_arg is not None:
            self._ann = self._ann_arg
        else:
            from ..io.annotations import build_annotation_table
            self._ann = build_annotation_table()

    def _applicable(self, graph: TypedInteractionGraph, u: str, v: str):
        """Yield (rule, rec_first, rec_second) for rules whose applies_to matches
        this pair's node types, binding entities in the declared order."""
        tu, tv = graph.node_type.get(u), graph.node_type.get(v)
        ru, rv = self._ann.get(u, {}), self._ann.get(v, {})
        for rule in self._rules:
            t1, t2 = rule.applies_to
            if tu == t1 and tv == t2:
                yield rule, ru, rv
            elif tu == t2 and tv == t1:
                yield rule, rv, ru


@register
class RuleVetoFilter(_RuleFilterBase):
    name = "rule_veto"
    stage = Stage.VETO
    _effects = frozenset({"veto"})

    def score(self, graph: TypedInteractionGraph, u: str, v: str) -> StreamScore:
        for rule, r1, r2 in self._applicable(graph, u, v):
            try:
                if rule.evaluate(bind_env(rule, r1, r2)):
                    return StreamScore(self.name, value=None, veto=True,
                                       flags=[rule.flag or rule.id],
                                       evidence={"rule": rule.id, "rationale": rule.rationale})
            except MissingField:
                continue
        return StreamScore(self.name, value=None)


@register
class RuleGradedFilter(_RuleFilterBase):
    name = "rules"
    stage = Stage.GRADED
    _effects = frozenset({"safer_negative", "riskier_negative"})

    def score(self, graph: TypedInteractionGraph, u: str, v: str) -> StreamScore:
        fired = []
        for rule, r1, r2 in self._applicable(graph, u, v):
            try:
                if rule.evaluate(bind_env(rule, r1, r2)):
                    fired.append((rule, _effect_value(rule.effect, rule.weight)))
            except MissingField:
                continue
        if not fired:
            return StreamScore(self.name, value=None,
                               evidence={"status": "no_rule_fired"})
        num = sum(r.weight * val for r, val in fired)
        den = sum(r.weight for r, _ in fired) or len(fired)
        value = round(num / den, 4)
        flags = [r.flag or r.id for r, _ in fired]
        evidence = {"fired": [{"id": r.id, "effect": r.effect,
                               "value": val, "rationale": r.rationale}
                              for r, val in fired]}
        return StreamScore(self.name, value=value, flags=flags, evidence=evidence)
