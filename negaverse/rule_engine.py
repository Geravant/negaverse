"""Declarative biology-rule engine (see rules/README.md).

Turns a YAML rule's machine-checkable `when` string into a safe, auditable
predicate over per-entity annotations — no `eval()`. The same rule objects feed
the deterministic RuleFilter (evaluates `when`) and the literature/LLM filter
(receives `rationale` as grounding).

`when` grammar (a restricted Python expression, validated at load):
  * boolean ops        a and b, a or b, not a
  * comparisons        <, <=, >, >=, ==, !=   (chained allowed)
  * arithmetic         + - * /                 (on numeric annotation fields)
  * named predicates   disjoint(x, y), overlap(x, y), shared(x, y),
                       jaccard(x, y), contains(x, v)
  * entity fields      <entity>.<field>        (entity ∈ the rule's binding names)
  * literals           numbers, strings, True/False

Entity binding for a matched pair: the two entities are always available
positionally as `a` and `b`; when the rule's `applies_to` types differ they are
*also* bound by type name (e.g. `protein`, `ligand`). A field that is absent from
an entity's annotation record raises MissingField → the rule abstains for that pair.
"""
from __future__ import annotations

import ast
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

RULES_DIR = "rules"


class RuleError(Exception):
    """A rule is malformed (bad schema or disallowed `when` expression)."""


class MissingField(Exception):
    """A referenced annotation field is absent → the rule abstains."""


# --- named predicates (the only callables a `when` may use) -------------
def _as_set(x):
    return x if isinstance(x, (set, frozenset)) else set(x)


_PREDICATES = {
    "disjoint": lambda x, y: len(_as_set(x) & _as_set(y)) == 0,
    "overlap": lambda x, y: len(_as_set(x) & _as_set(y)) > 0,
    "shared": lambda x, y: len(_as_set(x) & _as_set(y)),
    "jaccard": lambda x, y: (len(_as_set(x) & _as_set(y)) / len(_as_set(x) | _as_set(y)))
    if (_as_set(x) | _as_set(y)) else 0.0,
    "contains": lambda x, v: v in _as_set(x),
}

_CMP = {
    ast.Lt: lambda a, b: a < b, ast.LtE: lambda a, b: a <= b,
    ast.Gt: lambda a, b: a > b, ast.GtE: lambda a, b: a >= b,
    ast.Eq: lambda a, b: a == b, ast.NotEq: lambda a, b: a != b,
}
_BINOP = {ast.Add: lambda a, b: a + b, ast.Sub: lambda a, b: a - b,
          ast.Mult: lambda a, b: a * b, ast.Div: lambda a, b: a / b}


def _validate(node: ast.AST) -> None:
    """Reject anything outside the whitelisted grammar (called at load time)."""
    if isinstance(node, ast.BoolOp):
        if not isinstance(node.op, (ast.And, ast.Or)):
            raise RuleError(f"disallowed boolean op {node.op!r}")
        for v in node.values:
            _validate(v)
    elif isinstance(node, ast.UnaryOp):
        if not isinstance(node.op, (ast.Not, ast.USub, ast.UAdd)):
            raise RuleError("only `not` / unary +/- are allowed")
        _validate(node.operand)
    elif isinstance(node, ast.BinOp):
        if type(node.op) not in _BINOP:
            raise RuleError(f"disallowed arithmetic op {node.op!r}")
        _validate(node.left)
        _validate(node.right)
    elif isinstance(node, ast.Compare):
        for op in node.ops:
            if type(op) not in _CMP:
                raise RuleError(f"disallowed comparison {op!r}")
        _validate(node.left)
        for c in node.comparators:
            _validate(c)
    elif isinstance(node, ast.Call):
        if not isinstance(node.func, ast.Name) or node.func.id not in _PREDICATES:
            raise RuleError(f"unknown predicate: {ast.dump(node.func)}")
        if node.keywords:
            raise RuleError("predicates take positional args only")
        for a in node.args:
            _validate(a)
    elif isinstance(node, ast.Attribute):
        if not isinstance(node.value, ast.Name):
            raise RuleError("field access must be <entity>.<field>")
    elif isinstance(node, ast.Constant):
        if not isinstance(node.value, (int, float, str, bool)):
            raise RuleError(f"disallowed literal {node.value!r}")
    else:
        raise RuleError(f"disallowed expression: {ast.dump(node)}")


def _eval(node: ast.AST, env: dict[str, dict]) -> Any:
    if isinstance(node, ast.BoolOp):
        vals = (_eval(v, env) for v in node.values)
        return all(vals) if isinstance(node.op, ast.And) else any(vals)
    if isinstance(node, ast.UnaryOp):
        val = _eval(node.operand, env)
        if isinstance(node.op, ast.Not):
            return not val
        return -val if isinstance(node.op, ast.USub) else +val
    if isinstance(node, ast.BinOp):
        return _BINOP[type(node.op)](_eval(node.left, env), _eval(node.right, env))
    if isinstance(node, ast.Compare):
        left = _eval(node.left, env)
        for op, comp in zip(node.ops, node.comparators):
            right = _eval(comp, env)
            if not _CMP[type(op)](left, right):
                return False
            left = right
        return True
    if isinstance(node, ast.Call):
        return _PREDICATES[node.func.id](*[_eval(a, env) for a in node.args])
    if isinstance(node, ast.Attribute):
        entity = node.value.id
        if entity not in env:
            raise RuleError(f"unknown entity '{entity}' (have {sorted(env)})")
        rec = env[entity]
        if node.attr not in rec:
            raise MissingField(f"{entity}.{node.attr}")
        return rec[node.attr]
    if isinstance(node, ast.Constant):
        return node.value
    raise RuleError(f"disallowed expression: {ast.dump(node)}")


@dataclass
class Rule:
    id: str
    modality: str
    applies_to: tuple[str, str]
    when: str
    effect: str            # safer_negative | riskier_negative | veto
    weight: float
    rationale: str = ""
    source: str = ""
    flag: str | None = None
    _ast: ast.AST = field(default=None, repr=False)

    _EFFECTS = {"safer_negative", "riskier_negative", "veto"}

    def compile(self) -> "Rule":
        try:
            tree = ast.parse(self.when, mode="eval").body
        except SyntaxError as e:
            raise RuleError(f"rule {self.id!r}: cannot parse `when`: {e}") from e
        _validate(tree)
        self._ast = tree
        if self.effect not in self._EFFECTS:
            raise RuleError(f"rule {self.id!r}: bad effect {self.effect!r}")
        return self

    def evaluate(self, env: dict[str, dict]) -> bool:
        """True if the condition fires. Raises MissingField if an annotation is
        absent (caller treats that as abstain)."""
        return bool(_eval(self._ast, env))


_RULE_FILES = ("ppi.yaml", "pli.yaml")


def load_rules(rules_dir: str | Path = RULES_DIR) -> list[Rule]:
    """Load and validate every rule in `rules_dir/{ppi,pli}.yaml`.

    Scoped to these two named files (not `rules_dir/*.yaml`) because `rules/`
    also holds non-rule manifests (e.g. `sources.yaml`, a different schema
    entirely for `KnownPositiveVeto` — see `rules/SOURCES.md`).
    """
    d = Path(rules_dir)
    rules: list[Rule] = []
    for name in _RULE_FILES:
        path = d / name
        if not path.exists():
            continue
        for raw in (yaml.safe_load(path.read_text()) or []):
            try:
                r = Rule(
                    id=raw["id"], modality=raw["modality"],
                    applies_to=tuple(raw["applies_to"]), when=raw["when"],
                    effect=raw["effect"], weight=float(raw.get("weight", 0.0)),
                    rationale=raw.get("rationale", ""), source=raw.get("source", ""),
                    flag=raw.get("flag"),
                ).compile()
            except KeyError as e:
                raise RuleError(f"{path.name}: rule missing field {e}") from e
            rules.append(r)
    return rules


def bind_env(rule: Rule, rec_first: dict, rec_second: dict) -> dict[str, dict]:
    """Positional a/b always; type-name aliases when the two types differ."""
    env = {"a": rec_first, "b": rec_second}
    t1, t2 = rule.applies_to
    if t1 != t2:
        env[t1] = rec_first
        env[t2] = rec_second
    return env
