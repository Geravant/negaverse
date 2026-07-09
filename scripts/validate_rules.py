"""Validate rules/*.yaml and report which rules are ready vs. will abstain.

    PYTHONPATH=. python scripts/validate_rules.py

Exits non-zero if any rule fails to load/compile (bad schema or an unsafe/invalid
`when` expression) — safe to use as a pre-commit / CI check. For rules that load,
it lists the annotation fields each needs and whether they are currently populated
(a missing field means the rule silently abstains, which is allowed but worth
seeing).
"""
from __future__ import annotations

import ast
import sys

from negaverse.rule_engine import load_rules
from negaverse.io.annotations import build_annotation_table


def _referenced_fields(rule) -> set[tuple[str, str]]:
    return {(n.value.id, n.attr) for n in ast.walk(rule._ast)
            if isinstance(n, ast.Attribute) and isinstance(n.value, ast.Name)}


def main() -> int:
    try:
        rules = load_rules()
    except Exception as e:                       # RuleError etc.
        print(f"FAIL  rules did not load: {e}")
        return 1

    table = build_annotation_table()
    available: set[str] = set()
    for rec in table.values():
        available |= set(rec)

    print(f"loaded {len(rules)} rule(s); {len(table)} annotated node(s); "
          f"available fields: {sorted(available) or '(none)'}\n")
    ready = 0
    for r in rules:
        fields = _referenced_fields(r)
        missing = sorted(f for _, f in fields if f not in available)
        status = "READY " if not missing else "abstain"
        if not missing:
            ready += 1
        note = "" if not missing else f"  (missing: {', '.join(missing)})"
        print(f"  [{status}] {r.id:32} {r.modality:4} {r.effect:16}{note}")
    print(f"\n{ready}/{len(rules)} rule(s) ready with current annotations; "
          f"all rules parse and are safe.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
