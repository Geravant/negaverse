# Structural / biological rules

Declarative rules that feed negaverse's filters. A rule is added by editing a
YAML file here — no code — which keeps the "dynamic, up-to-date reasoning" claim
real and lets the biology side (rules) evolve independently of the engine.

- **Writing a rule by hand:** [`AUTHORING.md`](AUTHORING.md) — the step-by-step,
  with weight calibration and a validator (`scripts/validate_rules.py`).
- **Writing a rule from a paper:** the `rule-from-literature` Claude skill
  (`.claude/skills/rule-from-literature/`) — give it a PDF/URL/abstract and it
  drafts, validates, and appends the rule.
- **Registering an external positive-interaction database** (for the
  known-positive veto, not the rule engine): [`SOURCES.md`](SOURCES.md) /
  `sources.yaml` — a different mechanism from the rules below; see that doc
  for why.

One consumer, two uses:
- **Deterministic filters** evaluate the machine-checkable `when` condition
  against entity annotations and apply `effect`/`weight` to the graded score.
- **The literature / LLM filter** receives the `rationale` text verbatim as
  grounding context, so its judgement cites the same rules rather than its own
  priors.

## Field contract

| field | required | meaning |
|---|---|---|
| `id` | ✓ | unique, stable slug (used in provenance) |
| `modality` | ✓ | `ppi` or `pli` |
| `applies_to` | ✓ | ordered entity types the rule pairs, e.g. `[protein, ligand]` |
| `when` | ✓ | machine-checkable condition over entity annotations `a` / `b` (protein) or `ligand` (see below) |
| `effect` | ✓ | `safer_negative` \| `riskier_negative` \| `veto` |
| `weight` | ✓ | contribution to the graded score, `[0,1]` (ignored for `veto`) |
| `rationale` | ✓ | natural-language justification fed to the LLM |
| `source` | – | provenance / citation (use `TODO` while sourcing) |
| `flag` | – | short tag added to a record when the rule fires (defaults to `id`) |

## `when` expression language (implemented)

A restricted, **safe** Python expression — parsed and whitelisted via `ast`, never
`eval()`ed (`negaverse/rule_engine.py`). Allowed: boolean ops, comparisons,
arithmetic, the named predicates `disjoint / overlap / shared / jaccard / contains`,
`<entity>.<field>` access, and literals. Anything else is rejected at load.

**Entity binding.** The two entities of a matched pair are always available as `a`
and `b` (positional, in `applies_to` order); when the two `applies_to` types differ
they are also bound by type name — so `pli` rules can say `protein.pocket_volume`
and `ligand.volume`. A field absent from an entity's record → the rule abstains.

**Annotations** (`negaverse/io/annotations.py`) are `dict[node -> dict[field -> value]]`,
merged from whatever sources exist (currently `compartments` from GO cellular-component).
Add an annotation type = load it there under a new field name.

## Status

The generic loader + evaluator + `RuleGradedFilter`/`RuleVetoFilter` are **built**:
every rule here becomes a filter automatically, no code. `colocalization_mismatch`
is live (needs GO cellular-component annotations); the other rules are valid
templates whose `TODO` annotation fields simply make them abstain until sourced.
