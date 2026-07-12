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

**Per-node annotations** (`negaverse/io/annotations.py::build_annotation_table()`) are
`dict[node -> dict[field -> value]]`, merged from whatever sources exist (currently
`compartments` from GO cellular-component, plus `surface_hydrophobicity`,
`pocket_volume`, `pocket_hydrophobicity`, and `pocket_polarity` — computed with
`scripts/compute_surface_hydrophobicity.py` and `scripts/compute_pocket_descriptors.py`
respectively, populated once someone runs them for a given graph's nodes). Add a
per-node annotation type = load it there under a new field name — true for fields
sourced from external files/DBs.

**Pairwise annotations** (`build_pair_annotation_table()`, same file) are a second,
separate mechanism for fields whose value depends on *both* entities in a pair, not
one node alone — e.g. `string_experimental_score_with_b`
(`scripts/compute_string_experimental.py`). These load from
`node_a<TAB>node_b<TAB>value` files and get merged onto the right entity's record
per-pair, at score time, by `negaverse/streams/rules.py::_RuleFilterBase` — not
onto the shared per-node cache, since the same node needs a different value
depending on which partner it's being scored against.

**Graph-structural fields** (`degree`, `neighbors`, `graph_two_m`) are a third kind:
they depend on whichever graph is loaded, so they can't come from
`build_annotation_table()` (which correctly takes no graph argument). Instead,
`negaverse/streams/rules.py::_RuleFilterBase.fit()` merges them in from the live
graph via `_augment_with_graph()` — so rules referencing them genuinely work in the
real pipeline today (not staged/pending). The one place they'll always show as
missing is `scripts/validate_rules.py`, which checks annotations standalone with no
graph to augment with — that's a property of the standalone checker, not of whether
a topology rule works in production.

## Status

The generic loader + evaluator + `RuleGradedFilter`/`RuleVetoFilter` are **built**:
every rule here becomes a filter automatically, no code. 7 rules exist across
`ppi.yaml`/`pli.yaml`; `colocalization_mismatch` is live wherever GO
cellular-component annotations are populated (calibrated: AUROC 0.906/0.875 on
DRYAD, weaker on UPNA-PPI — see its `rationale`). Most of the rest abstain
because their annotation field is genuinely unsourced (`TODO` in `source`) or
simply hasn't been computed yet for the graph in question —
`hydrophobicity_interface` is calibrated and has a real loader
(`scripts/compute_surface_hydrophobicity.py`) but still abstains until someone
runs it for their graph's nodes. `evolutionary_coupling_absence` similarly has
a real, implemented pipeline (`scripts/compute_evolutionary_coupling.py`,
Evolutionary Rate Covariation via RERconverge) but hasn't been calibrated
against gold-standard PPI data yet — its threshold is still a placeholder.
`string_low_confidence_non_interaction` uses STRING's `experimental` channel
(`< 0.15`, graded — `scripts/compute_string_experimental.py`); STRING's
`cooccurence` channel was also tested as an alternative evolutionary-coupling
proxy and found no reliable separation on DRYAD/UPNA-PPI, so it isn't used.
The same `experimental` channel's opposite tail (`> 0.9`, strong direct
evidence) is deliberately
*not* a rule here — it's a known-positive source instead
(`rules/sources.yaml`'s `string_experimental_high_confidence`, built by
`scripts/build_known_positive_sources.py`), since that's a plain "documented
interaction" membership fact for `KnownPositiveVeto`, not a biological-
plausibility judgement — see `rules/SOURCES.md` and `AUTHORING.md` Step 5.
(A topology rule mirroring `TopologyFilter`'s `no_overlap` case,
`no_shared_neighbors_low_expected_edge`, previously existed here and was
removed: `TopologyFilter` already computes that exact signal more rigorously
as an independent stream, so the YAML rule only double-counted it rather than
adding new evidence — see `AUTHORING.md` Step 5.)
