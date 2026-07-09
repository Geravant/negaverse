# Structural / biological rules

Declarative rules that feed negaverse's filters. A rule is added by editing a
YAML file here — no code — which keeps the "dynamic, up-to-date reasoning" claim
real and lets the biology side (rules) evolve independently of the engine.

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

## Two open choices for the loader (Phase 1)

1. **`when` expression language.** Prefer a **fixed set of named predicates**
   (`disjoint(...)`, comparisons over declared annotation fields) over a general
   `eval()` — avoids an eval sandbox and keeps rules auditable. The examples use
   both a `disjoint(a.x, b.x)` predicate and simple comparisons; the loader
   defines which predicates/fields are legal.
2. **Where annotations come from.** Each entity needs an annotation record
   (compartments, hydrophobicity, pocket volume, logP, …) keyed by UniProt (protein)
   or InChIKey (ligand). The loader supplies these from a per-modality annotation
   table; a rule whose fields are missing simply abstains for that pair.

## Status

`ppi.yaml` and `pli.yaml` are **starter templates** — the schema is fixed, the
thresholds/sources are placeholders (`TODO`) to be filled as rules are sourced.
The filters that consume them are built in Phase 1.
