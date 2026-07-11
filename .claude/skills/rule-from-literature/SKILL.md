---
name: rule-from-literature
description: >
  Turn a biological or evolutionary finding (paper, abstract, claim) into a
  validated negaverse rule (rules/*.yaml). Use when the user provides a source
  and wants a protein–protein (ppi) or protein–ligand (pli) non-interaction /
  negative-scoring rule added.
---

# Build a negaverse rule from a literature or evolutionary source

You are extracting a **negative-interaction constraint** from a source and encoding
it as a declarative rule the engine runs (no Python). A rule only ever makes a
non-edge more or less believable — it never asserts an interaction.

You must follow `rules/AUTHORING.md` exactly; do not re-derive the contract.

## Required inputs and prep

You accept:
- A PDF path (read it).
- A URL (fetch it).
- A pasted abstract or quote.
- A one-line claim (plus modality hint if available).

Before authoring a rule:
- Read `rules/AUTHORING.md` for the full procedure, weight calibration, examples,
  and evidence-aware guidance.
- Read `rules/README.md` for the field contract and exact `when` grammar.
- Read `negaverse/io/annotations.py` to know which annotation fields exist.
- Read `rules/ppi.yaml` and `rules/pli.yaml` to see existing rule style.
- Read `negaverse/io/localization.py` to understand how `compartments` are loaded.

If given only a vague topic, ask the user for the specific finding or source
before writing a rule.

## Step 1 — Extract a negative constraint from literature or evolution

From the source, identify a statement of the form:

> "Entities with \<property or relationship\> are unlikely to / cannot interact."

Restate it in `AUTHORING.md`'s canonical shape before moving on:

> "If two entities have \<property or relationship\>, they are unlikely to
> interact, so a non-edge between them is a **safer** or **riskier** negative."

If the source doesn't reduce to that sentence, it's probably not a filter rule.

Examples you should look for:

- Localization / compartment constraints.
- Physicochemical constraints (pocket-volume mismatch, polarity/hydrophobicity mismatch).
- Topological constraints (no shared network neighbors + low configuration-model
  expected edge count; see `negaverse/streams/topology.py::TopologyFilter`).
- Database-confidence-score constraints (a pair scoring below a database's own
  minimum reporting threshold, e.g. STRING `combined_score < 0.15`).
- Expression / context (tissue, cell state, organism).
- Evolutionary constraints:
  - Lack of co-evolution signal between proteins across orthologs.
  - Lineage-specific ligands without conserved protein orthologs or pockets.
  - Species-specific interaction loss with conserved partners.
- Explicit non-interaction or negative-control experiments:
  - Text like “did not bind”, “no interaction detected”, “used as a negative control”.

If the finding is a **purely positive interaction predictor** (it predicts that
things DO bind), explain that this engine encodes negative constraints only and
stop instead of writing a rule.

## Step 2 — Choose modality and `applies_to`

Determine which modality the constraint belongs to:

- `ppi`: protein–protein interactions. Set `applies_to: [protein, protein]` and
  refer to the two entities as `a` and `b` in `when`.
- `pli`: protein–ligand interactions. Set `applies_to: [protein, ligand]` and
  refer to them as `protein.<field>` and `ligand.<field>` in `when`.

## Step 3 — Map to annotation fields and how they'd be computed

For the extracted constraint, decide which annotation fields it uses. Use only
fields that exist (or will exist) in `build_annotation_table()`:

- PPI fields: `a.compartments`, `b.compartments`,
  `a.surface_hydrophobicity`, `b.surface_hydrophobicity`,
  `a.evolutionary_coupling_score_with_b`, `a.string_score_with_b`,
  `a.interface_conservation`,
  `a.degree`, `b.degree`, `a.neighbors`, `b.neighbors`, `a.graph_two_m`.
- PLI fields: `protein.pocket_volume`, `ligand.volume`,
  `ligand.logp`, `protein.pocket_polarity`, `protein.pocket_hydrophobicity`,
  `ligand.class`, `ligand.origin`, `ligand.lineage_specificity`,
  `ligand.restricted_lineage_taxids`, `ligand.permeability_class`,
  `ligand.compartments`.
- Context fields: `protein.organism`, `protein.tissue_expression`,
  `protein.topology`, `protein.cell_state`, `protein.lineage_taxids`.

If a field does **not** exist yet, you may still author the rule. It will validate
and abstain until the field is populated. `AUTHORING.md` Step 3 documents the
specific tool/method that computes (or would compute) each field —
`scripts/compute_surface_hydrophobicity.py` (two-tier: DSSP+AlphaFold pLDDT
exposure/disorder masking when a confident structure exists, sequence-mean
fallback otherwise) for `surface_hydrophobicity`, EVcouplings for
`evolutionary_coupling_score_with_b`, Consurf for `interface_conservation`,
`scripts/compute_pocket_descriptors.py` (fpocket) for the `pocket_*` fields,
RDKit for `ligand.volume`/`ligand.logp`, LipidMaps/HMDB for `ligand.class`, and
`negaverse/streams/topology.py::TopologyFilter`'s own graph traversal for
`neighbors`/`graph_two_m` (don't re-derive full L3/RA scoring as a YAML rule —
that filter is the authority for it). For lineage-mismatch rules, use **NCBI
taxids** (e.g. via ete3's `NCBITaxa`) for `lineage_taxids`/
`restricted_lineage_taxids`, not organism names — names have synonym/casing
drift that breaks exact-match `when` conditions. Cite the relevant one when
telling the user what would activate the staged rule.

## Step 4 — Author the `when` expression

Construct `when` using only the whitelisted grammar:

- Set predicates: `disjoint`, `overlap`, `shared`, `jaccard`, `contains`.
- Comparisons and arithmetic: `< <= > >= == !=`, `+ - * /`.
- Boolean operators: `and`, `or`, `not`.
- Literals: numbers, strings, booleans.

Examples you may generate:

```yaml
when: "disjoint(a.compartments, b.compartments)"
when: "ligand.volume > protein.pocket_volume * 1.5"
when: "ligand.logp > 5 and protein.pocket_polarity > 0.5"
when: "disjoint(a.neighbors, b.neighbors) and (a.degree * b.degree) / a.graph_two_m < 0.01"
when: "a.evolutionary_coupling_score_with_b < 0.1"
when: "a.string_score_with_b < 0.15"
when: "a.surface_hydrophobicity > 0.44 or b.surface_hydrophobicity > 0.44"
when: "ligand.lineage_specificity == 'restricted_lineage' and disjoint(ligand.restricted_lineage_taxids, protein.lineage_taxids)"
```

Never invent new predicates; if the constraint cannot be represented using the
available predicates and fields, explain that it should be handled by the LLM
filter using `rationale`, not by a deterministic rule.

## Step 5 — Set `effect` and `weight`

Use AUTHORING.md’s calibration:

- Choose `effect`:
  - Prefer `safer_negative` and `riskier_negative`.
  - Reserve `veto` for hard impossibilities the user explicitly wants as hard drops.

- Choose `weight` based on reliability of non-interaction:
  - Strong, near-physical or very strong constraints (disjoint compartments,
    non-permeable vs strictly intracellular, configuration-model impossible,
    robust explicit non-binding) → `0.8–1.0`.
  - Strong tendencies with known exceptions (pocket-volume mismatch, polarity
    mismatch, strong absence of co-evolution) → `0.4–0.7`.
  - Weak priors (coarse expression mismatch, mild topology/evolutionary mismatch) → `0.1–0.3`.

Explain to the user how you chose `effect` and `weight` in terms of the source
(e.g. “multiple negative-control assays” vs “single nominalized statement”).

## Step 6 — Write `id`, `rationale`, `source`, and `flag`

For every rule:

- `id`: stable snake_case slug (e.g. `colocalization_mismatch`,
  `ligand_pocket_size_mismatch`, `evolutionary_coupling_absence`).
- `rationale`: 1–2 sentences explaining why the non-edge is safer or riskier;
  mention whether this is based on localization, physicochemical, evolutionary,
  or explicit non-binding evidence.
- `source`: real citation (first author, year, DOI/PMID) or `TODO — …`.
- `flag` (optional): short tag; defaults to `id` if omitted.

## Step 7 — Respect the literature screening rules

You must align with the literature workflow:

- Use direct binding evidence to **avoid** rules that would contradict known
  tight-binders.
- Use explicit non-interaction / negative-control evidence to justify higher
  weights only when context and entity resolution are clear.
- Do **not** encode “no evidence found” as a rule condition.
- When interpreting text:
  - Consider verbs (“binds”, “did not interact”), nominalizations, ellipsis,
    subordinate clauses, coordination, and negative-control phrasing.
  - Infer the relation only when the implied predicate is strongly supported
    by the local sentence context.
  - Preserve the original sentence or fragment in feedback to the user.
  - Distinguish explicit non-interaction from mere absence of reported interaction.
  - Treat ambiguous syntax or unclear negation as insufficient for a rule; mark
    such constraints as LLM-only (rationale, no deterministic `when`).

## Step 8 — Append, validate, and report back

Append to `rules/ppi.yaml` or `rules/pli.yaml` based on modality, preserving
formatting. Then run:

```bash
PYTHONPATH=. python scripts/validate_rules.py
```

Guardrails before you report back:

- One rule per distinct constraint; if a source implies two directions (e.g. a
  safer *and* a riskier reading), write two rules.
- Prefer `safer_negative`/`riskier_negative`; reserve `veto` for hard
  biophysical or topological impossibilities the user explicitly wants as a
  hard drop.
- If the constraint can't be reduced to the available fields + predicates, say
  so — hand it to the LLM literature filter via `rationale` instead of forcing
  a deterministic rule.

Report to the user:

- The source sentence(s) used.
- The final YAML rule entry.
- The `validate_rules.py` result: `READY` (annotations present) or `abstain`
  (list the missing fields and, if relevant, which annotation loader would
  activate it).
