---
name: rule-from-literature
description: >
  Turn a biological finding, paper, or abstract into a validated negaverse rule
  (rules/*.yaml). Use when the user provides literature (a PDF path, URL, pasted
  abstract, or a plain claim) and wants a non-interaction / negative-scoring rule
  added — e.g. "add a co-localization rule from this paper", "make a rule from
  this abstract", "encode this binding-pocket constraint as a rule".
---

# Build a negaverse rule from a literature source

You are extracting a **negative-interaction constraint** from a source and encoding
it as a declarative rule the engine runs (no Python). A rule only ever makes a
*non-edge* more or less believable — it never asserts an interaction.

**Read these first — they are the authoritative contract; do not re-derive them:**
- `rules/AUTHORING.md` — the step-by-step procedure and the weight-calibration table.
- `rules/README.md` — the field contract and the exact `when` grammar.
- `negaverse/io/annotations.py` — which annotation fields currently exist.

## Inputs you accept
A PDF path (Read it), a URL (WebFetch it), a pasted abstract/quote, or a one-line
claim. If given only a vague topic, ask for the specific finding or a source.

## Procedure

1. **Extract the constraint.** From the source, identify a statement of the form
   *"entities with <property relationship> are unlikely to / cannot interact."*
   Quote the sentence(s) you're basing it on. If the finding is a *positive*
   interaction predictor (it predicts that things DO bind), **stop and say so** —
   this engine encodes negatives only.

2. **Draft the rule** following `rules/AUTHORING.md` steps 2–6:
   - `modality` (`ppi`/`pli`) and `applies_to` (entity types, in the order `when`
     references them; positional `a`/`b` for same-type, type names for mixed).
   - `when` — use **only** the whitelisted grammar (predicates
     `disjoint/overlap/shared/jaccard/contains`, comparisons, arithmetic, `and/or/not`,
     literals). Never invent a predicate; the loader rejects anything else.
   - `effect` (`safer_negative`/`riskier_negative`/`veto`) and `weight` (0–1),
     calibrated to **how reliably the constraint implies non-interaction**, not to
     the prominence of the paper. Use the weight table in AUTHORING.md.
   - `rationale` — 1–2 sentences of the causal *why* (this text grounds the LLM filter).
   - `source` — real citation: first author, year, DOI/PMID. **Never fabricate a
     citation.** If you don't have it, write `TODO — <what's missing>`.
   - `id` — a stable, descriptive snake_case slug; `flag` (optional) short tag.

3. **Check the annotation fields.** Every `<entity>.<field>` in `when` must be a
   field produced by `build_annotation_table()`. If a field doesn't exist yet
   (e.g. `surface_hydrophobicity`, `pocket_volume`), **do not block** — the rule
   will load and simply abstain. Tell the user the rule is staged and what
   annotation loader would activate it (which source, keyed by which ID space).
   Offer to add the loader in a follow-up.

4. **Append** the entry to `rules/ppi.yaml` or `rules/pli.yaml` (match `modality`).
   Preserve the existing formatting and comments; add, don't rewrite.

5. **Validate** — run:
   ```bash
   PYTHONPATH=. python scripts/validate_rules.py
   ```
   It must exit 0 (all rules parse and are safe). Report whether the new rule is
   `READY` (annotations present) or `abstain` (list the missing fields). If it
   fails to parse, fix the `when` expression and re-run — do not leave a broken
   rules file.

## Guardrails
- One rule per distinct constraint; if a paper yields two (e.g. a safe *and* a
  risky direction), write two rules.
- Prefer `safer_negative`/`riskier_negative`; reserve `veto` for hard biophysical
  impossibilities the user explicitly wants as a hard drop.
- Keep the `when` machine-checkable. If the biology can't be reduced to the
  available fields + predicates, say so — that case is better handled by the LLM
  literature filter (which will receive this rule's `rationale` anyway), not by a
  deterministic rule.

## Report back
Show the user: the source sentence you used, the final YAML entry, and the
`validate_rules.py` result (READY vs. abstain + any missing fields / suggested
annotation loader).
