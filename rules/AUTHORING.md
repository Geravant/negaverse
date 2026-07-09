# Authoring a rule — from a biological fact to a validated YAML entry

This is the step-by-step for turning a piece of biology ("proteins in different
compartments can't interact", "a ligand bigger than the pocket can't bind") into a
rule the engine runs. For the field contract and the `when` grammar see
[`README.md`](README.md); this doc is the *procedure* and the judgment calls.

A rule only ever makes a **non-interaction more or less believable** — it never
asserts an interaction. Keep that framing: you are scoring how safe a *negative*
(non-edge) is.

---

## Step 1 — State the constraint as "if … then this non-edge is safer/riskier"

Write the biology in one plain sentence first, in this shape:

> *If two entities have <property relationship>, they are unlikely to interact, so a non-edge between them is a **safe** negative.*

Examples:
- If two proteins never share a subcellular compartment → **safer** negative.
- If a ligand is far larger than the target pocket → **safer** negative.
- If two proteins are both in the same complex-forming compartment and co-expressed → **riskier** negative (might be a hidden positive).

If you can't phrase it this way, it's probably not a filter rule (it may be a
positive-interaction predictor, which this engine does not encode).

## Step 2 — Pick `modality` and `applies_to`

- `modality`: `ppi` (protein–protein) or `pli` (protein–ligand).
- `applies_to`: the two entity **types**, in the order your `when` will reference
  them, e.g. `[protein, protein]` or `[protein, ligand]`.
  - Same type (`[protein, protein]`) → reference entities positionally as `a` and `b`.
  - Different types (`[protein, ligand]`) → reference them by type name:
    `protein.pocket_volume`, `ligand.volume`.

## Step 3 — Choose the annotation fields and check they exist

Every `<entity>.<field>` in your `when` must be a field in the annotation table
(`negaverse/io/annotations.py`). Currently populated:

| field | on | meaning |
|---|---|---|
| `compartments` | protein | set of GO cellular-component terms |

Need a field that isn't there yet (hydrophobicity, pocket volume, logP…)? Two choices:
1. **Write the rule anyway** — it will load, validate, and simply **abstain** until
   the field is populated. Good for staging rules ahead of data.
2. **Add the field**: load it in `build_annotation_table()` under a new key, keyed
   by the same node IDs the graph uses (UniProt / Ensembl / InChIKey). One loader,
   no engine changes.

## Step 4 — Write the `when` expression

Use only the safe grammar (validated at load; anything else is rejected):

- predicates over sets: `disjoint(x, y)`, `overlap(x, y)`, `shared(x, y)` (count),
  `jaccard(x, y)` (0–1), `contains(x, v)`
- comparisons: `< <= > >= == !=` (chained OK) and arithmetic `+ - * /`
- combine with `and` / `or` / `not`
- literals: numbers, strings (`== 'polar'`), `True` / `False`

```yaml
when: "disjoint(a.compartments, b.compartments)"
when: "ligand.volume > protein.pocket_volume * 1.5"
when: "a.surface_hydrophobicity < 0.2 and b.surface_hydrophobicity < 0.2"
```

Note the current predicates are **binary** — the rule either fires or it doesn't.
A rule fires on exactly one direction; if you want both "disjoint → safer" *and*
"co-localized → riskier", write **two** rules.

## Step 5 — Pick `effect` and calibrate `weight`

- `effect`: `safer_negative` (fires → confidence up), `riskier_negative` (fires →
  confidence down), or `veto` (fires → the pair is dropped entirely; use only for
  hard biophysical impossibilities).
- `weight` ∈ `[0, 1]` sets how strong the push is. The graded score maps:
  - `safer_negative`:  `value = 0.5 + 0.5 · weight`  (weight 0.8 → 0.9)
  - `riskier_negative`: `value = 0.5 − 0.5 · weight`  (weight 0.8 → 0.1)

Calibrate `weight` to **how reliably the constraint implies non-interaction**, not
to how famous the paper is:

| weight | use when | example |
|---|---|---|
| 0.8–1.0 | near-physical law; few exceptions | different compartment (can't co-occur) |
| 0.4–0.7 | strong tendency, real exceptions | hydrophobicity mismatch |
| 0.1–0.3 | weak prior / noisy signal | coarse co-expression |

When two rules fire on the same pair, their values are combined weighted by `weight`.

## Step 6 — Write `rationale` and `source`

- `rationale`: 1–2 sentences of *why*. This text is fed verbatim to the LLM filter
  as grounding, so make it a clear causal statement, not a citation dump.
- `source`: the citation — first author, year, and a DOI/PMID if you have it. Use
  `TODO — …` while sourcing; that's fine, it just records the gap.
- `flag` (optional): a short tag added to records the rule fires on (defaults to `id`).

## Step 7 — Validate

```bash
PYTHONPATH=. python scripts/validate_rules.py
```

This fails loudly if the rule won't parse or uses an unsafe/unknown construct, and
otherwise prints `READY` (annotations present) or `abstain` (with the missing
fields). All rules must parse; `abstain` is acceptable for staged rules.

---

## Worked example

Biology: *"Two proteins that are never in the same subcellular compartment cannot
physically interact."*

```yaml
- id: colocalization_mismatch
  modality: ppi
  applies_to: [protein, protein]
  when: "disjoint(a.compartments, b.compartments)"
  effect: safer_negative
  weight: 0.8                       # near-physical: strong
  flag: different_compartment
  rationale: >
    Two proteins that never share a subcellular compartment cannot physically
    interact, so a non-edge between them is a safe negative.
  source: "GO cellular_component"
```

`validate_rules.py` → `[READY] colocalization_mismatch` (because `compartments` is
populated). This is the rule that powers co-localization today — no Python.
