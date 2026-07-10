# Authoring a rule — from a biological / topological / evolutionary fact to a validated YAML entry

This is the step-by-step for turning a piece of biology or network topology
("proteins in different compartments can't interact", "a ligand bigger than the
pocket can't bind", "two nodes with no shared neighbors and a low expected edge
count under the configuration model are unlikely to be adjacent") into a rule
the engine runs. For the field contract and the `when` grammar see `README.md`;
this doc is the procedure and the judgment calls.

A rule only ever makes a **non-interaction more or less believable** — it never
asserts an interaction. Keep that framing: you are scoring how safe a *negative*
(non-edge) is.

---

## Step 1 — State the constraint as "if … then this non-edge is safer/riskier"

Write the biology (or topology/evolution) in one plain sentence first, in this shape:

> *If two entities have \<property or relationship\>, they are unlikely to interact,
> so a non-edge between them is a **safe** or **riskier** negative.*

Examples:

- If two proteins never share a subcellular compartment → **safer** negative (PPI).
- If a ligand is far larger than the target pocket → **safer** negative (PLI).
- If two proteins are both in the same complex-forming compartment and co-expressed
  → **riskier** negative (might be a hidden positive; use to decrease confidence).
- If two proteins show no detectable co-evolution signal across orthologs
  (e.g. low mutual information / evolutionary coupling) → **safer** negative (PPI).
- If a ligand is specific to a lineage that lacks the corresponding protein ortholog
  or conserved binding pocket → **safer** negative (PLI).
- If a known interaction in one species is not conserved in closely related species
  despite conservation of both proteins → **riskier** negative (species-specific
  regulation rather than a robust physical interaction).
- If two proteins share no common neighbors and their expected edge count under
  the configuration-model baseline (`deg(a)·deg(b) / 2·|E|`) is small → **safer**
  negative (matches the `no_overlap`/`easy_negative` case in
  `negaverse/streams/topology.py::TopologyFilter`).

If you can't phrase it this way, it's probably not a filter rule (it may be a
positive-interaction predictor, which this engine does not encode).

---

## Step 2 — Pick `modality` and `applies_to`

- `modality`:
  - `ppi` — protein–protein interactions (same-type entities).
  - `pli` — protein–ligand interactions, including lipids and other small molecules.

- `applies_to`: the two entity **types**, in the order your `when` will reference
  them, e.g. `[protein, protein]` or `[protein, ligand]`.
  - Same type (`[protein, protein]`) → reference entities positionally as `a` and `b`.
  - Different types (`[protein, ligand]`) → reference them by type name:
    `protein.pocket_volume`, `ligand.volume`, `protein.compartments`, `ligand.class`.

Topology-based rules use the same modalities; they still talk about protein–protein
or protein–ligand pairs, but may rely on graph-structural features (e.g. `a.degree`,
`b.degree`, `a.neighbors`, `b.neighbors`, `a.graph_two_m`) if those are exposed as
annotation fields. Full degree-normalised L3/RA scoring needs per-element lookups
the `when` grammar can't express — that reasoning already lives in
`negaverse/streams/topology.py::TopologyFilter`; treat it as the authority for
that signal rather than re-deriving it here.

---

## Step 3 — Choose the annotation fields and how they are computed

Every `<entity>.<field>` in your `when` must be a field in the annotation table
(`negaverse/io/annotations.py`). The table below summarizes the main fields,
where they live, and how they are intended to be computed or loaded.

| field                              | on       | meaning / computation                                                                                       |
|------------------------------------|----------|-------------------------------------------------------------------------------------------------------------|
| `compartments`                    | protein  | Set of GO cellular-component terms, loaded from TSV as in `localization.py` (one node → comma-separated compartments). |
| `surface_hydrophobicity`          | protein  | Sequence-based surface hydrophobicity score (e.g. Kyte–Doolittle / similar scale, aggregated over exposed or interface residues). |
| `evolutionary_coupling_score_with_b` | protein  | Score on `a` for its coupling with `b`, from sequence covariation (e.g. EVcouplings; aggregated and normalized to `[0,1]`). |
| `interface_conservation`          | protein  | Mean conservation over interface residues, derived from MSAs (Consurf/entropy) plus interface annotation (structure/docking/prediction). |
| `degree`                          | protein  | Graph degree of the protein node in the PPI / heterogeneous network.                                       |
| `neighbors`                       | protein  | Set of node IDs adjacent to this node in the graph currently loaded; pair with `disjoint`/`shared`/`jaccard` for common-neighbor reasoning (mirrors `TopologyFilter`'s `cn`). |
| `graph_two_m`                     | protein  | `2 × total edges` in the graph currently loaded — same value on every node. Combine with `a.degree`/`b.degree` for the real configuration-model expected-edge count `(a.degree * b.degree) / a.graph_two_m` (mirrors `TopologyFilter`'s `expected_config`). |
| `pocket_volume`                   | protein  | Binding pocket volume from structure-based pocket detection (e.g. fpocket / CASTp) on available structures or models. |
| `pocket_hydrophobicity`           | protein  | Hydrophobicity score for the binding pocket (e.g. fpocket hydrophobicity descriptor).                       |
| `pocket_polarity`                 | protein  | Polarity score for the binding pocket (e.g. fraction of polar residues/atoms in pocket).                    |
| `volume`                          | ligand   | Approximate molecular volume from 3D conformers or vdW volume descriptors computed from SMILES (e.g. RDKit). |
| `logp`                            | ligand   | cLogP computed from SMILES using a cheminformatics toolkit (e.g. RDKit).                                     |
| `class`                           | ligand   | Ligand class (lipid, small molecule, metabolite, etc.), from rule-based classification or external databases (LipidMaps, HMDB…). |
| `origin`                          | ligand   | High-level origin annotation (e.g. host-produced, restricted_lineage) from curated metabolite / pathway databases. |
| `lineage_specificity`             | ligand   | Coarse label — `restricted_lineage` or `broad_lineage` — from curated annotation. A category flag, not a comparable ID; pair with `restricted_lineage_taxids` below for actual matching. |
| `restricted_lineage_taxids`       | ligand   | Set of **NCBI taxids** for the clade(s) that produce/use this ligand, when `lineage_specificity == 'restricted_lineage'`. |
| `permeability_class`              | ligand   | Permeability class (e.g. non_permeable, low, high) derived from physchem properties (logP, TPSA, MW, HBD/HBA) via rules or classifier. |
| `compartments`                    | ligand   | Set of compartments where the ligand is present (e.g. plasma, cytosol, membrane), from curated localization data. |
| `organism`                        | protein  | Organism label or taxonomy ID for the protein — informational only; not comparable to a lineage category string (see `lineage_taxids`). |
| `lineage_taxids`                  | protein  | Set of **NCBI taxids** for the organism's full ancestor lineage (e.g. `ete3`'s `NCBITaxa().get_lineage(taxid)`). Use with `disjoint`/`overlap` against `ligand.restricted_lineage_taxids`. |
| `tissue_expression`               | protein  | Set or distribution of tissues where the protein is expressed.                                              |
| `topology`                        | protein  | Topology class (e.g. membrane, soluble, secreted).                                                          |
| `cell_state`                      | protein  | Cell state or condition labels, if available (e.g. activated, resting).                                     |

Additional fields you **plan** to use and how to compute them:

### PPI fields

- `a.surface_hydrophobicity`, `b.surface_hydrophobicity`  
  - **Primary (fast, universal)**: sequence-based hydrophobicity scales  
    (e.g. Kyte–Doolittle, Engelman, Eisenberg). Compute per-residue hydrophobicity
    from the protein sequence, then aggregate over solvent-exposed or predicted
    interface residues to get a normalized surface hydrophobicity score.
  - **Optional refinement (where structures exist)**: structure-based hydrophobic
    patch tools (e.g. MolPatch, Protein-sol patches) can be used offline to refine
    surface patch metrics, but rules should be written to work with sequence-based
    scores alone.

- `a.evolutionary_coupling_score_with_b`  
  - Use tools such as EVcouplings on sequence MSAs to compute evolutionary
    coupling scores between the two proteins (or domains). Aggregate contact
    probabilities or coupling metrics into a pair-level score (e.g. mean or max
    over interface positions), normalized to `[0, 1]`.

- `a.interface_conservation`  
  - Compute residue-level conservation (e.g. via Consurf, PSI-BLAST + entropy)
    over MSAs.
  - Combine with interface annotation (from docking, co-crystal structure, or
    predicted interface residues) to get an “interface conservation” score
    (mean conservation over interface positions).

- Topology fields (if available), computed straight from the graph the pipeline
  is currently running against — mirrors `negaverse/streams/topology.py`:
  - `a.degree`, `b.degree` — node degree.
  - `a.neighbors`, `b.neighbors` — set of adjacent node IDs; use
    `disjoint`/`shared`/`jaccard(a.neighbors, b.neighbors)` for common-neighbor
    reasoning (this is `TopologyFilter`'s `cn`) — no new predicate needed.
  - `a.graph_two_m` — `2 × |E|` for the current graph (same value on every
    node); use `(a.degree * b.degree) / a.graph_two_m` for the actual
    configuration-model expected-edge baseline (`TopologyFilter`'s
    `expected_config`).
  - Full degree-normalized L3 and resource-allocation scores need per-element
    degree lookups inside the shared-neighbor set, which the `when` grammar
    can't express (no subscripting/iteration) — that's exactly what
    `TopologyFilter` already computes; don't re-derive it as a YAML rule.

### PLI fields

Pocket descriptors (protein side):

- `protein.pocket_volume`, `protein.pocket_polarity`, `protein.pocket_hydrophobicity`
  - Use structure-based pocket tools such as **fpocket** (or CASTp, similar) on
    available protein structures or high-confidence models:
    - Detect pockets.
    - For the relevant pocket, store:
      - `pocket_volume` from fpocket.
      - `pocket_hydrophobicity` (fpocket’s hydrophobicity descriptor).
      - `pocket_polarity` (e.g. fraction of polar residues/atoms).
  - Not all proteins will have structures; rules must tolerate missing pocket
    fields by abstaining when these fields are absent.

Ligand descriptors (cheminformatics + annotation):

- `ligand.volume`
  - Approximate molecular volume from 3D conformers (e.g. RDKit-based volume)
    or vdW volume descriptors computed from SMILES.

- `ligand.logp`
  - Use cheminformatics tools (e.g. RDKit) to compute cLogP from SMILES.

- `ligand.class`
  - Derive from substructures or external classification (e.g. lipid vs small
    molecule vs metabolite), using rule-based classification or external
    databases (LipidMaps, HMDB).

- `ligand.origin`, `ligand.lineage_specificity`
  - Annotation fields, not raw descriptors. Fill from curated sources
    (metabolite/protein databases, your own tables), e.g. “restricted_lineage”,
    “broad_lineage”, “host-produced”, etc., in a way that is general across
    species.

- `ligand.restricted_lineage_taxids` (when `lineage_specificity == 'restricted_lineage'`)
  - The set of **NCBI taxids** for the clade(s) that produce/use this ligand.
    Use taxids, not organism names/strings — names have synonym/casing drift
    (“human” vs “Homo sapiens”) that silently breaks exact-match `when`
    conditions, while taxids are the canonical key for lineage/ancestor lookups
    (e.g. via ete3's `NCBITaxa`). Pair with `protein.lineage_taxids` (below)
    via `disjoint`/`overlap` — don't compare against `protein.organism`, which
    is a single label/ID, not a lineage category.

- `ligand.permeability_class`
  - Compute from physchem features (logP, TPSA, MW, HBD/HBA) using rules or a
    simple classifier to approximate permeability (e.g. “non_permeable”, “low”,
    “high”).

- `ligand.compartments`
  - Set of compartment labels where the ligand is present (e.g. plasma, cytosol,
    membrane), derived from curated localization data in metabolite databases or
    your own experimental annotation.

### Context fields

- `protein.organism`, `protein.tissue_expression`, `protein.topology`,
  `protein.cell_state`, etc., from existing gene/protein expression and
  annotation sources.
- `protein.lineage_taxids` — the organism's full ancestor lineage as a **set
  of NCBI taxids** (e.g. `ete3`'s `NCBITaxa().get_lineage(taxid)`). Pair with
  `ligand.restricted_lineage_taxids` via `disjoint`/`overlap` for
  lineage-mismatch rules.

---

### Missing fields and staged rules

Need a field that isn't there yet (hydrophobicity, pocket volume, logP, degree,
co-evolution score…)? Two choices:

1. **Write the rule anyway** — it will load, validate, and simply **abstain**
   until the field is populated. Good for staging rules ahead of data (e.g.
   adding configuration-model or evolutionary rules before all features exist).
2. **Add the field**: load it in `build_annotation_table()` under a new key,
   keyed by the same node IDs the graph uses (UniProt / Ensembl / InChIKey / CID).
   One loader, no engine changes — **true for static fields** (compartments,
   pocket volume, logP, taxid lineages, ...) that come from external files/DBs
   independent of which graph is loaded.

Localization fields should be compatible with the TSV format used by
`localization.py` (one node per line,
`node<TAB>compartment1,compartment2,…`).

**Graph-structural fields are different.** `degree`, `neighbors`, and
`graph_two_m` depend on whichever graph is currently loaded, but
`build_annotation_table()` (`negaverse/io/annotations.py`) takes no graph
argument today, and its only call sites (`scripts/validate_rules.py`,
`negaverse/streams/rules.py::_RuleFilterBase.fit()`) call it with none —
`fit()` has `graph` available and simply doesn't pass it through. So these
three fields will always abstain until someone extends
`build_annotation_table(graph=None)` to populate them from `graph` when given,
and updates the two call sites to pass `graph` in. That's a small, real engine
change (not just a loader) — do it before relying on any topology rule from
Step 4 in production.

---

## Step 4 — Write the `when` expression

Use only the safe grammar (validated at load; anything else is rejected):

- Predicates over sets:
  - `disjoint(x, y)` — sets share no elements (e.g. compartments mismatch).
  - `overlap(x, y)` — sets share at least one element.
  - `shared(x, y)` — count of shared elements.
  - `jaccard(x, y)` — set similarity in `[0, 1]`.
  - `contains(x, v)` — membership / substring.

- Comparisons: `<`, `<=`, `>`, `>=`, `==`, `!=` (chained OK) and arithmetic `+ - * /`.

- Boolean composition: `and`, `or`, `not`.

- Literals: numbers, strings (`== 'polar'`, `== 'lipid'`), `True`, `False`.

Examples:

```yaml
when: "disjoint(a.compartments, b.compartments)"
when: "ligand.volume > protein.pocket_volume * 1.5"
when: "ligand.logp > 5 and protein.pocket_polarity == 'polar'"
when: "disjoint(a.neighbors, b.neighbors) and (a.degree * b.degree) / a.graph_two_m < 0.01"
when: "a.evolutionary_coupling_score_with_b < 0.1"
when: "ligand.lineage_specificity == 'restricted_lineage' and disjoint(ligand.restricted_lineage_taxids, protein.lineage_taxids)"
```

Note the current predicates are **binary** — the rule either fires or it doesn't.
If you need graded behavior, express it through thresholds on numeric fields
(e.g. `jaccard(a.compartments, b.compartments) < 0.1`).

A rule fires on exactly one direction; if you want both "disjoint → safer" *and*
"co-localized → riskier", write **two** rules.

---

## Step 5 — Pick `effect` and calibrate `weight`

- `effect`:
  - `safer_negative` — fires → confidence in the non-edge goes up.
  - `riskier_negative` — fires → confidence in the non-edge goes down
    (the pair looks more like a hidden positive).
  - `veto` — fires → the pair is dropped entirely; use only for hard biophysical
    or topological impossibilities.

- `weight` ∈ `[0, 1]` sets how strong the push is. The graded score maps:
  - `safer_negative`:  `value = 0.5 + 0.5 · weight`  (weight 0.8 → 0.9).
  - `riskier_negative`: `value = 0.5 − 0.5 · weight`  (weight 0.8 → 0.1).

Calibrate `weight` to **how reliably the constraint implies non-interaction**, not
to how famous the paper is.

Typical ranges:
| weight | use when                            | example                                        |
|--------|--------------------------------------|------------------------------------------------|
| 0.8–1.0| near-physical law; few exceptions   | disjoint compartments; non-permeable vs cytosolic |
| 0.4–0.7| strong tendency, real exceptions    | hydrophobicity mismatch; pocket size mismatch; strong absence of co-evolution |
| 0.1–0.3| weak prior / noisy signal           | coarse co-expression; mild topology/evolutionary mismatch |

Topology-specific guidance:
- No shared neighbors (`disjoint(a.neighbors, b.neighbors)`) combined with a
  low configuration-model expected-edge count (`(a.degree * b.degree) /
  a.graph_two_m` near 0) is the `no_overlap`/`easy_negative` case
  `TopologyFilter` already floors at `value ≈ 0.98` — a comparable YAML rule
  can use `weight` in the 0.7–0.9 range.
- Full L3/RA-based scoring is a more refined version of this same signal, but
  isn't expressible as a `when` rule (see Step 3) — use `TopologyFilter`'s
  output for that rather than approximating it here.

Evolutionary-specific guidance:
- Strong absence of co-evolution signal across well-sampled orthologs → `weight`
  in the 0.5–0.7 range.
- Lineage-specific ligands without conserved binding partners or pockets → `weight`
  in the 0.4–0.6 range.
- Species-specific interaction loss with conserved proteins may be better encoded
  as `riskier_negative` for non-edges in the species where the interaction has
  been **lost** (not observed) — ortholog conservation elsewhere still argues
  against confidently calling that non-edge a safe negative.

When two rules fire on the same pair, their values are combined weighted by
`weight`.

---

## Step 6 — Write `rationale`, `source`, and `flag`

- `rationale`: 1–2 sentences of **why**. This text is fed to the LLM filter as
  grounding, so make it a clear causal statement, not a citation dump.
- `source`: the citation — first author, year, and a DOI/PMID if you have it.
  Use `TODO — …` while sourcing; that records the gap.
- `flag` (optional): a short tag added to records the rule fires on
  (defaults to `id`).

Example rationales:

```yaml
rationale: >
  Two proteins that never share a subcellular compartment cannot physically
  interact, so a non-edge between them is a safe negative.

rationale: >
  A ligand substantially larger than the target's binding pocket cannot be
  accommodated, so a non-edge is a safe negative.

rationale: >
  Two proteins with no shared interactors and a configuration-model expected
  edge count near zero are rarely adjacent in this network, so a non-edge
  between them is a safer negative.

rationale: >
  Proteins with no detectable co-evolution signal across orthologs are less
  likely to form a conserved physical complex, making a non-edge a safer
  negative.
```

---

## Step 7 — Evidence-aware rule authoring

Rules encode **biological, evolutionary, and topological constraints**, not the
raw outcome of literature search. They must be consistent with the way your
literature pipeline treats evidence.

### Respect direct binding and explicit non-interaction

The literature workflow supports two directions:

- Detect evidence of direct or strongly supported physical binding.
  - If found, remove the pair from the negative pool and place it into an
    external validation dataset.
- Detect explicit non-interaction or negative-control evidence.
  - Only use this to strengthen negative confidence when the evidence is explicit
    and entity resolution is confident.

For rule authoring, this implies:

- Do not write rules that try to override strong, direct binding evidence.
  Rules are priors; direct curated binding data wins.
- Explicit non-interaction / negative-control evidence can justify **higher
  weights** for safer negatives when:
  - experimental context is clear (species, tissue, assay type, conditions),
  - identifiers are resolved unambiguously to the correct protein/ligand.

### Do not encode “absence-of-evidence” as a rule

Time caps and retrieval caps in literature screening are **operational limits**,
not biological evidence. Therefore:

- Do **not** author rules that say “if no evidence is found in database or
  literature, the non-edge is safe”.
- Pairs with `no_evidence_found` or `needs_manual_review` are handled by the
  screening pipeline (confidence tiers, output files), not by deterministic rules.
- Rules must be based on positive constraints: localization, pocket geometry,
  physicochemical mismatch, organism/tissue/state, conservation/co-evolution,
  or explicit non-binding — not on “we did not see this in BindingDB/ChEMBL”.

### Be consistent with literature expression rules

Biological events and interaction statements may be expressed as:

- Standard verbal predicates (“X binds Y”, “X did not interact with Y”).
- Nominalizations (“binding of X to Y”, “non-interaction of X and Y”).
- Ellipsis (“X interacted with Y, but not with Z”).
- Subordinate or reduced clauses with omitted but inferable verbs.
- Coordinated statements where one predicate governs multiple entities.
- Negative-control phrasing where non-binding is implied by assay outcomes.

When you derive rules from literature:

- Infer binding or non-binding only when the omitted or implied predicate
  is strongly supported by local sentence context.
- Preserve the original sentence or fragment used for inference when reporting
  back to the user.
- Distinguish explicit non-interaction (“did not bind”, “no interaction detected”)
  from mere absence of any reported interaction.
- Treat nominalized events as candidate evidence, not noise.
- Treat elliptical or abbreviated constructions as lower-confidence unless
  entity-role assignment is clear.
- Mark ambiguous syntax, unresolved scope, or unclear negation for manual
  review rather than encoding them as deterministic rules.

Rules should capture constraints that remain valid regardless of whether the
literature search budget (time cap, retrieval cap) is exhausted.

---

## Step 8 — Validate

Run:

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

`validate_rules.py` → `[READY] colocalization_mismatch` (because `compartments`
is populated). This is the rule that powers co-localization today — no Python.