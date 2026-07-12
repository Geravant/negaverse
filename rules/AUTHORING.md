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
  negative — but this is *already* the `no_overlap`/`easy_negative` case
  `negaverse/streams/topology.py::TopologyFilter` floors at `value ≈ 0.98`
  (and more rigorously: it also checks L3 length-3 paths, not just common
  neighbors). Don't also encode this as a YAML rule — `TopologyFilter` and the
  rule engine are independent streams that both feed the same combined score,
  so a YAML rule duplicating this exact condition would double-count one
  topological fact rather than add new evidence. (An earlier version of this
  project *did* ship such a rule, `no_shared_neighbors_low_expected_edge`; it
  was removed for exactly this reason — see Step 5's redundancy note.)
- If STRING's physical-subnetwork confidence for a pair falls below STRING's
  own minimum reporting threshold (`combined_score < 0.15`) → **safer** negative
  (PPI) — the database finds no supporting evidence in any of its channels
  (experiments, curated DBs, co-expression, text-mining, genomic context).
- If either protein in a pair has unusually HIGH exposed surface hydrophobicity
  → **safer** negative (PPI) — counterintuitive at first, but empirically
  calibrated against two gold-standard PPI benchmarks (DRYAD, UPNA-PPI): real
  interactions have LOWER exposed hydrophobicity than real non-interactions,
  consistent with interface hot spots being enriched in aromatic/cation-pi
  residues (Trp, Tyr, Arg — Bogan & Thorn 1998) rather than classic
  Kyte-Doolittle-hydrophobic ones. See `rules/ppi.yaml`'s
  `hydrophobicity_interface` rationale for the full writeup — a worked example
  of Step 5's "empirical calibration can reverse your initial premise" guidance.

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
| `surface_hydrophobicity`          | protein  | Two-tier Kyte-Doolittle score (`scripts/compute_surface_hydrophobicity.py`): Tier 1 aggregates over solvent-exposed (DSSP RSA), ordered (AlphaFold pLDDT) residues when a confident structure exists; Tier 2 falls back to a whole-sequence mean otherwise. See `rules/ppi.yaml`'s `hydrophobicity_interface` for the calibrated (and direction-reversed) threshold. |
| `evolutionary_coupling_score_with_b` | protein (pairwise) | Evolutionary Rate Covariation between `a` and `b` (`scripts/compute_evolutionary_coupling.py`, RERconverge) — Pearson correlation of their relative-evolutionary-rate vectors across a fixed vertebrate species panel. Genuinely pairwise (loaded via `build_pair_annotation_table()`, not the per-node table); not yet calibrated. |
| `string_experimental_score_with_b` | protein (pairwise) | Score on `a` for its STRING (v12.0) `experimental` channel with `b` — direct wet-lab assay evidence, normalized to `[0,1]` (raw score ÷ 1000). Not `combined_score`: that blends in weaker indirect evidence (text-mining, coexpression). Genuinely pairwise (`build_pair_annotation_table()`). Backs `ppi.yaml`'s `string_low_confidence_non_interaction` (`< 0.15`, STRING's own reporting cutoff, safer_negative) via `scripts/compute_string_experimental.py`. The opposite tail (`> 0.9`) is a known-positive source instead, not a rule — see `rules/sources.yaml`'s `string_experimental_high_confidence` and `scripts/build_known_positive_sources.py`. |
| `interface_conservation`          | protein  | Mean conservation over interface residues, derived from MSAs (Consurf/entropy) plus interface annotation (structure/docking/prediction). |
| `degree`                          | protein  | Graph degree of the protein node in the PPI / heterogeneous network.                                       |
| `neighbors`                       | protein  | Set of node IDs adjacent to this node in the graph currently loaded; pair with `disjoint`/`shared`/`jaccard` for common-neighbor reasoning (mirrors `TopologyFilter`'s `cn`). |
| `graph_two_m`                     | protein  | `2 × total edges` in the graph currently loaded — same value on every node. Combine with `a.degree`/`b.degree` for the real configuration-model expected-edge count `(a.degree * b.degree) / a.graph_two_m` (mirrors `TopologyFilter`'s `expected_config`). |
| `pocket_volume`                   | protein  | Binding pocket volume (Å³), fpocket's `Volume` on the top-ranked pocket (`scripts/compute_pocket_descriptors.py`), confident AlphaFold structures only. |
| `pocket_hydrophobicity`           | protein  | fpocket's native (unnormalized) `Hydrophobicity score` for the top-ranked pocket (`scripts/compute_pocket_descriptors.py`).                       |
| `pocket_polarity`                 | protein  | Numeric `[0,1]` fraction of polar pocket atoms, fpocket's `Proportion of polar atoms` ÷ 100 (`scripts/compute_pocket_descriptors.py`).                    |
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

- `a.surface_hydrophobicity`, `b.surface_hydrophobicity` — **implemented**,
  `scripts/compute_surface_hydrophobicity.py`:
  - **Tier 1 (structure present, confident — mean AlphaFold pLDDT ≥ 70)**: run
    `mkdssp` (via `Bio.PDB.DSSP`) on the AlphaFold model to get each residue's
    relative solvent accessibility (RSA); aggregate Kyte-Doolittle over
    residues that are both solvent-exposed (RSA ≥ 0.25) and ordered/confident
    (per-residue pLDDT ≥ 70, used as the disorder-masking proxy). DSSP results
    are cached per structure (`local-docs/alphafold/<acc>.dssp.json`) since
    re-running it is the pipeline's dominant cost.
  - **Tier 2 (no usable structure)**: mean Kyte-Doolittle over the whole
    sequence (`scripts/compute_hydrophobicity.py`'s original proxy).
  - Tested alternatives that did **not** replace this: the Wimley-White
    interfacial hydrophobicity scale (comparable separation, no clear win) and
    Spatial Aggregation Propensity / SAP (weaker signal once accounting for
    its much lower structure-only pair coverage — no sequence fallback exists
    for a spatial-patch method). See the calibration script's docstring for
    the full three-way comparison.

- `a.evolutionary_coupling_score_with_b` — **implemented, not yet calibrated**.
  EVcouplings/EVcomplex (residue-level direct-coupling analysis) was evaluated
  and rejected: its binaries do install, but real use needs a large jackhmmer
  reference database and per-pair species-matched ortholog alignments with
  strict quality gating — hours of compute per family and a high abstention
  rate, not viable at PPI-benchmark calibration scale. Implemented instead:
  **Evolutionary Rate Covariation (ERC)** — Clark & Aquadro (2010); the same
  method behind the mitonuclear-coevolution literature (Zhang et al. 2015,
  Weng et al. 2016, Forsythe et al. 2021) — via **RERconverge**
  (github.com/nclark-lab/RERconverge), a real, actively-maintained R package.
  Pipeline: `scripts/fetch_orthologs.py` (Ensembl REST homology, a fixed
  16-species vertebrate panel) → `scripts/build_gene_alignments.py` (MAFFT) →
  `scripts/estimate_phangorn_trees.R` (branch-length ML fitting on a **fixed**
  master topology, `scripts/data/vertebrate_master_tree.nwk` — RERconverge's
  own recommended method at this scale; a free per-gene topology search, e.g.
  FastTree, disagrees with the master tree far too often at ~16 taxa and gets
  discarded as "discordant") → `scripts/rerconverge_runner.R`
  (readTrees/getAllResiduals) → `scripts/compute_evolutionary_coupling.py`
  (Pearson correlation between the two genes' relative-evolutionary-rate
  vectors, gated at `MIN_SHARED_BRANCHES` — a correlation over too few shared
  branches is noise, not signal, so the pair simply abstains rather than
  getting a low-confidence number).
  **Unlike every other field so far, this one is genuinely pairwise** (A's
  coupling *with B specifically*, not a property of A alone) — it's loaded via
  `build_pair_annotation_table()` (`negaverse/io/annotations.py`), a second
  mechanism alongside the per-node `build_annotation_table()`, reading
  `node_a<TAB>node_b<TAB>value` files and merged onto the right entity's
  record per-pair at score time (`streams/rules.py::_RuleFilterBase`) — see
  that module for why a naive per-node cache can't represent this correctly.
  **Scope**: the species panel and master tree are vertebrate-specific
  (matching this project's human PPI calibration benchmarks); bacterial or
  viral query proteins simply have no Ensembl vertebrate gene mapping and
  abstain gracefully, rather than producing a meaningless score.
  `scripts/calibrate_evolutionary_coupling_threshold.py` (same
  subsample/Youden's-J/protein-disjoint methodology as
  `hydrophobicity_interface`) is written and runs end-to-end, but the
  real calibration run hasn't been performed yet — `evolutionary_coupling_absence`'s
  threshold in `rules/ppi.yaml` is still the pre-calibration placeholder.

- `a.interface_conservation`  
  - Compute residue-level conservation (e.g. via Consurf, PSI-BLAST + entropy)
    over MSAs.
  - Combine with interface annotation (from docking, co-crystal structure, or
    predicted interface residues) to get an “interface conservation” score
    (mean conservation over interface positions).

- `a.string_experimental_score_with_b`
  - `scripts/string_channel.py` (shared plumbing) + `scripts/compute_string_experimental.py`:
    resolve UniProt accessions to STRING's native `<taxid>.<Ensembl_protein_id>`
    keys via STRING's own alias file
    (`local-docs/string/9606.protein.aliases.v12.0.txt.gz`, `UniProt_AC` type),
    then read the `experimental` column (not `combined_score`) from
    `local-docs/string/9606.protein.links.detailed.v12.0.txt.gz` and divide by
    1000 to normalize to `[0, 1]`. Deliberately the `experimental` channel,
    not `combined_score`: a hard veto (see below) needs direct binding
    evidence, not a blend that also includes text-mining/coexpression.
  - The same channel's opposite tail (`> 0.9`) is deliberately *not* a rule
    here — that level of direct evidence is a plain "this pair is documented
    as interacting" fact, handled as a known-positive source instead
    (`rules/sources.yaml`'s `string_experimental_high_confidence`,
    `scripts/build_known_positive_sources.py`) so it outright removes the
    pair via `KnownPositiveVeto`, same as BioGRID/IntAct. See
    `rules/SOURCES.md` for why sources.yaml and the rule engine are
    different mechanisms, and Step 5 below for the general principle.
  - STRING's `cooccurence` channel (a different, evolutionary-coupling-style
    signal — phylogenetic profiling, not direct evidence) was also tested
    against DRYAD/UPNA-PPI and found no reliable separation (AUROC ~0.54-0.57,
    CI barely excluding 0.5 even at full-dataset scale, driven mostly by
    "known to STRING at all" rather than cooccurence specifically); its
    one-off compute/calibration scripts were removed once that was settled.

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
  — **implemented**, `scripts/compute_pocket_descriptors.py`: runs **fpocket**
  on the AlphaFold model (reusing the same fetch as `surface_hydrophobicity`'s
  Tier 1, confident structures only — no sequence fallback exists for pocket
  geometry), reads the top-ranked pocket ("Pocket 1", fpocket's own
  druggability-adjacent ranking) from its `info.txt`:
    - `pocket_volume` — fpocket's `Volume` (Å³), used as-is.
    - `pocket_hydrophobicity` — fpocket's `Hydrophobicity score`, native
      (unnormalized) scale; no existing rule reads this field yet.
    - `pocket_polarity` — fpocket's `Proportion of polar atoms` ÷ 100, a
      **numeric** `[0,1]` fraction (not a category string — `rules/pli.yaml`'s
      `physicochemical_incompatibility` compares it as `> 0.5`).
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

**Graph-structural fields work differently, and already work today.**
`degree`, `neighbors`, and `graph_two_m` depend on whichever graph is
currently loaded, so they can't come from `build_annotation_table()`
(`negaverse/io/annotations.py`), which correctly takes no graph argument.
Instead, `negaverse/streams/rules.py::_RuleFilterBase.fit()` calls
`build_annotation_table()` for the static fields and then merges these three
graph-derived ones on top via `_augment_with_graph()`, using the live graph it
already has at `fit()` time — so any rule referencing them genuinely fires
today in the real pipeline (`RuleGradedFilter`/`RuleVetoFilter`), not just in
theory. The one place they'll *always* show as missing is
`scripts/validate_rules.py` — that script calls `build_annotation_table()`
directly, standalone, with no graph to augment with, so its report is a
property of that standalone checker's limited context, not of whether a
topology rule actually works in production.

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
when: "ligand.logp > 5 and protein.pocket_polarity > 0.5"
when: "a.evolutionary_coupling_score_with_b < 0.1"
when: "a.string_experimental_score_with_b < 0.15"
when: "a.surface_hydrophobicity > 0.44 or b.surface_hydrophobicity > 0.44"
when: "ligand.lineage_specificity == 'restricted_lineage' and disjoint(ligand.restricted_lineage_taxids, protein.lineage_taxids)"
```

The grammar can express `disjoint(a.neighbors, b.neighbors) and (a.degree *
b.degree) / a.graph_two_m < 0.01` (no-shared-neighbors + low expected edge
count) too — but don't write that rule: it's exactly `TopologyFilter`'s
`no_overlap`/`easy_negative` case, already computed more rigorously (L3 paths
too) as an independent stream. Encoding it again in YAML would double-count
that one topological fact against the same combined score. See Step 5.

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
  - `veto` — fires → the pair is dropped entirely; use only for hard
    biophysical or topological impossibilities. `RuleVetoFilter` picks up any
    `effect: veto` rule automatically — same zero-code-change mechanism as
    `RuleGradedFilter`, just a different stage — but think carefully before
    reaching for it for *positive* evidence: a plain "this pair is documented
    as interacting somewhere" fact (e.g. a database hit, or a very high
    STRING `experimental` score) isn't a biological-plausibility judgement,
    it's a membership fact, and belongs in `rules/sources.yaml`
    (`KnownPositiveVeto`) instead — see `rules/SOURCES.md` for why that's a
    different mechanism from this one. `ppi.yaml`'s
    `string_low_confidence_non_interaction` and `sources.yaml`'s
    `string_experimental_high_confidence` are the same STRING channel at
    opposite tails, deliberately split across the two mechanisms this way.

- `weight` ∈ `[0, 1]` sets how strong the push is. The graded score maps:
  - `safer_negative`:  `value = 0.5 + 0.5 · weight`  (weight 0.8 → 0.9).
  - `riskier_negative`: `value = 0.5 − 0.5 · weight`  (weight 0.8 → 0.1).

Calibrate `weight` to **how reliably the constraint implies non-interaction**, not
to how famous the paper is.

Typical ranges:
| weight | use when                            | example                                        |
|--------|--------------------------------------|------------------------------------------------|
| 0.8–1.0| physicochemical complementarity at the interface itself; the strongest, most general category | hydrophobicity mismatch; pocket size/polarity mismatch |
| 0.4–0.7| strong tendency, real (structural) exceptions | disjoint subcellular compartments |
| 0.1–0.3| weak prior / noisy signal           | coarse co-expression; mild topology mismatch |

**Why physicochemical constraints outrank compartment/localization ones, not the other way
around.** An earlier version of this table put "disjoint compartments" in the top bucket as a
"near-physical law" and hydrophobicity in the second tier — that ranking has it backwards.
Physicochemical-complementarity rules (hydrophobicity, pocket fit) describe a property of the
*interface itself* — the actual chemistry that drives or prevents binding — and the same kind of
reasoning generalizes across modalities (`pli.yaml`'s `physicochemical_incompatibility` is the PLI
analog of `ppi.yaml`'s `hydrophobicity_interface`). "Different compartment," by contrast, is a
property of *where the molecules happen to have been observed*, which is a structurally weaker,
narrower claim:
- **It's PPI-only.** Ligands and metabolites diffuse and get transported across compartments
  constantly, so "the ligand was annotated as extracellular" doesn't constrain whether it could
  reach a cytosolic pocket the way a real permeability or size constraint does — that's exactly
  why `pli.yaml` has no colocalization-style rule; it uses `permeability_class` instead.
- **It's softer than it sounds even within PPI.** Many proteins genuinely shuttle between
  compartments (transcription factors moving nucleus↔cytoplasm, stress-induced relocalization,
  cell-cycle-dependent movement), and GO cellular-component annotations are an aggregated snapshot
  across whatever studies happened to observe the protein — not a guarantee the two proteins never
  co-occur anywhere. "Few exceptions" oversells that.

The downstream ablation study (`docs/FILTER-EFFECTIVENESS.md` §3) found `hydrophobicity_interface`
showed a consistent, repeated real contribution across both datasets and both models, while
`colocalization_mismatch` showed ~0 net contribution — but that ablation ran while DRYAD's GO
cellular-component coverage was still 0% (an ENSG/UniProt ID mismatch in
`scripts/fetch_go_localization.py`, since fixed — it's natively UniProt-compatible, the loader just
had a clobber-not-merge bug preventing DRYAD's accessions from ever being written). With coverage
fixed (93% of DRYAD's UniProt accessions), a direct calibration
(`scripts/calibrate_colocalization_threshold.py`, n=300/class) shows the signal is real and, on
DRYAD specifically, comparable to or stronger than hydrophobicity's own calibrated numbers (AUROC
0.906 optimistic / 0.875 protein-disjoint; the shipped rule fires on 71% of true negatives while
misfiring on only 3% of true positives). It's markedly weaker on UPNA-PPI (AUROC 0.658 / 0.738;
40% of true positives wrongly flagged) — exactly the dataset-dependent "real exceptions" character
this band describes, since UPNA-PPI's negatives are topology-hard rather than compartment-based.
`rules/ppi.yaml` reflects this: `hydrophobicity_interface` at `weight: 0.8`,
`colocalization_mismatch` raised to `weight: 0.7` — the top of this band, evidenced by the strongest
calibration numbers of any rule in this category, but still below the physicochemical band because
the PPI-only/shuttling caveats above still hold regardless of any one dataset's numbers.

Topology-specific guidance:
- **Don't write a rule for "no shared neighbors + low configuration-model
  expected edge count."** `TopologyFilter` already floors exactly this case
  (`no_overlap`/`easy_negative`) at `value ≈ 0.98`, more rigorously than a YAML
  rule could (it also checks L3 length-3 paths, not just common neighbors).
  A YAML rule duplicating this — `rules/ppi.yaml` shipped one,
  `no_shared_neighbors_low_expected_edge`, and it was later removed for this
  reason — doesn't add evidence; it double-counts one topological fact,
  because `TopologyFilter` and the rule engine's output are independent
  streams that both feed the same combined score.
- Full L3/RA-based scoring is a more refined version of this same signal, and
  isn't expressible as a `when` rule anyway (see Step 3) — use
  `TopologyFilter`'s output for any topology-derived weighting, don't
  approximate it here.

Evolutionary-specific guidance:
- Strong absence of co-evolution signal across well-sampled orthologs → `weight`
  in the 0.5–0.7 range.
- Lineage-specific ligands without conserved binding partners or pockets → `weight`
  in the 0.4–0.6 range.
- Species-specific interaction loss with conserved proteins may be better encoded
  as `riskier_negative` for non-edges in the species where the interaction has
  been **lost** (not observed) — ortholog conservation elsewhere still argues
  against confidently calling that non-edge a safe negative.

Database-confidence-score guidance:
- A pair scoring below a database's own minimum reporting threshold (e.g.
  STRING `combined_score < 0.15`) aggregates absence of evidence across that
  database's channels (experiments, curated DBs, co-expression, text-mining,
  genomic context) — weight in the 0.5–0.6 range. Comprehensive, but not a
  physical-law-strength constraint the way disjoint compartments is.

**Empirical calibration can reverse your initial premise — don't discard the
field, flip the rule.** `hydrophobicity_interface`'s original premise ("low
mutual hydrophobicity → safer negative") was intuitive but wrong: calibrating
against two gold-standard PPI benchmarks (DRYAD, UPNA-PPI;
`scripts/calibrate_hydrophobicity_threshold.py`) showed real interactions
have *lower* exposed hydrophobicity than real non-interactions — the opposite
direction. Rather than conclude the field is useless, check whether the
reversed relationship has literature support (here: PPI interface hot spots
are enriched in aromatic/cation-pi residues, not classic hydrophobic ones —
Bogan & Thorn 1998) and flip `effect`/`when` accordingly. A field that
separates the classes with AUROC clearly below 0.5 is just as informative as
one clearly above 0.5 — only AUROC ≈ 0.5 (no separation either way) means
the field itself isn't useful for this rule.

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
  Proteins with no detectable co-evolution signal across orthologs are less
  likely to form a conserved physical complex, making a non-edge a safer
  negative.

rationale: >
  STRING's physical-subnetwork confidence score aggregates evidence across
  experiments, curation, co-expression, and text-mining; a pair scoring below
  STRING's own reporting threshold has no supporting evidence in any channel,
  making a non-edge a safer negative.

rationale: >
  Calibration against gold-standard PPI benchmarks showed real interactions
  have lower exposed hydrophobicity than real non-interactions — interface
  hot spots are enriched in aromatic/cation-pi residues, not classic
  hydrophobic ones. A pair where either protein shows unusually high exposed
  hydrophobicity is therefore a safer negative, not a riskier one.
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
  weight: 0.7                       # top of the strong-tendency band, PPI-only, real exceptions (see Step 5)
  flag: different_compartment
  rationale: >
    Two proteins that never share a subcellular compartment are unlikely to
    physically interact, so a non-edge between them is a safer negative.
  source: "GO cellular_component"
```

`validate_rules.py` → `[READY] colocalization_mismatch` (because `compartments`
is populated). This is the rule that powers co-localization today — no Python.