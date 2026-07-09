# negaverse — Implementation Plan

Actionable plan in the priority order agreed in the Discord spec (`local-docs/discord-spec.txt`),
grounded in the project brief, the flowchart, and the current prototype.

## Locked decisions (from the spec — don't relitigate)

- **Two modalities only:** PPI first (full vertical slice: candidate → filter → score → benchmark → viz), then PLI as the generalization module reusing the same architecture. Cut protein-RNA/DNA, full-text review, and the self-training loop.
- **Hourglass scoring:** cheap hard **vetoes at the front** (known-positive / DB screening) → **cheap graded filters in parallel** in the middle (topology, chemistry, co-localization — merged) → **gated LLM literature** at the back, expandable into the parallel layer if quality demands.
- **Filters are small independent modules** that each output `score + flags + provenance`. Keep them pluggable; avoid major refactors.
- **Topology = NetworkX + scipy.sparse, no graph DB.** No-overlap pairs are *not* skipped — treat as lower-topology-risk / easier negatives, and validate that assumption in benchmarking rather than hard-coding it.
- **Embeddings (ESM2 protein, MolFormer ligand) are for visualization, not filtering** — concatenate per-pair, UMAP to show hard negatives separate less than random negatives.
- **Visualization is simple and demo-focused:** show the tool separates datasets better than random.

## Current code → what changes

Have: `streams/{structured,embedding,literature}`, `pipeline`, `fusion`, `matching`, `io/`, `llm/`, `eval`, `schema`, `cli`. The existing "embedding" stream is actually graph-topology (Jaccard link-prediction) — it becomes the **topology filter** (ESM2/MolFormer are separate viz). The `Stream` ABC is already ~90% of the filter interface; the refactor below is light, not a rewrite.

---

## Phase 0 — Filter plugin layer + hourglass staging (foundation, ~0.5 day)

Enables everything downstream and directly answers Igor's "make it frictionless to add/modify filters" ask. Deliberately thin — extend the existing `Stream` abstraction, don't replace it.

1. **`Filter` contract** (evolve `streams/base.py`): add
   - `stage: {VETO, GRADED, GATED}` — where it runs in the hourglass.
   - `modalities: set[str]` — `{"ppi"}`, `{"pli"}`, or both.
   - `score(graph, u, v) -> FilterResult(value|None, veto, flags, evidence)` (already close to `StreamScore`).
2. **Registry** (`streams/registry.py`): `@register` decorator + `active_filters(modality, config)`; adding a filter = write a subclass + register it. No pipeline edits needed.
3. **Rework `pipeline.py` into the hourglass:** VETO pass (drop) → GRADED pass (parallel score + merge) → GATED pass (LLM on contested tail). Each filter's sub-score + evidence goes into provenance.
4. **`docs/ADDING-A-FILTER.md`** — the 20-line "how to add a filter" guide (contract, registration, config weight, test stub).

**Acceptance:** existing structured/topology/literature run unchanged through the new staged pipeline; a trivial new filter can be added in one file with no pipeline changes.

---

## Phase 1 — PPI vertical to demo shape (TOP PRIORITY)

The complete slice and the backbone of the presentation. In order:

1. **Human↔human positive set + in-space Negatome** (`io/`). Load HuRI or an IntAct/BioGRID human physical-interaction subset so Negatome golds share the node space. *Unblocks the benchmark and the gold-recall metric.*
2. **Structured DB screening — VETO** (`streams/structured.py` + `io/`). Union of known positives (input graph + 1–2 sources: IntAct/BioGRID) removed before scoring; Negatome as reference. This is the front of the hourglass.
3. **Topology filter — GRADED** (`streams/topology.py`, scipy.sparse). Configuration-model expected-edge prob `(k_u·k_v)/2m` + **L3** (length-3 paths via `A³`) + common-neighbor gate. No-overlap pairs → floor score + "easy-negative" bucket (cheap short-circuit, validated in benchmark). Replaces the Jaccard-only version.
4. **Co-localization filter — GRADED** (`streams/colocalization.py`, PPI, if easy). Different-compartment pairs are implausible interactors → safer negatives. Source from GO cellular-component / the SARS `table_s3` localization data. Cheap, high-signal.
5. **Gated literature — GATED** (already built). Keep as-is for now; add abstract retrieval in Phase 4 if time.
6. **Benchmark harness — the money shot** (`bench/`). Train a simple RF (and optionally a light GNN) on positives + (random negatives) vs positives + (negaverse negatives); report **AUROC / AUPRC** and the train/test gap. This is the hypothesis test ("do hard negatives beat random?").
7. **Visualizations** (`viz/`, matplotlib/plotly, reuse `out/negatives.jsonl` + `stats.json`):
   - degree-distribution overlap; shortest-path & common-neighbor KDEs (positive vs random vs hard);
   - **UMAP** of concatenated ESM2 embeddings (positive / random / hard) — the separability story;
   - similarity distributions (ESM2 cosine);
   - **Sankey / bar** "% removed per filter" (filter transparency).

**Acceptance:** one command runs PPI end-to-end and emits the AUROC/AUPRC comparison + the UMAP + the filter-transparency plot, with negaverse negatives measurably harder than random.

---

## Phase 2 — PLI module (second MVP, reuse the architecture)

Demonstrates generalization. New pieces only where the biology differs:

1. **Bipartite loader + entity resolution** (`io/pli.py`): proteins → UniProt, ligands → canonical SMILES + InChIKey; collapse duplicates.
2. **Chemical feasibility filter — GRADED** (`streams/chemistry.py`, RDKit): ECFP fingerprints + Tanimoto to the target's known binders, plus basic property compatibility. The PLI differentiator.
3. **PLI topology — GRADED:** same config-model/L3 idea on the bipartite protein-ligand graph.
4. **Gold + benchmark:** DUD-E / LIT-PCBA provide ready gold negatives → immediate validation + the PLI AUROC/AUPRC comparison. Scope to a target family to stay tractable (per the compute analysis).
5. **MolFormer embeddings for the PLI UMAP** (viz only).

**Acceptance:** the same pipeline/CLI, `--modality pli`, produces a matched negative set + benchmark on a DUD-E target family, proving the architecture generalizes.

---

## Phase 3 — Integrate, polish, buffer (~1 day, assume slippage)

End-to-end run on both modalities, the small viz dashboard, README + demo slides. Keep a day of buffer.

---

## Cross-cutting: extensibility notes (for Igor)

The frictionless-modification requirement, made concrete. Aim: **adding a filter or a data source touches one file and needs no pipeline edits.**

- **Filter contract + registry** (Phase 0): a new filter is a subclass declaring `stage`, `modalities`, and `score()`, plus `@register`. Config lists active filters and fusion weights; the pipeline discovers them.
- **Data loaders as plugins:** every loader returns a `TypedInteractionGraph` (+ optional feature tables), so a new positive/gold source is one loader function with a `path=` arg.
- **Declarative config** (`configs/*.yaml`): active filters, weights, gate threshold, LLM provider/model, n_eval/n_train — so tuning "what works" is a config edit, not a code edit (this is the tuning surface Igor wants to hand off).
- **Per-filter provenance + sub-scores** already flow to output → ablation is "drop a filter from config and re-benchmark."
- **Friction points to fix (noted, not blocking):** (1) fusion weights are currently a dict in code → move to config; (2) `match_on_type` is modality-specific → fold into the modality config; (3) the LLM gate criterion (contested set) is hard-coded → make it a pluggable selector; (4) no per-filter unit-test scaffold yet → add a `tests/filters/` template so each new filter ships with a test.

## Cross-cutting: structural-biology rule interface (for Lucyberry)

So sourced rules (hydrophobicity for PPI, ligand-pocket mismatch for PLI, physicochemical compatibility) feed both deterministic filters and the LLM without bespoke code each time:

- **One declarative rule format** (YAML/JSON): `{id, modality, applies_to, condition, evidence, weight}`. Deterministic filters evaluate `condition` against entity annotations; the literature/LLM filter receives the same rule text as context so its judgement is grounded in the same rules.
- This lets Lucyberry add a rule by editing a rules file — no code — and keeps the "dynamic, up-to-date reasoning" claim from the brief real. We'll stub the format in Phase 1 and populate it as rules are sourced.

---

## Mapping to the 5-day timeline

| Day | Focus |
|---|---|
| 0.5 | Phase 0 — filter plugin layer + hourglass staging + ADDING-A-FILTER doc |
| 1 | PPI benchmark (money shot) + human↔human positives (unblocks it) |
| 2 | PPI visualizations (UMAP, distributions, Sankey) + structured-veto + topology upgrade |
| 3 | Co-localization filter; **freeze scope**; start PLI (loader + chemistry filter) |
| 4 | PLI DUD-E benchmark; strengthen filters; abstract retrieval for literature if time |
| 5 | Integrate both modalities, viz dashboard, README/slides, buffer |

**Highest-leverage first move:** Phase 0's thin plugin layer (so the tuning surface exists for handoff) immediately followed by the Day-1 PPI benchmark — the hypothesis test everything else supports.
