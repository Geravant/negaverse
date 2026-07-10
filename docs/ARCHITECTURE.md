# negaverse — Architecture Design (v0.1)

*A universal tool for generating compatible, matched negative datasets from interaction-based inputs (protein–protein, protein–ligand, protein–RNA, protein–DNA, …).*

Status: brainstorm draft · Anchor vertical: protein–protein interactions (PPI) · Claude Life Sciences Hackathon

---

## 1. The problem, precisely

Biomedical interaction datasets record what *does* interact. What *doesn't* interact is rarely published and almost never curated at scale, so machine-learning pipelines fabricate negatives — usually by randomly pairing entities not known to interact. This is the single largest source of avoidable error in interaction prediction, for three reasons that the literature makes concrete:

- **False negatives contaminate the label.** A random non-edge may simply be an interaction nobody has tested yet. Negatome's authors put it plainly: *"just because two proteins have not yet been reported as interacting does not mean that they actually do not interact."* With human PPIs, the tested fraction is a small slice of the estimated total, so a random pair is often an untested positive rather than a true negative.
- **Naive negatives leak shortcuts.** Park & Marcotte show that random sampling encodes the degree distribution ("hubbiness"): a model can score pairs well by learning *"this protein is a hub"* rather than any real interaction signal. Composition and abundance biases do the same.
- **The easy/hard tradeoff is real and counterintuitive.** Koyama et al. show that negatives chosen to be *very distant* from positives are "easy" and actually **hurt** generalization to external data; the *informative* negatives sit near the decision boundary.

negaverse's job is to replace "random non-edge" with a **matched, confidence-scored, provenance-carrying negative** that is defensible on biological, physical, topological, and literature grounds — and to do it through one universal engine that specializes per interaction type.

## 2. Design principles (each earns its place from a paper)

**P1 — Two negative sets, never one (Park & Marcotte 2011).** There are two *fundamentally different* sampling problems, and conflating them invalidates results. Evaluation negatives must be **unbiased and representative** of the true negative population so benchmark scores generalize; training negatives may be **deliberately biased/hard** because the goal is to train well, not to estimate population performance. negaverse therefore emits **two labeled products from the same run** — an `eval` set and a `train` set — and refuses to merge them.

**P2 — Mine informative, near-boundary negatives (Koyama et al. 2023).** Don't just push negatives far from positives. Use a controllable "distance to the positive manifold" so we can dial hardness, and prefer ambiguous, near-boundary pairs for training. A self-training loop re-scores candidates and promotes the informative ones.

**P3 — Anchor on gold negatives (Negatome 2.0).** Experimentally- and literature-validated *non*-interactions are rare and precious. Inject them as high-confidence anchors, use them to calibrate the confidence scale, and use them as the tool's own evaluation ground truth.

**P4 — Every negative is auditable.** Each emitted pair carries a confidence score **and** a provenance trail (which filters fired, which streams scored it, what evidence). Because biology gets revised, negatives must be re-scorable as knowledge and models improve — provenance makes that possible.

**P5 — Universal core, typed plugins.** One graph engine; interaction-type specifics (features, filters, gold sources) live behind a plugin interface so PPI, protein–ligand, protein–RNA and protein–DNA share the pipeline and differ only where the biology differs. Concretely, the orchestrator never names a specific filter or assumes a PPI statistic: the **hardness signal** comes from whichever GRADED filter declares `provides_hardness` (topology, for PPI); the **disagreement routing** compares the stream pairs in `PipelineConfig.disagree_pairs` (default the PPI graph views); and the **eval-matching confounder** is `PipelineConfig.match_weight_fn` (graph degree by default). A new modality supplies its own without touching `pipeline.py`.

## 3. Core abstraction

negaverse operates on a **typed interaction graph**:

- **Nodes** are typed entities: `Protein`, `Ligand`, `RNA`, `DNA`. Each node carries type-specific features supplied by a plugin (sequence, structure, chemical descriptors, annotations).
- **Edges** are observed positive interactions from the input dataset (optionally weighted by assay confidence).
- A **negative** is a proposed non-edge `(u, v)` accompanied by a confidence score in `[0,1]`, a hardness score, a set/mode label (`train` | `eval`), and provenance.

The engine is agnostic to whether the graph is bipartite (protein–ligand) or homogeneous (protein–protein); plugins declare the node types an edge connects and which feature/filter/gold providers apply.

```
EntityPlugin
  ├─ features(node)            → sequence / structure / descriptors / annotations
  ├─ hard_filters()            → rules that EXCLUDE likely-false-negatives
  ├─ plausibility_scorers()    → biological / physical / topological priors
  ├─ gold_negative_sources()   → curated non-interactions for this type
  └─ embedders()               → sequence / structure / graph representations
```

## 4. The pipeline

Data flows through six layers. Layers 2–3 remove or down-weight likely false negatives; Layer 4 anchors; Layer 5 shapes the distribution; Layer 6 refines and calibrates.

**Layer 1 — Candidate generation.** Enumerate non-edges of the positive graph. Full enumeration is `K(K−1)/2 − P` and usually intractable, so we sample a working pool with configurable strategy (uniform, degree-aware, or embedding-guided — see §5.3), sized to the requested train/eval volumes.

**Layer 2 — Hard exclusion (false-negative removal).** Drop candidates that are *probably actually positive*:
- present as a positive in any *other* interaction database (union of external sources);
- **homology/paralog transfer** — if A–B interact, close homologs A′–B′ likely interact, so exclude them;
- shared-complex membership or strong co-annotation;
- high family–family interaction propensity.
A candidate surviving Layer 2 is "not-known-positive and not-obviously-positive."

**Layer 3 — Implausibility scoring (the "reasoning" layer).** Score how *biologically/physically plausible* an interaction would be — a plausible-but-unobserved pair is a *risky* negative; an implausible pair is a *safe* one. Signals: subcellular co-localization (proteins that never share a compartment can't bind), tissue/cell-type co-expression, GO/pathway relationships; physical/topological compatibility (size, interface, docking feasibility; for ligands, chemistry/ADMET rules). This layer produces the **structured-filter stream's** contribution to the score (§5.1).

**Layer 4 — Gold-negative injection.** Merge curated true negatives (Negatome and equivalents) as high-confidence anchors and as a held-out calibration/evaluation set for negaverse itself.

**Layer 5 — Matching & balancing.** Shape the negative distribution to match the positives on confounders so the downstream model can't cheat: **degree-match** (defuse hubbiness, per P1), **similarity-match** to positives at a *controlled distance* (per P2), and stratify on any declared covariate (family, length, assay). This is where the `eval` set is made representative and the `train` set is made informative.

**Layer 6 — Confidence assignment & iterative refinement.** Fuse the three scoring streams (§5) into a calibrated confidence, then run an optional **self-training loop** (Koyama): train a probe model on current labels, re-score the candidate pool, promote near-boundary pairs into the hard-negative set, and flag high-scoring "negatives" as suspected false negatives for review.

## 5. The three scoring streams

Confidence is not a single heuristic; it's a fusion of three independent views. Igor's addition — a geometric/graph-embedding stream — sits alongside the structured and literature streams as a co-equal third. Keeping them independent means each can be validated, ablated, and trusted (or distrusted) on its own.

### 5.1 Structured-filter stream (rules & databases)
Deterministic, explainable signals from structured biology: known-positive exclusion, homology transfer, localization/co-expression, GO/pathway, family propensity, physical/topological compatibility (Layers 2–3). Strengths: transparent, no training data needed, high precision on exclusions. This stream can hard-veto (a known positive is never emitted as a negative) as well as contribute graded plausibility.

### 5.2 Literature-reasoning stream (embed + retrieve + reason)
A modernized Negatome. Where Negatome 2.0 text-mined *negated* predicate-argument structures ("A does **not** bind B") with Excerbt/Senna plus manual curation, we use literature embeddings + retrieval + an LLM reasoning step:
- embed abstracts/full text; retrieve passages about the candidate pair and their families;
- an LLM extracts explicit *non-interaction* evidence, *positive* evidence (→ route to false-negative flagging), and *absence* of evidence;
- output an evidence-weighted score with **citations** feeding the provenance trail.
This is the layer that most directly answers Lucy's "literature embedding/search rules," and it degrades gracefully — no evidence simply means this stream abstains rather than guesses.

### 5.3 Graph / geometric-embedding stream (Igor's extension)
Learned representations give a *topological and geometric* view the other two streams can't:
- **Graph embeddings / link prediction** over the interaction network (and a wider heterogeneous bio-knowledge graph): node2vec / GNN / KG-embedding produce a link-prediction score for `(u,v)`. **High score ⇒ likely false negative** (exclude/flag); **low score ⇒ safe negative**; **mid/near-boundary ⇒ the informative hard negatives P2 wants.**
- **Geometric/sequence/structure embeddings** of the entities themselves (e.g. protein language-model and structure embeddings) give a **continuous distance to the positive manifold**, which is exactly the knob Layer 5 uses to sample negatives at a *controlled* hardness instead of the crude "far = easy" heuristic Koyama warns against.
- The embedding space also powers **embedding-guided candidate generation** in Layer 1 (sample where hard negatives actually live) and **degree-matching** in embedding space.

**Fusion.** The three stream scores combine into one calibrated confidence with per-stream weights (learnable against the gold-negative anchors from Layer 4). Streams can abstain; a hard veto from the structured stream overrides. The fused output records each stream's contribution so a user can see *why* a pair is a negative.

## 6. Transductive vs inductive splitting

Lucy explicitly calls out both settings, and the split is where leakage kills benchmarks:

- **Transductive** — all entities appear in training; we predict unseen *edges*. Negatives are sampled over the known node set; the eval negatives (P1) must be degree/covariate-matched.
- **Inductive** — test entities are **unseen** in training. The splitter partitions *nodes* first, forbids any train/test edge or negative from bridging the split, and (critically) prevents homology leakage — a test protein that is a close homolog of a train protein is quasi-seen. negaverse offers a homology-aware inductive splitter as a first-class option.

## 7. Output contract

A negaverse run emits the input positives plus a matched negative set in a format aligned to the input, where each negative record carries:

| field | meaning |
|---|---|
| `u`, `v` | the entity pair (typed IDs) |
| `mode` | `train` or `eval` (P1 — never mixed) |
| `confidence` | calibrated `[0,1]` that the pair is a true non-interaction |
| `hardness` | distance-to-positive-manifold percentile (for curriculum/sampling) |
| `streams` | per-stream sub-scores: `structured`, `literature`, `embedding` |
| `provenance` | filters fired, evidence citations, source DBs, versions |
| `flags` | e.g. `suspected_false_negative`, `gold_anchor` |

Everything is versioned (data sources, model, ruleset) so a dataset can be regenerated or re-scored when biology is revised (P4).

## 8. MVP scope — the PPI vertical

Build the universal core but validate one vertical end-to-end:

- **Entity plugin:** `Protein` — features from sequence embeddings (protein language model) + structure where available; localization/co-expression/GO annotations.
- **Positives:** a standard human PPI set; **gold negatives:** Negatome 2.0 for anchoring and calibration.
- **Streams for MVP:** structured (known-positive exclusion + localization + homology) and embedding (network link-prediction + PLM-distance) as the two workhorses; literature stream stubbed behind the same interface, filled in as the differentiator.
- **Concrete test case in-repo:** the SARS-CoV-2 host–pathogen interactome spreadsheets (`local-docs/…sars-cov2-spreadsheets/`: baits, preys, drugs, network) give a real, self-contained PPI (and protein–ligand) dataset to dogfood the pipeline against.
- **Validation:** hold out Negatome gold negatives and a slice of positives; check that (a) negaverse ranks gold negatives high-confidence, (b) held-out positives are *not* emitted as high-confidence negatives (false-negative rate), and (c) a downstream PPI model trained on negaverse negatives generalizes better on external data than one trained on random negatives — the Koyama success criterion.

## 8.5 Vertical MVP — the walking skeleton

§8 says *what* the PPI vertical contains; this says *how thin* to build it so it runs end-to-end in hackathon time and produces one undeniable result. The guiding move is a **walking skeleton**: the thinnest path from a positive graph to a scored `train`/`eval` split that exercises every architectural seam, then thickens only where the demo needs it.

**Keep all three streams — that fusion is the thesis, not a feature to defer.** The cut is *depth within a stream* and *breadth of the platform*, never the number of streams. Each stream ships in its cheapest incarnation that still contributes a real, independent view:

- **Structured (thin):** known-positive exclusion (union of 1–2 external DBs) + a localization/co-expression lookup. Deterministic, no training, can hard-veto. This is the non-negotiable floor — a known positive must never surface as a negative.
- **Embedding (thin):** node2vec link-prediction over the positive graph + PLM-distance as the hardness knob. One cheap component delivers both hard-negative mining and the controlled-distance matching Layer 5 needs; runs in seconds on a graph this size.
- **Literature (thin, gated):** Claude runs **only on the top-K contested / near-boundary pairs**, emitting a cited "safe negative / suspected false negative" card. This abstains on everything else — which resolves the §10 cost/latency question (LLM-per-million-pairs is never attempted) while still putting the stream's differentiated reasoning in the demo.

**Fusion (thin):** a weighted combine with per-stream sub-scores retained; weights fit against the Negatome anchors (Layer 4). Streams may abstain (literature abstains on all but the gated pairs); a structured veto overrides. This is the whole provenance/auditability story, and it's cheap.

### The one number that has to move

Everything above serves a single claim (the Koyama success criterion, §8):

> A downstream PPI model trained on negaverse negatives generalizes **better on external data** than one trained on random negatives — while negaverse ranks Negatome golds high-confidence and does *not* emit held-out positives as confident negatives.

If the demo shows that curve, the argument is won. Anything not on the path to that number is a cut candidate.

### In-scope vs deferred

| In the MVP | Deferred (stub the interface, name it in the demo) |
|---|---|
| `Protein` plugin only | `ligand` / `rna` / `dna` plugins; the full universal surface |
| Layer 2 exclusion (known-positive + homology) | Layer 3 physical/topological + docking feasibility |
| All 3 streams, thin; simple weighted fusion | self-training loop; learnable per-stream weights beyond anchor-fit |
| Degree-matching (Layer 5) | embedding-guided candidate sampling (Layer 1); similarity-match tuning |
| Transductive split + `train`/`eval` products | inductive homology-aware splitter |
| node2vec + PLM-distance | GNN / KG-embedding over a wider bio-knowledge graph |

### Two traps to design around

1. **Data-space mismatch.** Negatome is **human–human** non-interactions; the in-repo SARS-CoV-2 Gordon set is **viral–human bipartite**. Calibrate and validate confidence on a **human PPI positive set** (Negatome golds are in-distribution there); use the SARS-CoV-2 interactome as the *live-generation demo* dataset, not the calibration target. Keep these two roles explicit.
2. **Circularity.** If the embedding stream generates candidates *and* scores confidence *and* the downstream validation model is also embedding-based, the win is self-fulfilling. Keep eval-set scoring independent of the generation signal, and validate with features/model distinct from the generation stream.

### Demo, in three acts

1. **Generate:** point negaverse at the SARS-CoV-2 network; watch it emit a matched `train`/`eval` negative set with per-pair provenance and the three sub-scores.
2. **Trust:** show negaverse ranks Negatome golds high-confidence, flags a planted held-out positive as a *suspected false negative*, and open one literature card with its citations.
3. **Prove:** the A/B curve — a PPI classifier trained on negaverse negatives vs random negatives, evaluated on the held-out/external set.

### Suggested build order

Skeleton first (graph load → uniform-sample candidates → known-positive exclusion → degree-match → emit `train`/`eval` with an empty provenance record), verified end-to-end on the small SARS-CoV-2 graph. Then thicken one seam at a time in demo-value order: embedding stream → fusion + Negatome calibration → validation harness (the number) → gated literature cards. Each thickening keeps the pipeline runnable, so there is always a demoable artifact.

## 9. Proposed module layout

```
negaverse/
  core/
    graph.py            # typed interaction graph
    pipeline.py         # Layer 1–6 orchestration
    splitting.py        # transductive / inductive (homology-aware)
    fusion.py           # stream combination + calibration
    schema.py           # negative record / output contract
  streams/
    structured/         # rules, DB exclusion, plausibility
    literature/         # embed + retrieve + LLM reasoning
    embedding/          # graph + geometric embeddings, link prediction
  plugins/
    protein.py          # MVP
    ligand.py  rna.py  dna.py   # later
  gold/                 # Negatome & other curated negatives
  eval/                 # benchmarks, ablations, false-negative audits
```

## 10. Open questions to resolve next

- **Confidence calibration:** with so few gold negatives, how do we calibrate the `[0,1]` scale — Negatome anchors only, or semi-supervised via the self-training loop?
- **External positive DBs for Layer 2:** which union of sources defines "known positive" for exclusion, and how do we version it?
- **Literature stream cost/latency:** LLM-per-pair doesn't scale to millions of candidates — do we gate it to only near-boundary or contested pairs?
- **How "universal" at MVP:** is the ligand/RNA/DNA plugin surface designed now (interfaces only) or deferred until the PPI vertical proves out?
- **Downstream integration target:** is there a specific model/benchmark negaverse must plug into, which would fix the output format concretely?

---

*Grounding papers: Park & Marcotte, "Revisiting the negative example sampling problem for predicting PPIs," Bioinformatics 2011 · Koyama et al., "Improving Compound–Protein Interaction Prediction by Self-Training with Augmenting Negative Samples," JCIM 2023 · Blohm et al., "Negatome 2.0," Nucleic Acids Research 2014.*
