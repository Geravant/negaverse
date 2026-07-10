# Information-geometry (IG) features as negaverse gates

Principles ported from the sinain-hud lecture *["Memory as Geometry"](https://github.com/…/sinain-hud/blob/main/docs/lecture-memory-as-geometry.md)*
into negaverse's negative-selection pipeline. Each is a small, self-contained
mechanism with a **prototype** in [`negaverse/ig/`](../negaverse/ig/), unit tests
in [`tests/test_ig_features.py`](../tests/test_ig_features.py), and an
**evaluation harness** in [`scripts/eval_ig_features.py`](../scripts/eval_ig_features.py)
(results → `out/ig_eval.json`).

> The lecture's throughline: *meaning is geometry.* negaverse has the same shape
> in different clothes — a protein pair is a point (graph-topology features +
> biological annotations), and it already names the axis: `hardness =
> distance-to-positive-manifold percentile` (`schema.py`). So the "sphere of
> meaning" here is the interactome, and the "frozen background cloud" is the
> positive manifold (risky) or the Negatome gold negatives (safe).

One asymmetry to keep in mind throughout: sinain wants facts *far* from the
background (surprising = keep). negaverse wants **two** things — *safe* negatives
(far from the positive manifold, high confidence) for the **eval** set, and
*hard* negatives (near the boundary but still true non-edges) for the **train**
set. Several features below therefore become two gates, one per product.

---

## The mapping at a glance

| Lecture chapter | negaverse gate it becomes | Prototype | Verdict from evaluation |
|---|---|---|---|
| Ch4 Entropy-weighted fusion | Per-candidate stream weights: trust the decisive stream | `ig/entropy_fusion.py`, wired into `pipeline.py` (`fusion_mode="entropy"`) | **Strong — but only with stream-reported confidence.** Scalar proxy backfires. |
| Ch5 Determinantal (DPP) selection | Diverse eval/train set curation | `ig/dpp.py` | **Modest on real embeddings; removes the worst near-duplicates.** |
| Ch1 Surprisal vs a frozen cloud | Resemblance to the positive manifold — over graph *and* sequence (ESM2) feature spaces | `ig/surprisal.py`, `scripts/eval_esm2_manifold.py` | **Gold-cloud = null (0.47). Graph manifold real (0.88). ESM2 sequence manifold is the *independent* axis (corr ~0.2) → 3-way fusion 0.81→0.88.** |
| Ch7 Geometric routing (relative margin) | Margin between positive-like and negative-like clouds, not an absolute cutoff | `ig/margin.py` | Component of C's relative-margin score (0.95). |
| Ch6 Recurrence / Hawkes | Corroboration: confidence = independent streams that echo the verdict | (design; uses `stream_disagreement`) | Precondition (stream independence) holds in negaverse. |
| Ch2 Canonicalization / medoid | Dedup near-identical candidate negatives (paralogs, same complex) | (design) | Subsumed by DPP's diversity term. |
| Ch8 Scaffolding / sufficient statistic | Code computes the structural facts; LLM only verbalizes | (already partly in `streams/literature.py` + rules) | Sharpen, don't rebuild. |
| Ch3 Change-points + "supersede, not delete" | Version-aware freshness; never silently drop a flagged pair | (already: `suspected_false_negative`) | Philosophy already in the codebase. |
| Ch9 The honest chapter | Not a gate — the design discipline for all of the above | — | Structural > semantic; find the binding rung; abstain. |

---

## The four prototyped features

### 1. Entropy-weighted fusion (Ch4) — `ig/entropy_fusion.py`

**Idea.** sinain's read-side fusion (Cronen-Townsend 2002 query-clarity): trust
each channel in proportion to how *sharply* its scores spike. In negaverse each
stream returns a scalar `value ∈ [0,1]` = confidence-it's-a-true-non-interaction.
The distributional analog of peakedness for a scalar is the **binary entropy**
`H(value)`: maximal at 0.5 (a guesser), zero at the extremes (committed). Weight:

```
w_stream = base · (1 + λ · decisiveness),   decisiveness = 1 − H_bin(value)
```

`λ=0` recovers the current fixed-weight mean exactly, so it is a **non-breaking,
opt-in** change. Wired into the pipeline via `PipelineConfig(fusion_mode="entropy",
fusion_lam=…)`; default stays `"mean"`.

**Where it slots in.** `fusion.py::fuse` (GRADED merge) and
`pipeline.py::_fuse_confidence` (final confidence). A stream may also publish its
own peakedness in `evidence["confidence"]`, which overrides the scalar proxy.

**Evaluation (controlled, with ground truth).** A *good* witness (informative,
noisy) fused with a *flaky* specialist that is correct on half the pairs and
**confidently wrong** on the other half:

| strategy | AUROC |
|---|---|
| good witness alone | 0.963 |
| mean fusion (+ flaky) | 0.809 |
| entropy fusion, **scalar proxy** | 0.802 — *backfires* |
| entropy fusion, **reported confidence** | **0.940** — *the win* |

**The honest finding (belongs in the design):** the scalar `|value−0.5|` proxy
**cannot distinguish "confident because informed" from "extreme by luck"** — it
up-weights the confidently-wrong witness and makes fusion slightly *worse*. The
real payoff needs a stream to **report its own competence** (`evidence
["confidence"]`), exactly as sinain weights by a full *distribution's* entropy,
not one Bernoulli scalar. **Action:** have `TopologyFilter` (from its L3/RA
spread) and `LiteratureFilter` (from the LLM's answer distribution) populate
`evidence["confidence"]`; only then turn `fusion_mode="entropy"` on.

### 2. Determinantal (DPP) selection (Ch5) — `ig/dpp.py`

**Idea.** Don't fill the negative budget with six copies of one region; fill it
with pairs pointing in *different directions*. "Span a chunk of space" = the
determinant of the Gram matrix, so a DPP selects quality-weighted, mutually-
diverse subsets: `P(S) ∝ det(L_S)`, `L_ij = q_i·q_j·cos(x_i,x_j)`. `q` = confidence
(eval) or hardness (train); the cosine is over a pair-embedding.
`greedy_map_dpp` is the standard fast greedy MAP (Chen et al. 2018).

**Where it slots in.** An alternative selector in `matching.py` (`hard_train` /
`degree_matched_eval`).

**Evaluation (real HuRI, spectral Hadamard pair-embeddings).** Pool of 1500
link-like candidate non-edges; compared at DPP's realized size (n=48, the
embedding's effective rank):

| selector | mean quality | mean pairwise cos | **max pairwise cos** | unique proteins | clusters |
|---|---|---|---|---|---|
| top-k by quality | 0.347 | 0.354 | **0.998** | 89 | 10 |
| DPP | 0.310 | 0.359 | **0.987** | 91 | 11 |

**Finding:** on real HuRI embeddings the *aggregate* diversity gain is modest,
but DPP reliably drops the **near-exact duplicates top-k admits** (max cosine
0.998 → 0.987 — top-k grabbed two essentially identical negatives) for a small
quality cost. The effect grows when the high-quality tail is more clustered,
which is exactly the regime real hard-negative mining produces (dense
subgraphs). Worth it for the train set; tune the quality/diversity trade before
using on eval.

### 3. Gold-negative surprisal (Ch1) — `ig/surprisal.py`

**Idea.** Freeze a background cloud and score a pair by its top-k-mean cosine to
it. Two backgrounds, one primitive (`background_similarity`):
* **gold negatives (Negatome)** — resemblance = a *confident safe negative*.
* **the positive manifold** — resemblance = a *suspected false negative* → flag.

**Evaluation.** Two passes. *Leaky* (experiment C): embeddings built on the full
HuRI graph. *Leakage-free* (experiment C2): embeddings and features built on a
**train-only subgraph**; held-out positives vs gold negatives never enter the
representation, exactly as `bench/benchmark.py` does. Both on real HuRI +
Negatome (349 gold pairs mapped into HuRI space).

| signal | AUROC |
|---|---|
| gold-cloud resemblance separates gold negatives | **0.47 — null** |
| positive-manifold, *leaky* embeddings | 0.93 |
| positive-manifold, **leakage-free** (spectral Hadamard) | **0.884** |
| negaverse's own `TopologyFilter` risk (baseline) | 0.885 |
| **topology risk + surprisal, fused** | **0.902** |

**Findings (revised by the leakage-free re-check):**
1. **Gold-cloud resemblance is a null** (0.47 ≈ chance) — Negatome negatives don't
   cluster in embedding space. Don't build that stream.
2. **The positive-manifold signal is real and mostly survives** de-leaking
   (0.93 → 0.884); the leak was worth only ~0.05 AUROC.
3. **But it does not beat topology *alone*** — negaverse's existing `TopologyFilter`
   already scores 0.885 on the identical task. A naive "promote surprisal to a
   confidence stream" would be **redundant**.
4. **They are complementary, though** — correlation is only **+0.64**, and
   **fusing** topology risk with spectral surprisal lifts AUROC to **0.902**,
   above either alone. The spectral embedding captures *global* graph structure;
   topology's L3/RA capture *local* structure. That ~+1.6-point lift is the real,
   evidence-backed improvement.
5. **Representation matters:** the **Hadamard** operator on **spectral** node
   embeddings is the right recipe (0.88); `avg` is worse (0.82), `l1` much worse
   (0.69), and raw hand-crafted **topological features under cosine-surprisal
   invert** (0.18 — anti-correlated), so don't use them here. `k` (10/25/50) is
   insensitive.

**Revised action:** add a spectral-manifold surprisal as a *complementary*
GRADED stream (Hadamard, k≈10) and let fusion combine it with topology — not as a
replacement for the hardness axis. Skip the gold-negative and topological-feature
variants. (The 0.902 above is a plain z-score average; a fitted or reported-
confidence fusion should do better still — ties back to feature 1.)

#### 3b. The sequence (ESM2) manifold is the genuinely *independent* axis

The graph-spectral surprisal correlated 0.64 with topology because both are built
from the same graph. The predicted improvement was a manifold on a *different*
feature space — a protein's **sequence** (ESM2) rather than its interaction
partners. Tested on the **DRYAD PPI** benchmark (which ships sequences +
precomputed ESM2-t6 embeddings + matched positive/negative controls), same
leakage-free surprisal recipe, via [`scripts/eval_esm2_manifold.py`](../scripts/eval_esm2_manifold.py)
(→ `out/esm2_manifold_eval.json`), stable over 4 seeds:

| axis | single-axis AUROC | source |
|---|---|---|
| ESM2 **sequence** manifold | 0.75 | pretrained ESM2-t6 (320-d) |
| spectral **graph** manifold | 0.81 | SVD of train positives |
| negaverse topology risk | 0.81 | L3/RA, local |

| axis pair | correlation |
|---|---|
| spectral ~ topology | **+0.70** (redundant — same graph) |
| **ESM2 ~ spectral** | **+0.20** (nearly independent) |
| **ESM2 ~ topology** | **+0.17** (nearly independent) |

| fusion (z-score average) | AUROC |
|---|---|
| ESM2 + spectral | **0.865** |
| ESM2 + spectral + topology | **0.881** |

**Findings:**
1. ESM2 alone is *weaker* than the graph arm on DRYAD (0.75 vs 0.81) — expected,
   since DRYAD is graph-sparse and this is the *smallest* ESM2 (8 M params); a
   larger ESM2 (t33, 1280-d) would raise the sequence floor.
2. But ESM2 is **genuinely orthogonal** — correlation ~0.2 with both graph axes,
   versus 0.70 between the two graph axes. It carries the independent signal the
   graph-derived streams cannot.
3. So **fusion pays off far more than before**: best single axis 0.81 → **0.88**
   combined (+6–7 points, vs the +1.6 from fusing the two correlated graph axes on
   HuRI). This is the "three independent lenses" thesis confirmed with numbers:
   network (topology) + relational (spectral) + molecular (ESM2).
4. **Representation is operator-specific:** ESM2 pairs want `concat`(min,max) or
   `avg` (0.74–0.76); `hadamard` *destroys* the dense semantic vector (0.61) —
   the opposite of the graph manifold, where Hadamard won. Match the operator to
   the space.

**Action:** ESM2-manifold surprisal is the highest-value new stream — it is the
independent axis. On sequence-rich modalities (PLI: ESM2 for protein, MolFormer
for ligand) it should matter even more. Fuse all three; don't pick one.

#### 3c. Is the `suspected_false_negative` flag valid? (independent check)

The `ManifoldSurprisalFilter` (`streams/manifold.py`) ships the flag first. To
trust it we graded it against an *independent* yardstick — real interactions it
never saw. Split HuRI positives train/held-out; fit the manifold on train only;
the held-out edges are **hidden positives** (a negative-generator that emits one
has made a false negative). Ask whether the flag catches them, via
[`scripts/eval_manifold_flags.py`](../scripts/eval_manifold_flags.py):

| question | result |
|---|---|
| flag signal separates hidden positives from clean negatives | AUROC **0.896**, flag **12×** more likely on a hidden positive |
| topology alone, same task (baseline) | AUROC **0.930** |
| manifold on pairs topology calls *safe* (its blind spot) | AUROC **0.68** |
| eval-set cleanliness (5% contamination) | flag halves it (460 → 250 hidden) at a ~4% clean-flag cost |

**Findings:** the flag is **valid** — it genuinely catches hidden false negatives,
measured against biology it never saw (not the circular trap). But it does **not
beat topology alone** (0.896 < 0.930); topology already does this job. Its real
contribution is **complementary**: in topology's blind spot it still finds hidden
positives at 0.68. So the manifold's unique value lives exactly where it
*disagrees* with topology — which is the evidence-backed case for using it as a
second flag layered on topology, and for routing topology-vs-manifold
disagreements to the gated review, rather than as a standalone signal.

### 4. Relative-margin gating (Ch7) — `ig/margin.py`

**Idea.** When two exemplar clouds overlap, an absolute threshold fails; gate on
the *margin* between "looks like the negative cloud" and "looks like the positive
cloud," with a floor so a pair far from both **abstains** rather than guessing.
negaverse's absolute cutoffs — topology's `0.98` floor for no-overlap pairs, and
the rules' fixed `0.5 ± 0.5·weight` map — are the natural targets. The relative
margin is the third row of experiment C (AUROC 0.95), carried by the positive-
manifold term.

---

## The Ch9 discipline (this governs every gate above)

The lecture's honest chapter is the most transferable part, and it is not a gate
— it is how to judge one:

1. **Structural selection beats semantic selection.** Duration worked because
   operands were selected by a *field* (`occurred_at`); COUNT/SUM failed because
   "is this a bike expense?" is a semantic judgment code can't make. negaverse's
   strongest rule, co-localization, is strong for the same reason —
   `disjoint(a.compartments, b.compartments)` reads a GO field. The weaker rules
   (`surface_hydrophobicity`, `logp`, still `TODO — source to confirm`) lean
   semantic. **Rule:** trust structural gates *as gates*; treat semantic ones as
   soft flags, never vetoes.
2. **Find the binding rung before adding a filter.** The ladder here is
   `candidate-generation → gate/selection → fusion → export-cap`. A new gate is
   worthless if candidate generation never produced the boundary pairs, or the
   annotation field the rule needs is absent (rules already *abstain until
   fields are supplied* — a recall limit). Diagnose which rung binds first.
3. **Abstain, don't guess; supersede, don't delete.** Both already live in
   negaverse (`StreamScore.abstains`; `suspected_false_negative` instead of a
   silent drop). Any new gate inherits this.

---

## Running it

```bash
python3 -m tests.test_ig_features            # unit tests for the mechanisms
PYTHONPATH=. python3 scripts/eval_ig_features.py   # A (synthetic) + B, C (real HuRI/Negatome)
```

Experiments B and C skip cleanly if `local-docs/huri`, `local-docs/negatome2`,
or `local-docs/mappings` are absent (see the README for how to fetch them).

## Recommended order to productionize

1. **Stream-reported confidence, then entropy fusion — DONE.** `TopologyFilter`
   (structural support) and `LiteratureFilter` (vote unanimity) now populate
   `evidence["confidence"]`; the manifold filters report their peakedness too.
   `_fuse_confidence` prefers reported confidence over the scalar proxy (which
   alone backfires), so `PipelineConfig(fusion_mode="entropy")` is safe to enable.
   (A: 0.81 → 0.94 with reported confidence.) See `tests/test_reported_confidence.py`.
2. **Manifold surprisal as GRADED streams — DONE.** Both shipped as opt-in
   filters over one shared base (`streams/manifold.py`): (a) **SequenceManifoldFilter**
   — ESM2 (or any per-protein) embeddings, `concat` operator, the *independent*
   axis (corr ~0.2), buildable from a `.npz` (`scripts/build_esm2_embeddings.py`,
   `io.load_embeddings_npz`) and abstaining for unembedded nodes; the natural fit
   for sequence-rich PLI. (b) **ManifoldSurprisalFilter** — spectral graph, `Hadamard`,
   complementary to topology (corr 0.64). Verified end-to-end on DRYAD with all
   three axes fused. Skip the gold-negative and raw-topological-feature variants.
3. **DPP for the train set.** Swap `hard_train`'s top-k for `greedy_map_dpp` over
   pair-embeddings; keep top-k for eval until the quality/diversity trade is
   tuned. (B: removes the 0.998-cosine duplicates.)
4. **Disagreement-driven GATED routing — DONE.** Pairs where topology and the
   manifold disagree by ≥ `PipelineConfig.disagree_route_thresh` are flagged
   `topology_manifold_disagreement` and routed to the gated review (prioritised
   within the cap), not just the low-confidence tail — spending the scarce LLM
   budget where the two independent graph views conflict, which §3c showed is
   where the manifold's unique signal lives. Verified end-to-end on HuRI (the
   flagged pairs are topology-says-safe / manifold-says-risky). See
   `pipeline._disagreement_keys`, `tests/test_routing.py`.
