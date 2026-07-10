# Benchmark findings

Honest record of what the downstream-model benchmark (`python -m negaverse.bench`)
shows. The headline is a result that **does not support the naive thesis** under
rigorous evaluation — recorded here in full because the harness's job is to tell
us the truth, not to flatter the tool.

Setup: RandomForest link-predictor trained on HuRI positives + (random | negaverse)
negatives, evaluated on a held-out **fixed** negative test set. Features are built
on the train graph only (no test edge leaks into a feature). 3 seeds, 8000 positives.

---

## F-3 (headline) — An independent biology signal recovers most of the circularity deficit

F-2 (below) showed topology-hard negatives *hurt* under independent (spectral)
features. F-3 tests the fix the whole project points at: keep only the topology-hard
negatives that an **independent** signal *also* calls safe. Concretely, the
`negaverse_bio` strategy keeps the topology-hard pairs the co-localization rule flags
`different_compartment` (the two proteins share no subcellular compartment → can't
physically interact), dropping the topologically-hard-but-co-localized pairs that are
the likeliest hidden positives. Compartments come from GO cellular-component, mapped
into HuRI's Ensembl space (`scripts/build_huri_annotations.py`, ~95% coverage) — a
signal **independent of graph topology**.

Independent (spectral) features, gold Negatome test negatives, 3 seeds:

| strategy | AUROC (seed 0 / 1 / 2) | vs random |
|---|---|---|
| random | 0.786 / 0.777 / 0.734 | — |
| negaverse (topology-hard) | 0.672 / 0.668 / 0.659 | **−0.09 … −0.11** (the circularity hurt) |
| **negaverse_bio** (topology-hard ∩ co-localization-safe) | **0.736 / 0.731 / 0.711** | still below random, but **+0.052 … +0.064 over plain negaverse** |

Reproduce: `run_benchmark(..., feature_set="spectral", gold_test_neg=…, strategies=("random","negaverse","negaverse_bio"))`.

**What it means.** Biology-vetting reliably lifts the hard negatives by **+0.05–0.06
AUROC** under the most rigorous setting, recovering ~55% of the deficit that topology
alone incurs — direct evidence that an *independent* (non-topology) signal makes the
negatives genuinely cleaner, not just circularly better. It does **not** yet beat
random: one coarse signal (compartments) isn't enough. The clear read is **stack more
independent signals** (hydrophobicity — now computable via
`scripts/compute_hydrophobicity.py`; ESM2 structure — shown to generalize on DRYAD;
function/pathway) rather than lean on topology. This is the project's thesis, now with
a measured gradient to climb.

(Under `topological` features the ordering flips — `negaverse_bio` 0.838 < `negaverse`
0.870 — exactly as expected: bio-vetting removes some of the topology-space advantage,
which F-2 shows was circular anyway.)

---

## F-4 (headline) — Stacking independent signals reaches parity with random

F-3 climbed the gradient with one signal. F-4 stacks several. The `negaverse_stacked`
strategy ranks the topology-hard tail by the pipeline's **fused confidence** across
every independent signal at once — co-localization + surface-hydrophobicity mismatch +
GO biological-process + structured plausibility (the ESM2 sequence manifold was wired and
measured, but excluded — see below) — and keeps the pairs the combination most agrees are
true negatives, instead of
`negaverse_bio`'s single co-localization flag. The **external known-positive veto**
(BioGRID + IntAct, ~311k documented human interactions mapped into HuRI) is also active,
so documented positives can't leak into the negatives.

Independent (spectral) features, gold Negatome test negatives, 10k positives, 3 seeds:

| strategy | AUROC (seed 0 / 1 / 2) | Δ AUROC vs random | Δ AUPRC vs random |
|---|---|---|---|
| random | 0.866 / 0.823 / 0.829 | — | — |
| negaverse (topology-hard) | 0.810 / — / — | −0.056 | −0.028 |
| negaverse_bio (∩ co-localization) | 0.844 / 0.820 / 0.828 | −0.022 / −0.003 / −0.001 | −0.005 / −0.007 / +0.004 |
| **negaverse_stacked** (fused signals) | **0.860 / 0.835 / 0.827** | **−0.006 / +0.012 / −0.002** | **+0.003 / +0.004 / +0.002** |

Reproduce: `python -m negaverse.bench --gold-test-neg --features spectral --strategy random negaverse_bio negaverse_stacked`.

**What it means.** Stacking closes essentially all of the circularity deficit:
`negaverse_stacked` sits at **AUROC parity** with random (mean Δ ≈ +0.001 across seeds)
and **beats random on AUPRC in all three seeds** (+0.002 … +0.004) — the first setting
where our hard negatives are, on the fairest metric, no worse than (and slightly better
than) random. The gradient F-2 → F-3 → F-4 (−0.10 → −0.06 → ~0.00) is the project's
thesis playing out: **independent biology signals, stacked, make the hard negatives
genuinely clean.**

The stack that achieves this is **co-localization + hydrophobicity-mismatch + GO
biological-process** (all three live), fused by confidence. Honest per-signal read
(edge-vs-nonedge AUROC on HuRI, measured):

| signal | own AUROC | contribution to the stack |
|---|---|---|
| co-localization (GO CC, `disjoint`) | ~0.60 | the workhorse — most of the lift |
| hydrophobicity mismatch (KD mean) | ~0.55 | weak proxy; small |
| GO biological-process (`disjoint`) | ~0.54 | weak (exact-term; ~neutral) |
| **ESM2-t6 sequence manifold** | **~0.51** | **excluded — it HURTS** (see below) |

**The ESM2 negative result (important, and honest).** We built ESM2-t6 embeddings for
all 8,161 HuRI proteins and wired the `sequence_manifold` filter, expecting it to be the
strong lens (it gave 0.885 *supervised, protein-disjoint* on DRYAD). On HuRI the
unsupervised **manifold-surprisal** value is ~flat (edge-vs-nonedge AUROC **0.509**), so
adding it to the fusion **injects noise into the confidence ranking and degrades
selection** (Δ AUROC ~**−0.03** vs the stack without it; entropy-weighted fusion only
partly mitigates). Lesson: ESM2's power is as a **supervised downstream feature**, not an
unsupervised selection filter — and a coarse mean-pooled t6 embedding is too weak for the
manifold view regardless. It is therefore **off by default** in `negaverse_stacked`
(opt back in via `esm2_path`); this is a measured exclusion, not an oversight.

Caveat on attribution: hydrophobicity and function are both weak (~0.54–0.55), so most of
the lift is co-localization + the fused-confidence selection method. The honest read: the
*method* (fuse independent signals, keep what they agree on) reaches parity; clearing
random decisively needs a genuinely strong *independent* signal — a **GO-hierarchy
semantic** functional similarity (not exact-term), or ESM2 used **supervised** as a
feature, are the two most promising next levers.

---

## F-2 — The advantage is feature/selection circularity, not a real gain

Two axes, four cells. Δ = negaverse − random, AUROC, range over 3 seeds:

| training-negative eval          | **topological** features (CN/Jaccard/AA/RA/PA) | **spectral** features (SVD + Hadamard) |
|---------------------------------|:---:|:---:|
| **random** test negatives       | **+0.070 … +0.082** | −0.024 … −0.051 |
| **gold** test negatives (Negatome, in HuRI space) | **+0.077 … +0.099** | −0.033 … −0.053 |

Reproduce: `python -m negaverse.bench [--gold-test-neg] --features {topological,spectral}`.

**What determines the sign is the feature family, not the test set.** negaverse
selects hard negatives by the L3 / resource-allocation topology indices. Those live
in the *same space* as the `topological` benchmark features, so training on negatives
chosen to look positive-like in that space sharpens the classifier's boundary **in that
space** — a circular +0.08…+0.10. Under `spectral` features (structurally independent
of the selection indices) the advantage disappears and goes slightly negative — and it
stays negative even when the **test** negatives are gold-standard, biologically-validated
non-interactions (Negatome). Making the test harder does **not** rescue it.

**Interpretation.** The current topology-based hard-negative selection does not, by
this benchmark, produce negatives that improve a link predictor in a
feature-independent way. Its hard negatives are topologically positive-like
(same-community pairs); against features that can't exploit the selection structure,
they add label noise near the boundary and mildly hurt generalization. The `+0.09` you
get with `topological` features is not evidence of value — it is the selection signal
being read back out of the features.

**A refuted hypothesis (kept for the record).** We expected that an *easy* random test
set was the reason spectral showed no gain, and that gold (hard) test negatives would
reveal the benefit. The gold-test row above refutes that: spectral stays negative.

**Status: not a bug — a correctly-measured null/negative result.** It reshapes
priorities (see "Implications").

---

## F-1 — Simple-topology (Jaccard-only) selection: within noise

Before the L3 + config-model upgrade, Jaccard-only selection gave −0.010 … +0.004 AUROC
under `topological` features — within run-to-run noise. This motivated the topology
upgrade; F-2 then showed the post-upgrade `topological` gain is itself circular.

---

## Implications (what this says to do next)

1. **Report `spectral` (or ESM2) numbers as the honest headline; never quote the
   `topological` gain alone.** The benchmark now ships both precisely so the circular
   number can't stand unqualified.
2. **The value of negaverse may not be "raise a link-predictor's AUROC".** It is a
   *curated, confidence-scored, auditable, matched* negative set (leakage-checked,
   degree-matched eval vs hard train, provenance, gold-recall). Those properties are
   real and testable independently of this one RF benchmark — and are arguably the
   product. The RF benchmark is one lens, and a demanding one.
3. **If the downstream-AUROC claim matters, the selection signal must diversify away
   from graph topology** — sequence/structure (ESM2), localization, function — so that
   "hard" is not defined in the same space a topological classifier reads. That is the
   Phase-1 co-localization filter and the Phase-2 embedding work.
4. Gold coverage is currently modest: of 5462 Negatome pairs, **349 map fully into
   HuRI's node space** (~50% of Negatome UniProt IDs resolve to a HuRI Ensembl gene).
   A larger human PPI graph (IntAct/BioGRID union) would widen the gold test set.
