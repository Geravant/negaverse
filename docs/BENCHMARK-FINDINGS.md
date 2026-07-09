# Benchmark findings

Honest record of what the downstream-model benchmark (`python -m negaverse.bench`)
shows. The headline is a result that **does not support the naive thesis** under
rigorous evaluation — recorded here in full because the harness's job is to tell
us the truth, not to flatter the tool.

Setup: RandomForest link-predictor trained on HuRI positives + (random | negaverse)
negatives, evaluated on a held-out **fixed** negative test set. Features are built
on the train graph only (no test edge leaks into a feature). 3 seeds, 8000 positives.

---

## F-2 (headline) — The advantage is feature/selection circularity, not a real gain

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
