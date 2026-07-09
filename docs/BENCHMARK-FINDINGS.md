# Benchmark findings

Honest record of what the downstream-model benchmark (`python -m negaverse.bench`)
has shown, including a result that *reverses* under a more rigorous setup. The
harness is the deliverable; these findings are why the harness is designed the
way it is.

Setup: RandomForest link-predictor trained on HuRI positives + (random | negaverse)
negatives, evaluated on a held-out, **fixed** negative test set. Features are built
on the train graph only (no test edge leaks into a feature). 3 seeds, 8000 positives.

---

## F-2 — The hard-negative "win" was feature/selection circularity

**Two feature families, opposite conclusions.**

| feature family | what it is | Δ negaverse − random (AUROC, 3 seeds) |
|---|---|---|
| `topological` | hand-crafted local indices: CN, Jaccard, Adamic-Adar, RA, pref-attach | **+0.070 … +0.082** |
| `spectral` | truncated-SVD node embeddings of the train adjacency, Hadamard per pair | **−0.024 … −0.051** |

Run it yourself: `python -m negaverse.bench --features topological` vs `--features spectral`.

**What happened.** negaverse selects hard negatives partly by the L3 / resource-allocation
topology indices. Those indices live in the *same space* as the `topological` benchmark
features, so training on negatives chosen to look positive-like in that space sharpens the
classifier's boundary **in that space** — an inflated, circular +0.07. Under `spectral`
features (structurally independent of the selection indices) the advantage not only
disappears, it goes slightly negative.

**Why negative, not just zero.** The hard negatives are topologically positive-like —
typically same-community pairs. Against an **easy** test negative set (uniform random pairs,
mostly cross-community), training on same-community negatives teaches the classifier that
"same community ⇏ interaction," which *hurts* it on the easy cross-community test negatives.
Hard training negatives only pay off when the **test distribution also contains hard cases**.

**Takeaways (both acted on):**
1. **Always report an independent feature family.** The `spectral` option exists precisely
   so a margin can't hide behind feature/selection overlap. A result that survives `spectral`
   is real; one that only shows under `topological` is suspect. (Done — this is F-2 itself.)
2. **The test negatives must be realistic, not easy.** An unbiased *random* test negative set
   understates the value of hard negatives, because it contains no hard cases to reward them.
   The meaningful benchmark uses **biologically-validated non-interactions** (Negatome golds)
   as the test negatives — hard, real negatives. That is the next benchmark upgrade.

**Status.** Not a bug — a correctly-measured null (indeed anti-) result under the rigorous
setup. It says: *on an easy random test set, negaverse's current hard negatives do not
generalize beyond the selection feature space.* Whether they help on a **hard/gold** test set
is the open question the Negatome-test-negative upgrade answers.

---

## F-1 — Simple-topology (Jaccard-only) selection: within noise

Before the L3 + config-model topology upgrade, the Jaccard-only selection gave a
negaverse-vs-random margin of −0.010 … +0.004 AUROC under `topological` features —
within run-to-run noise. This motivated the topology upgrade. Note F-2 now shows that
the post-upgrade +0.07 under `topological` features is itself an artifact; the honest
bottom line is F-2, not the raw +0.07.
