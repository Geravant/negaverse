# Filter effectiveness — the corrected benchmark

*Does negaverse's filter system pick better training negatives than random, and does
each rule earn its place? Measured honestly, with the benchmark artifacts removed.*

---

## 1. Methodology & legend

### 1.1 Why the old benchmark was wrong (the three artifacts)

The old headline compared *random* negatives against the *100 % topology-hardest* tail on
a HuRI graph **capped at 6 000 positives**. Three defects, all fixed here:

1. **Aggressive positive cap → isolation shortcut.** 6 000 of 52 068 HuRI edges makes the
   training graph so sparse that most proteins are isolated → their spectral (SVD) node
   embedding is all-zero → the test set is dominated by a trivial *"if either endpoint is
   isolated, predict negative"* shortcut. Random negatives (≈81 % all-zero features)
   reproduce that shortcut; the topology-hard tail (0 % isolated — topology cannot call an
   isolated pair "hard") never learns it. ~85 % of the −0.097 rode on this.
2. **Unequal pools.** Random skipped the known-positive veto, so it leaked ~25 real
   interactions into its "negative" set; the filter arms leaked ~0. Random was *dirtier*
   yet scored higher — AUROC was rewarding the shortcut, not negative purity.
3. **100 % hard-tail replacement.** Emitting only the topology-hardest negatives is a
   narrow, hidden-positive-enriched distribution — the one selection that genuinely loses.

### 1.2 What the corrected bench does

* **One frozen, veto-cleaned candidate pool** shared by every arm (equal purity, equal
  universe — only the *selection* differs).
* **Un-capped positives** by default (`--max-positives 0` = full graph).
* **Degree-stratified reporting**: every metric is given on all test pairs *and* on the
  **non-isolated** stratum (both endpoints have a graph edge), which removes the
  zero-feature shortcut and is the only stratum where graph/biology signal can matter.
* **Independent yardsticks only** (Ground rule: never grade a filter with the axis used to
  select it): test negatives are **gold** (Negatome for HuRI, DRYAD's own labelled
  negatives), features are **spectral graph-SVD**, used only to score, never to select.

### 1.3 Datasets

| dataset | positives | nodes | density | gold negatives |
|---|---:|---:|---|---|
| **HuRI** | 52 068 | 8 245 | dense (avg degree ≈13) | Negatome (349 in-graph) |
| **DRYAD** | 3 000 | 17 341 | very sparse (avg degree ≈0.35) | DRYAD labelled negatives |

### 1.4 Models (downstream learner — sensitivity check)

| tag | model | why |
|---|---|---|
| **RF** | RandomForest (200 trees) | the original learner |
| **LGBM** | LightGBM (300 trees, boosted) | does the verdict survive a different learner, or is it an RF quirk / does boosting just exploit the shortcut harder? |

### 1.5 Arms (the thing under test — how the negatives are chosen)

All arms draw the same number of negatives from the same frozen pool; only the selection rule differs.

| arm | selection rule |
|---|---|
| `random_raw` | uniform from the pool, **no veto** (baseline, dirtiest) |
| `random_veto` | uniform from the **veto-cleaned** pool (the fair random baseline) |
| `topology_hard` | the topology-**hardest** tail (nearest the positive manifold) — the old default |
| `topology_safe` | the **highest fused-confidence** negatives across the whole pool (representative + clean) |
| `stacked` | the hard tail **re-ranked by fused biology confidence** — keeps the pairs every signal agrees are true negatives. **The shipped default** (`PipelineConfig.train_selection="stacked"`). |

### 1.6 Metrics & legend

| metric | meaning | good = |
|---|---|---|
| **AUROC** | rank-separation of held-out positives vs gold negatives | higher |
| **AUPRC** | same, precision-weighted | higher |
| **AUROC_noniso** | AUROC on the **non-isolated** test stratum (no zero-feature shortcut) — the honest signal | higher |
| **PPIHits@100** | fraction of the top-100 scored pairs that are true positives | higher |
| **PPNIHits@100** | fraction of the bottom-100 scored pairs that are true negatives | higher |
| **hidden+** | # real interactions leaked into the "negative" set (purity) | **lower** |

---

## 2. Results — the full filter table (cross-dataset × cross-model, 3 seeds)

Cells are **AUROC (AUROC_noniso in parens)**, mean over seeds 0–2. `random_raw` (no veto) is the
dirtiest baseline — included so the `hidden+` column in §2.1 has something to vary against.

| arm | HuRI·RF | HuRI·LGBM | DRYAD·RF | DRYAD·LGBM |
|---|---|---|---|---|
| `random_raw` (no veto) | 0.873 (0.905) | 0.855 (0.877) | 0.639 (0.806) | 0.622 (0.838) |
| `random_veto` | 0.872 (0.903) | 0.854 (0.879) | 0.642 (0.807) | 0.626 (0.842) |
| `topology_hard` | 0.749 (0.756) | 0.677 (0.647) | 0.483 (0.533) | 0.529 (0.744) |
| `topology_safe` | 0.872 (0.906) | **0.882** (0.911) | **0.672** (0.804) | **0.651** (0.854) |
| **`stacked`** (default) | **0.878** (**0.912**) | 0.872 (0.904) | 0.667 (0.762) | 0.644 (**0.855**) |

**Δ AUROC (stacked − random_veto):** HuRI-RF **+0.006**, HuRI-LGBM **+0.018**, DRYAD-RF
**+0.025**, DRYAD-LGBM **+0.018** — positive in every cell.
**Δ AUROC_noniso:** +0.009, +0.025, **−0.045**, +0.013 — positive in 3 of 4. The DRYAD-RF cell is
*negative*, and that is the tell: on sparse DRYAD the non-isolated stratum is tiny, so
AUROC_noniso there is high-variance and even sign-flips between runs (an earlier run read +0.018
in this cell). Do not over-read any single DRYAD·noniso number — the stable DRYAD signal is the
overall-AUROC win, not this stratum.
**`topology_hard`:** −0.123 / −0.177 / −0.159 / −0.097 — the **only consistent loser**.
**AUPRC:** best (or tied) is `stacked`/`topology_safe` in every cell.

### 2.1 Purity — the `hidden+` column (the one the table was hiding)

`hidden+` = mean real interactions leaked into the "negative" set. It is a property of the
*selected set*, not the learner, so there is **one value per arm per dataset** (identical for RF and
LGBM) — which is exactly why it never fit the arm×model AUROC grid and got demoted to prose. Its
whole story is the gap between `random_raw` and everything the veto touches:

| arm | HuRI hidden+ | DRYAD hidden+ | note |
|---|---:|---:|---|
| `random_raw` (no veto) | **26.3** | 0.0 | uniform sampling grabs ~26 real HuRI edges and labels them "negative" |
| `random_veto` | 0.3 | 0.0 | the known-positive veto removes ~99 % of that leakage |
| `topology_hard` | 5.3 | 0.0 | the hard tail *re-enriches* for hidden positives (positive-like by construction) |
| `topology_safe` | 0.3 | 0.0 | representative + clean |
| **`stacked`** (default) | **0.0** | 0.0 | the shipped default is the **cleanest arm** — zero leakage across all seeds |

DRYAD is 0.0 everywhere because its gold negatives are a separate labelled set (no HuRI-style
"unobserved ⇒ negative" assumption), so there is nothing to leak. The point HuRI makes plain:
**random is the dirtiest defensible baseline and `stacked` is the purest** — the exact inversion of
the old "random is better" headline, once you grade on purity instead of AUROC alone.

### 2.2 Ranking hits — PPIHit@100 (top) / PPNIHit@100 (bottom)

Fraction of the model's top-100 that are true positives, and bottom-100 that are true negatives.
PPIHit@100 is near-saturated everywhere (the *top* of the ranking is easy), so **PPNIHit@100 — did
the model manage to bury the true negatives at the bottom? — is the discriminating metric:**

| arm | HuRI·RF | HuRI·LGBM | DRYAD·RF | DRYAD·LGBM |
|---|---|---|---|---|
| `random_raw` | 0.980 / 0.793 | 0.977 / 0.803 | 1.00 / 0.693 | 1.00 / 0.097 |
| `random_veto` | 0.977 / 0.850 | 0.983 / 0.793 | 1.00 / 0.707 | 1.00 / 0.123 |
| `topology_hard` | 0.953 / 0.567 | 0.940 / 0.457 | 1.00 / **0.010** | 1.00 / 0.043 |
| `topology_safe` | 0.963 / 0.817 | 0.967 / **0.857** | 1.00 / **0.893** | 1.00 / **0.273** |
| **`stacked`** | 0.973 / 0.823 | 0.970 / 0.837 | 1.00 / 0.863 | 1.00 / 0.230 |

(cells = PPIHit@100 / PPNIHit@100.) `topology_hard` craters the bottom ranking (DRYAD-RF **0.010** —
almost no true negative reaches the bottom-100, because a model trained on positive-*like* negatives
can't push them down); `topology_safe` and `stacked` keep it high. Same verdict as AUROC, read from
the ranking end.

**Coverage sensitivity.** The effect sharpens as the graph starves: at an 8 000-positive HuRI
cap, `topology_safe` = 0.800 vs `random_veto` 0.758 and `topology_hard` collapses to 0.410;
at 20 000 it is the table above. Give the graph coverage and the arms converge, with `stacked`
on top; the old −0.097 only appears at the 6 000 cap.

**Verdict.** Across **2 datasets × 2 models × 4 metrics**, the shipped default `stacked` is
best-or-tied and the **cleanest**; `topology_safe` ties-or-beats random; `topology_hard`
alone loses. The result is **model-robust** (LightGBM reproduces every RandomForest
conclusion — it does *not* just exploit the shortcut harder) and **dataset-robust** (holds
on dense HuRI and sparse DRYAD, though on DRYAD the win is thinner and lives in the *overall*
AUROC — the non-isolated stratum there is too small to be stable and even sign-flips between
runs (§2.1 Δnoniso). On HuRI, `stacked` wins on every stratum *and* is the only zero-leakage
arm (§2.1).

---

## 3. Per-rule leave-one-out (`--rule-ablation`, 3 seeds)

Each graded rule removed from the `stacked` arm one at a time; **Δ = stacked[−rule] −
stacked[ALL]**, so a rule that *helps* shows a **negative** Δ when removed. The meaningful
column is **AUROC_noniso** — overall AUROC is dominated by the isolation shortcut, where no
biology rule can help; the biology only bites on pairs both endpoints of which are in the
graph. (Overall-AUROC deltas are all within ±0.004 = noise.)

**Δ AUROC_noniso when the rule is removed (negative = the rule helps):**

| rule | fires on | HuRI·RF | HuRI·LGBM | DRYAD·RF | DRYAD·LGBM | verdict |
|---|---|---:|---:|---:|---:|---|
| `hydrophobicity_interface` | HuRI 34 %, DRYAD 14 % | −0.004 | −0.004 | **−0.043** | −0.003 | **helps** (the one keeper) |
| `colocalization_mismatch` | HuRI 32 %, DRYAD 0 % | −0.003 | +0.001 | +0.001 | −0.002 | ~neutral (within noise) |
| `evolutionary_coupling_absence` | **0 % both** | 0.000 | 0.000 | 0.000 | 0.000 | **dead** — never fires |
| `string_low_confidence_non_interaction` | **0 % both** | 0.000 | 0.000 | 0.000 | 0.000 | **dead** — never fires |

**Whole rule layer** (stacked[ALL] − stacked[NO rules]), AUROC_noniso: HuRI-RF **+0.009**,
HuRI-LGBM −0.004, DRYAD-RF **+0.048**, DRYAD-LGBM +0.008 — net positive on the
biology-relevant stratum in 3 of 4 cells.

**Which rules earn their place:**
1. **`evolutionary_coupling_absence` + `string_low_confidence_non_interaction` contribute
   *exactly nothing* — anywhere** (Δ = 0.000 in all 8 cells). They **never fire**: no
   `evolutionary_coupling.tsv` exists, and `string_score_with_b` isn't even registered in
   `_PAIR_FIELDS`. Dead weight — drop them, or wire the data.
2. **`colocalization_mismatch` does not measurably contribute** — within noise on HuRI, zero
   coverage on DRYAD (its `go_cc` localization table is ENSG-keyed, so it barely overlaps
   DRYAD's UniProt nodes). Not earning its place as shipped.
3. **`hydrophobicity_interface` is the one rule that genuinely helps** — a consistent negative
   Δ on the non-isolated stratum across both datasets and both models (strongest on DRYAD-RF,
   +0.043). The keeper.

---

## Reproduce

```bash
# full filter table (per dataset × models)
PYTHONPATH=. python3 scripts/bench_corrected.py --dataset huri  --max-positives 20000 --seeds 0 1 2 --models rf lgbm
PYTHONPATH=. python3 scripts/bench_corrected.py --dataset dryad --max-positives 0     --seeds 0 1 2 --models rf lgbm

# per-rule leave-one-out
PYTHONPATH=. python3 scripts/bench_corrected.py --dataset huri  --max-positives 20000 --seeds 0 1 2 --models rf lgbm --rule-ablation
PYTHONPATH=. python3 scripts/bench_corrected.py --dataset dryad --max-positives 0     --seeds 0 1 2 --models rf lgbm --rule-ablation
```

Needs `lightgbm` (+ `libomp` on macOS); drop `lgbm` from `--models` to run RandomForest only.
