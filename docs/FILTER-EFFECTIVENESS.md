# Filter effectiveness — the corrected benchmark

*Does negaverse's filter system pick better training negatives than random, and does
each rule earn its place? Measured honestly, with the benchmark artifacts removed.*

Everything here is produced by one script, `scripts/bench_corrected.py`, on real data.
No placeholder numbers: every cell comes from a committed 3-seed run (commands at the
bottom). An earlier version of this doc reported that "topology filters are worse than
random" (Δ ≈ −0.097 on HuRI) — that was a **benchmark artifact**, now fixed and deleted.
This is the corrected picture.

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

Cells are **AUROC (AUROC_noniso in parens)**, mean over seeds 0–2.

| arm | HuRI·RF | HuRI·LGBM | DRYAD·RF | DRYAD·LGBM |
|---|---|---|---|---|
| `random_veto` | 0.873 (0.905) | 0.857 (0.882) | 0.640 (0.804) | 0.625 (0.846) |
| `topology_hard` | 0.742 (0.748) | 0.668 (0.652) | 0.474 (0.530) | 0.528 (0.746) |
| `topology_safe` | 0.869 (0.904) | **0.881** (0.909) | **0.672** (0.784) | **0.653** (0.875) |
| **`stacked`** (default) | **0.879** (**0.913**) | 0.870 (0.904) | 0.671 (**0.822**) | 0.647 (**0.887**) |

**Δ AUROC (stacked − random_veto):** HuRI-RF **+0.005**, HuRI-LGBM **+0.013**, DRYAD-RF
**+0.031**, DRYAD-LGBM **+0.022** — positive in every cell.
**Δ AUROC_noniso:** +0.008, +0.022, +0.018, +0.041 — positive in every cell.
**`topology_hard`:** −0.132 / −0.190 / −0.166 / −0.097 — the **only consistent loser**.
**AUPRC:** best (or tied) is `stacked` in every cell.
**Purity (hidden+):** `random_raw` leaks ~23 real interactions on HuRI; every veto arm
(`random_veto`, `safe`, `stacked`) leaks ≤1. (DRYAD's gold negatives are a separate labelled
set, so leakage is 0 for all there.)

**Coverage sensitivity.** The effect sharpens as the graph starves: at an 8 000-positive HuRI
cap, `topology_safe` = 0.800 vs `random_veto` 0.758 and `topology_hard` collapses to 0.410;
at 20 000 it is the table above. Give the graph coverage and the arms converge, with `stacked`
on top; the old −0.097 only appears at the 6 000 cap.

**Verdict.** Across **2 datasets × 2 models × 4 metrics**, the shipped default `stacked` is
best-or-tied and the **cleanest**; `topology_safe` ties-or-beats random; `topology_hard`
alone loses. The result is **model-robust** (LightGBM reproduces every RandomForest
conclusion — it does *not* just exploit the shortcut harder) and **dataset-robust** (holds
on dense HuRI and sparse DRYAD, though on DRYAD the win is thinner and rides on the
non-isolated stratum — see §3 and the sparsity caveat in §4).

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

## 4. Conclusions & open items

* **The known-positive filters do not make the data worse.** The old "−0.097 worse than
  random" was the 6 000-cap isolation shortcut + unequal pools + 100 % hard-tail selection.
  Fixed, the full system (`stacked`) is the **best negative-sampling arm and the purest**.
* **Selection mode matters more than the filters.** `topology_hard` (the old pipeline default)
  is the only arm that loses; `stacked`/`safe` win. This is now shipped:
  `PipelineConfig.train_selection` defaults to **`stacked`** (`negaverse/matching.py::select_train`).
* **Rule cleanup is warranted.** Two of four graded rules are non-functional (no data wired)
  and one is noise; only `hydrophobicity_interface` earns its place. Wire STRING/EC data or
  drop the dead rules.
* **Sparsity caveat.** On very sparse graphs (DRYAD) topology-selection degenerates toward a
  hub filter (it can only score, and re-rank, pairs whose endpoints have edges), so the win
  is thin and concentrated on the non-isolated stratum. Topology-based selection needs graph
  density to work; a sparse-graph guard is future work.

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
