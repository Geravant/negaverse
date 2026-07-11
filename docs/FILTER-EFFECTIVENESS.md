# Filter-effectiveness testing — the percolation scenario

*How we prove each filter earns its place, alone and in combination — without
fooling ourselves.*

This doc is a **test protocol**, not a results dump. It says exactly which run
produces which number, which yardstick that number is read against, and why the
yardstick is independent of the filter under test. Result cells start **empty**
and are filled only by a real run (§8). The few numbers already in the ledger are
ones we have genuinely measured, tagged `[measured]`; everything else is `[TODO]`.

---

## 0. Ground rules (non-negotiable)

Every test in this doc obeys four rules. A result that breaks any of them is not
a weak result — it is a **void** result, deleted, not reported.

1. **Independent yardstick.** Never grade a filter with the same axis used to
   *select* the negatives. Selecting on topology and scoring on topology
   manufactures a gain out of nothing ("picked with a ruler, graded with the same
   ruler" — `docs/BENCHMARK-FINDINGS.md` F-1..F-3). Each test below names its
   yardstick and states why it is orthogonal to the filter under test. The two
   yardsticks we trust:
   - **Gold negatives** — Negatome (PPI) / DRYAD's own labelled negatives. Curated
     non-interactors nobody's filter had a hand in choosing.
   - **A held-out independent feature axis** — spectral graph SVD or ESM2 sequence
     embeddings, used *only* to score, never to select, in that test.
2. **No placeholder statistics.** Every reported number comes from a committed
   script run on real data. No "illustrative", no "roughly", no hand-filled cells.
   If a run hasn't happened, the cell says `[TODO]` and nothing else.
3. **No mislabeled positives.** A hard negative that survives every silent filter
   is *labelled* negative by "we haven't seen it in the positive list" — which is
   exactly how a hidden positive slips in. Before any effectiveness number is
   read, the emitted hard set is verified by the GATED judge and every
   `suspected_false_negative` is **dropped**, not merely flagged. An effectiveness
   score computed over a set that still contains suspected positives is void.
4. **Protein-disjoint splits.** The pairs used to build/calibrate a filter and the
   pairs used to score it share no protein. No per-row leakage, no
   train/eval overlap. (`degree_matched_eval` + `hard_train` already exclude eval
   keys; gold-negative benches split on proteins, not pairs.)

---

## 1. Two questions, never conflated

"Is this filter effective?" is two different questions with two different tests.
Keeping them apart is the whole game.

| # | Question | Test type | Metric | Higher = |
|---|----------|-----------|--------|----------|
| **Q1** | Does the filter's signal *rank* real non-interactors above interactions? | **Separation** — rank pairs by the raw sub-score, measure against gold. | Separation AUROC (pos vs gold-neg) | better signal |
| **Q2** | Do negatives *chosen* with this filter train a better downstream link-predictor than random negatives? | **Value** — train an RF on positives + these negatives, test on held-out positives + **gold** negatives. | Downstream model AUROC; report `Δ = negaverse − random` | more useful negatives |

Q1 is cheap and answers "is there signal here at all". **Q2 is the only one that
decides whether the filter belongs in production.** A filter can win Q1 and *lose*
Q2 — that is not a paradox, it is the central finding of this project (§7).

Separation AUROC is an honest proxy; `negaverse.bench` (Q2) is the final word
(`scripts/bench_rules.py:24`).

---

## 2. The percolation model — where each filter sits

Candidates fall through an hourglass. A test targets one band of it.

```
  admissible non-edges                          scripts / knobs
        │
   ┌────▼─────┐  VETO (funnel)      any veto=True drops the pair
   │  VETO    │  known_positive_veto, rule_veto
   └────┬─────┘
   ┌────▼─────┐  GRADED (parallel)  every filter scores; fuse → confidence
   │  GRADED  │  structured, topology*, rules, [manifold], [sequence_manifold]
   └────┬─────┘  *provides_hardness → drives the hard/easy split
        │        matching: degree_matched_eval (eval) + hard_train (train)
   ┌────▼─────┐  GATED (funnel)     runs ONLY on the contested tail
   │  GATED   │  literature (LLM judge) — verifies & DROPS suspected positives
   └────┬─────┘
        ▼
   emitted eval + train sets (full provenance)
```

**The filter roster under test** (name · stage · default · what it claims):

| Filter | Stage | Default | Independent of? | Claim |
|--------|-------|---------|-----------------|-------|
| `known_positive_veto` | VETO | on | — | Never emit a known positive (IntAct/BioGRID union). |
| `rule_veto` | VETO | on | — | Drop pairs a hard YAML rule forbids. |
| `structured` | GRADED | on | topology, sequence | Promiscuous hubs are risky negatives; implausible pairs are safe. |
| `topology` | GRADED | on (hardness driver) | sequence, gold | L3+RA graph link-prediction risk = the hard/easy axis. |
| `rules` | GRADED | on | topology, sequence | Biology signals (co-localization, hydrophobicity, coupling…) mark safer negatives. |
| `manifold` | GRADED | **opt-in** | gold (not topology — ~0.64 corr) | Global-graph surprisal; disagreement partner for topology. |
| `sequence_manifold` | GRADED | **opt-in** | topology & spectral (~0.2 corr) | ESM2 sequence surprisal — the genuinely independent axis. |
| `literature` | GATED | on but `enabled=False` | all graph/seq axes | LLM verifies the contested hard tail; drops hidden positives. |

Testing "in combination" is a config change, never a code change: pass an explicit
`filters=[...]` list, flip `enabled=True`, or supply embeddings. That is what makes
the ablation matrix (§5) cheap to run exhaustively.

---

## 3. Per-filter scenarios (each filter alone)

Each scenario isolates **one** filter, scores against a yardstick that filter did
not touch, and states a pass bar. Setup column = the `filters=[...]` list (VETO
scaffolding `known_positive_veto + structured` is kept so the pipeline runs, but is
never the thing being scored).

### 3.1 `topology`
- **Isolate:** `[known_positive_veto, structured, topology]`, `manifold`/`sequence_manifold` off.
- **Q1 yardstick — gold.** Rank candidates by topology risk; measure separation of
  HuRI positives vs **Negatome** gold negatives. *Independent:* Negatome is
  UniProt-curated non-interactors, not derived from HuRI topology.
  Script: `scripts/eval_manifold_flags.py` (topology comparison arm) / `bench_rules.py` topology column.
- **Q2 yardstick — gold + spectral features.** Train RF on HuRI positives +
  topology-hard negatives; test on held-out positives + Negatome, spectral
  features. `scripts/bench_negaverse_vs_random.py`. **Report Δ vs random.**
- **Pass bar:** Q1 separation AUROC > 0.5 clearly; **Q2 Δ ≥ 0** (this is the bar it
  currently *fails* alone — see §7).

### 3.2 `structured`
- **Isolate:** `[known_positive_veto, structured]`.
- **Q1 yardstick — gold.** Does promiscuity-prior rank gold negs above positives?
- **Q2:** same value harness. Weak-but-safe prior — we expect small, non-negative Δ.
- **Pass bar:** does not *hurt* Q2 (Δ not significantly < 0); positive Q1 signal is a bonus.

### 3.3 `rules` (each YAML rule, separately and pooled)
- **Isolate:** run with a single rule enabled, then all rules pooled.
  Rules live in `rules/ppi.yaml` (co-localization, hydrophobicity, evolutionary-coupling, STRING-low-confidence).
- **Q1 yardstick — gold + vs-hard.** `scripts/bench_rules.py` already does this per
  rule: separation-vs-random, separation-vs-hard (Negatome), coverage, and
  **leave-one-out** `Δhard_if_removed`. *Independent:* biology rules use GO / sequence
  / STRING, none of which is the graph-topology selection axis.
- **Pass bar per rule:** non-trivial coverage **and** a leave-one-out contribution
  that doesn't vanish. A rule with 0 marginal contribution is cut, not kept.

### 3.4 `manifold` (spectral, opt-in)
- **Isolate:** `[known_positive_veto, structured, manifold]`.
- **Q1 yardstick — gold, leakage-free.** `scripts/eval_manifold_flags.py`: fit the
  positive manifold on a **train split**, score a disjoint held-out split, measure
  whether the `suspected_false_negative` flag lifts hidden-positive vs clean
  separation. *Caveat logged:* manifold ~0.64 correlated with topology, so its
  standalone value is small — its real job is **disagreement** (§6), not solo use.
- **Pass bar:** flag lift > 0 on the held-out split; contamination-removal improves
  a downstream clean metric.

### 3.5 `sequence_manifold` (ESM2, opt-in) — the independent axis
- **Isolate:** `[known_positive_veto, structured, sequence_manifold]` with ESM2 `.npz`.
- **Q1 yardstick — gold, and cross-axis.** `scripts/eval_esm2_manifold.py`:
  separation of DRYAD pos vs neg on the sequence axis, **plus** its correlation to
  the graph axes (~0.2 — genuinely independent). This independence is the point:
  it is the yardstick the *other* filters get graded against.
- **Q2 — the rescue test.** `scripts/bench_features_ablation.py`: does ESM2 as a
  *feature* rescue topology-hard negatives? (measured: features lift 0.73→0.91, but
  only *halve* topology-hard harm — §7.)
- **Pass bar:** Q1 separation > graph axes on sequence-decidable pairs; low corr to
  topology confirmed.

### 3.6 `literature` (GATED judge)
- **Isolate:** enable on a fixed, frozen hard set; `enabled=True`, Haiku default.
- **Yardstick — gold overlap.** Of the pairs the judge calls
  `suspected_false_negative`, what fraction are in a gold **positive** list it never
  saw (IntAct/BioGRID held out)? That is its precision at catching hidden positives.
- **Pass bar:** judge-dropped pairs are enriched for held-out positives vs the
  retained set; `risky_coverage` stat = 1.0 (every routed pair actually got a verdict).
- **Cost note:** feature-hashed persistent cache (`literature.py`) makes re-runs
  near-free; Haiku keeps the cold run cheap.

### 3.7 VETO filters (`known_positive_veto`, `rule_veto`)
- These are **correctness gates, not effectiveness signals** — they don't get an
  AUROC, they get a **leakage audit**: with the veto ON, count emitted pairs that
  appear in the union known-positive set. **Must be exactly 0.** With it OFF, the
  same count is the contamination it prevents (report it — that's its value).

---

## 4. The single-filter ledger (Q1 separation + Q2 value)

One row per filter, both questions. `[measured]` = real run committed; `[TODO]` = not yet run.

| Filter | Q1 sep-AUROC vs gold | Q2 Δ vs random (downstream) | Yardstick | Verdict |
|--------|---------------------:|----------------------------:|-----------|---------|
| `structured` | `[TODO]` | `[TODO]` | Negatome / spectral | — |
| `topology` | `[TODO]` | **−0.097** `[measured, HuRI]` | Negatome / spectral | fails Q2 alone |
| `rules` (pooled) | `[TODO]` | `[TODO]` | Negatome | — |
| `rules` (per-rule LOO) | see `out/rules_bench_*.json` | n/a | Negatome vs-hard | — |
| `manifold` | `[TODO]` (held-out flag lift) | `[TODO]` | gold, leakage-free | disagreement-only |
| `sequence_manifold` | `[TODO]` | `[TODO]` | DRYAD gold, cross-axis | independent axis ✓ |
| `literature` | precision@drop `[TODO]` | n/a (verifier) | held-out positives | — |

Cross-cutting facts already measured, used as fixed reference points (not filter
rows): ESM2 **as features** lifts downstream 0.73→0.91 `[measured, DRYAD]`; real
AF2-Multimer pDockQ separates HuRI-vs-random at **0.70** `[measured, huintaf2]`;
sequence↔graph axis correlation ≈ 0.2, manifold↔topology ≈ 0.64 `[measured]`.

---

## 5. Combination scenarios (percolation in stacks)

Filters interact: an independent second axis can rescue a first that fails alone.
We test combinations three ways.

### 5.1 Add-one-in (does a filter help on top of the base stack?)
Base = default PPI stack `known_positive_veto + rule_veto + structured + topology + rules`.
For each opt-in filter X ∈ {`manifold`, `sequence_manifold`, `literature`}:
run **base** vs **base + X** through the Q2 value harness; report `Δ = (base+X) − base`.

| Added filter | Q2 AUROC base | Q2 AUROC base+X | Δ | Verdict |
|--------------|--------------:|----------------:|--:|---------|
| `manifold` | `[TODO]` | `[TODO]` | `[TODO]` | — |
| `sequence_manifold` | `[TODO]` | `[TODO]` | `[TODO]` | — |
| `literature` (drop-FN) | `[TODO]` | `[TODO]` | `[TODO]` | — |

### 5.2 Leave-one-out (is each default filter pulling its weight?)
Full stack minus one filter at a time, Q2 harness. A filter whose removal doesn't
lower Q2 AUROC is a candidate for demotion to opt-in. (`bench_rules.py` already does
this at the *rule* granularity via `Δhard_if_removed`; this extends it to filters.)

| Removed filter | Q2 AUROC (stack − filter) | Δ vs full | Keep? |
|----------------|--------------------------:|----------:|-------|
| `structured` | `[TODO]` | `[TODO]` | — |
| `topology` | `[TODO]` | `[TODO]` | — |
| `rules` | `[TODO]` | `[TODO]` | — |

### 5.3 Independent-axis stacking (the key hypothesis)
The reason topology fails Q2 alone is that topology-hard negatives are
positive-*like* on the topology axis — some are hidden positives. Stacking a
**genuinely independent** axis (`sequence_manifold` / ESM2, corr ≈ 0.2) should
recover the deficit. Test the ladder:

| Stack | Q2 Δ vs random | Note |
|-------|---------------:|------|
| `topology` only | **−0.097** `[measured]` | fails |
| `topology + manifold` | `[TODO]` | manifold not independent (0.64) — expect little |
| `topology + sequence_manifold` | `[TODO]` | independent axis — expect recovery toward 0 |
| `topology + sequence_manifold + literature`(drop-FN) | `[TODO]` | verified hard — the real test |

**Hypothesis to confirm or kill:** only the *verified* stack (bottom row) flips Δ
to ≥ 0. If even that fails, the honest conclusion is "for this dataset, random
negatives are the right default" — and we report that, per Ground Rule 2.

---

## 6. Disagreement & routing scenarios (the manifold's real job)

`manifold` correlates 0.64 with `topology`, so it adds little as a *score*. Its
value is where the two independent graph views **disagree** — those pairs are the
interesting ones. Two tests:

1. **Routing works** (already covered by `tests/test_routing.py`): pairs where
   `|topology − manifold| ≥ disagree_route_thresh` are flagged
   `topology_manifold_disagreement` and routed to GATED even at unremarkable
   confidence. Assert every flagged pair reaches the judge.
2. **Disagreement is informative:** of the routed disagreement pairs, is the
   judge's `suspected_false_negative` rate higher than on a random contested
   sample? If yes, disagreement earns its routing cost. `[TODO]` — needs a run with
   `enabled=True`, `disagree_pairs=[("topology","manifold")]`.

---

## 7. What we already know (so tests aim at open questions, not settled ones)

These are `[measured]`, load-bearing, and shape every scenario above:

- **Topology-hard negatives do not beat random downstream** (HuRI Q2 Δ = −0.097).
  Selecting negatives to look positive-like adds label noise.
- **ESM2 as model features** lifts downstream AUROC 0.73→0.91 — but only **halves**
  the harm from topology-hard negatives; it does **not** flip Δ positive. Strong
  evidence the hard set is contaminated with hidden (mislabeled) positives.
- **Real AF2-Multimer pDockQ** separates HuRI-vs-random at 0.70 — the structural
  signal is real, which is *why* topology-hard pairs are so easily positives.
- **Axis independence:** sequence↔graph ≈ 0.2 (independent, useful for stacking &
  grading); manifold↔topology ≈ 0.64 (redundant as score, useful as disagreement).

The one open, decision-grade question this protocol exists to answer:

> **Does verifying the emitted hard set (LLM judge, drop suspected FNs) make a
> negaverse hard-negative stack beat random downstream (Q2 Δ ≥ 0)?**

§5.3 bottom row is that test. It is not yet run because the drop-on-`suspected_false_negative`
policy in the GATED merge is still pending (flag today, drop tomorrow).

---

## 8. How to run each scenario

All commands read real data from `local-docs/` and write JSON to `out/`. Nothing
below fabricates a number.

```bash
# Q1 separation, per biology rule (coverage + leave-one-out contribution)
python -m scripts.bench_rules --dataset huri      # + Negatome hard negatives
python -m scripts.bench_rules --dataset dryad      # + ESM2 co-evolution column
python -m scripts.bench_rules --dataset huintaf2 --external af2   # real pDockQ

# Q2 value — the decision: negaverse vs random, on gold negatives
python -m scripts.bench_negaverse_vs_random --dataset huri   # Negatome gold
python -m scripts.bench_negaverse_vs_random --dataset dryad  # DRYAD gold

# Independent-axis rescue (ESM2 features vs graph features × random vs hard)
python -m scripts.bench_features_ablation --dataset dryad

# Manifold flag, leakage-free (fit on train split, score held-out)
python -m scripts.eval_manifold_flags

# Sequence vs spectral vs topology axes + fusion + cross-correlation
python -m scripts.eval_esm2_manifold

# Routing + verification (needs a key; Haiku default, cache makes re-runs cheap)
#   run the pipeline with enabled=True, disagree_pairs=[("topology","manifold")]
```

**Run order:** §3 per-filter Q1 → §4 per-filter Q2 → §5 combinations → §6 routing.
Fill a ledger cell **only** from a committed run. A blank cell is more honest than
a guessed one.

---

## 9. Measured run — 2026-07-11 (with Lucy's structure-aware hydrophobicity live)

First full pass of the §8 suite, on real data. Lucy's calibrated
`hydrophobicity_interface` now runs on structure-aware `surface_hydrophobicity`
(DSSP + AlphaFold, `scripts/build_huri_surface_hydrophobicity.py`; 5,426/8,163
HuRI genes structure-based). All numbers below are `[measured]`.

**Q2 — the decision (`bench_negaverse_vs_random`, spectral features, gold negatives, 3 seeds):**

| dataset | random | negaverse (topology-hard) | Δ | **stacked** | Δ |
|---|---:|---:|---:|---:|---:|
| HuRI  | 0.786 | 0.689 | **−0.097** | **0.780** | **−0.006** |

→ Topology-hard alone **fails** Q2 (−0.097); the full independent-signal stack
(co-localization + structure-aware hydrophobicity, fused-confidence selection) nearly
**erases the deficit** (−0.006, seed 0) but does not yet *beat* random.

**CORRECTION.** An earlier pass reported stacked ≈ −0.000, but that was an **artifact**:
`run_benchmark` had no `negaverse_stacked` branch, so the strategy fell through to
*random* negatives. Fixed in `benchmark.py` (real `_negaverse_stacked_negatives` +
`_negaverse_verified_negatives`); the −0.006 above is the corrected, real number.

**§7 verified-stack test — RESOLVED (2026-07-11).** The earlier attempt was blocked
because the judge saw only opaque **ENSG gene IDs**. Fixed: `build_ensg_symbol_map.py`
now maps **8,163/8,245 HuRI nodes (99%)** to gene symbols (was 629/7.6%), wired into the
judge via `LiteratureFilter(names=...)`. Three measurements now settle it:

1. **The judge works — held-out detection (`eval_judge_holdout_recall.py`, k=150, seed 0).**
   Labeled task, independent of the pipeline: POSITIVES = BioGRID∪IntAct interactions that
   are HuRI non-edges (real hidden positives), NEGATIVES = random HuRI non-edges absent from
   every DB. With gene symbols the judge flags **45.3%** of hidden positives vs only **16.0%**
   of true negatives — **separation +29.3%** (and clears negatives as `safe_negative` 78% vs
   49% on positives). With symbols the judge is a genuine hidden-positive detector.

2. **Downstream AUROC can't see it, and that's expected.** Corrected `negaverse_verified`
   (drop-and-backfill from the *stacked* set — the earlier version was confounded, training on
   the least-confident tail vs stacked's most-confident head) drops ≈stacked. Dropping ~2% of
   4,800 negatives is below the AUROC noise floor (seed variance ≈0.05); a 200-tree RF is
   robust to a couple-percent relabel. Downstream AUROC is the wrong instrument for a small
   relabel — hence measurement #1.

3. **The pool-precision oracle is confounded by the veto (`eval_judge_flag_precision.py`).**
   `KnownPositiveVeto` already loads BioGRID+IntAct (`rules/sources.yaml`) and strips every such
   pair *before* the judge runs: **0%** of the hard pool is a BioGRID/IntAct interaction, vs
   **1%** of random pairs. So any hidden positive left in the pool is one **no wired DB records**
   — unvalidatable against those same DBs. The judge's role downstream of the veto is catching
   *database-unrecorded* hidden positives; database-known ones are already gone.

Net: the judge adds real signal (#1); the residual topology-hard harm that survives the veto is
either genuine hardness or DB-unrecorded hidden positives — indistinguishable with current oracles.

**§5.3 ESM2-rescue (`bench_features_ablation`, DRYAD):**

| features | random | topo-hard | Δ |
|---|---:|---:|---:|
| graph (spectral) | 0.731 | 0.353 | −0.378 |
| esm2 (sequence) | 0.913 | 0.778 | −0.135 |

→ ESM2 features lift the baseline (0.73→0.91) and **halve** topo-hard harm, but don't
flip Δ positive — the hard set is contaminated with hidden positives (confirms §7).

**Axis independence (`eval_esm2_manifold`):** esm2↔spectral 0.19, esm2↔topology 0.16
(independent); spectral↔topology 0.70 (redundant). Fused esm2+spectral+topology = 0.881.

**Per-rule Q1 (`bench_rules`, coverage · sep-vs-Negatome-hard · leave-one-out Δhard):**
- **HuRI:** `colocalization_mismatch` 32% · Δhard **−0.092** (biggest contributor);
  `hydrophobicity_interface` (structure-aware) 34% · Δhard −0.032; pooled 0.287 vs-hard.
- **DRYAD:** `coev:esm2_cosine` 34% · **0.773** vs-hard (the strong signal here);
  `hydrophobicity_interface` 14% · 0.522.

**Per-rule Q2 — downstream, whole-stack (`bench_rule_ablation_downstream`, HuRI, spectral
features, Negatome gold, 3 seeds).** The "final word" `bench_rules` defers to, run per rule:
each graded rule is left out of the full stack (`veto+structured+topology+rules`) and the
change in downstream link-predictor AUROC is measured. `Δ = AUROC(ALL) − AUROC(ALL−R)`.

| config | mean AUROC | Δ |
|---|---:|---:|
| ALL graded rules | 0.7644 | — |
| NONE (structured+topology only) | 0.7572 | layer worth **+0.0072** |
| ALL − `colocalization_mismatch` | 0.7663 | **−0.0019** |
| ALL − `hydrophobicity_interface` | 0.7653 | **−0.0008** |
| ALL − `evolutionary_coupling_absence` | 0.7644 | 0.0000 (never fires on HuRI) |
| ALL − `string_low_confidence_non_interaction` | 0.7644 | 0.0000 (never fires on HuRI) |

→ **On HuRI, no graded rule earns its keep at Q2.** Every leave-one-out Δ (±0.002) is an order
of magnitude *below* seed variance (per-seed ALL: 0.785 / 0.774 / 0.734, spread ≈0.05); the
layer's +0.0072 is itself within noise. Sharpest point: `colocalization_mismatch` is the **Q1
strongest** rule (Δhard −0.092) yet downstream **neutral-to-negative** (−0.0019) — separating
from Negatome ≠ making better training negatives, the project's core thesis, now at stack level.
Two rules have **zero HuRI coverage** (no ESM2-coupling / STRING data wired), so HuRI can't
evaluate them — they need a dataset where they fire (DRYAD coev Q1 = 0.773). The base
`structured+topology` already captures HuRI's usable signal.

**Manifold flag, leakage-free (`eval_manifold_flags`):** on pairs topology calls SAFE,
the manifold flag finds hidden positives at AUROC **0.68**; on a 5%-contaminated eval set
it removed 210/460 injected positives (460→250) at a cost of 352 clean flags.

**Real AF2-Multimer pDockQ (`convert_huintaf2`, published huintaf2 scores, no folding):**
HuRI-vs-random **0.704**, Hu.MAP-vs-random 0.603 — the structural interface signal is real,
which is *why* topology-hard pairs are so often hidden positives.

Artifacts: `out/{rules_bench_huri,rules_bench_dryad,rules_bench_huintaf2,esm2_manifold_eval,manifold_flags_eval,af2_scores}.*` (gitignored).
