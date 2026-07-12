# negaverse — evaluation at a glance

*One page for the jury. What the system does, the proof it works, and every flag/command in one place.
Full methodology and per-rule breakdown live in [`FILTER-EFFECTIVENESS.md`](FILTER-EFFECTIVENESS.md).*

---

## The claim, in one sentence

Training a model to predict "do these two proteins interact?" needs **negative** examples.
The common shortcut — pair proteins at random — silently mislabels real-but-untested
interactions as "non-interacting," poisoning the training data. **negaverse selects negatives
that are screened, matched, confidence-scored, and explained — and we measured that it works.**

---

## 1. The headline proof — the hidden-positive injection backtest

We hid **K = 1,000 real interactions** inside the candidate pool (bypassing the known-interaction
screen, so they are genuinely "hidden") and measured what fraction each selection strategy
**wrongly picks as a negative**. This is the exact failure mode negaverse claims to prevent.

*Selection rate is model-independent (selection happens before any learner). AUROC = downstream damage.*

| strategy | HuRI: hidden+ selected | DRYAD: hidden+ selected | verdict |
|---|---:|---:|---|
| `topology_hard` — naive hard-negative mining | **74.6 %** | **64.3 %** | ☠️ poisons training data |
| `random_veto` — screened random baseline | 7.6 % | 0.8 % | leaks some |
| `topology_safe` | 0.3 % | 0.0 % | clean |
| **`stacked` — the shipped default** | **0.6 %** | **0.0 %** | ✅ catches ~100 % |

> **Naive hard-negative mining grabs ~3 of every 4 hidden positives. negaverse's default lets ~0 through.**
> Downstream, that contamination is catastrophic: `topology_hard` collapses to AUROC 0.33–0.39 (from ~0.86),
> and it is **worse under LightGBM** (boosting amplifies mislabeled-negative harm) — so the case for
> the clean default is stronger, not weaker, on a stronger learner.

This is the number for the slide.

---

## 2. It also beats random — and is the *purest* set

Downstream AUROC (held-out positives vs. **gold** non-interactions — Negatome for HuRI, DRYAD's own
labelled negatives), one frozen veto-cleaned pool shared by every strategy, mean over 3 seeds.

| strategy | HuRI·RF | HuRI·LGBM | DRYAD·RF | DRYAD·LGBM | hidden+ leaked |
|---|---|---|---|---|---:|
| `random_veto` (fair random baseline) | 0.872 | 0.854 | 0.642 | 0.626 | HuRI 0.3 |
| `topology_hard` (old default) | 0.749 | 0.677 | 0.483 | 0.529 | HuRI 5.3 |
| `topology_safe` | 0.872 | **0.882** | **0.672** | **0.651** | HuRI 0.3 |
| **`stacked` (default)** | **0.878** | 0.872 | 0.667 | 0.644 | **HuRI 0.0** |

**Δ vs. random (stacked − random_veto):** +0.006 / +0.018 / +0.025 / +0.018 — **positive in all four cells.**
**Purity:** `stacked` is the only **zero-leakage** strategy (random raw leaks ~26 real HuRI edges;
the veto removes ~99 % of that). The old "random beats us" headline was an artifact of a starved graph
and unequal screening — corrected here.

**Robustness:** the verdict holds on a **dense** graph (HuRI, 52k edges) and a **sparse** one
(DRYAD, avg degree 0.35), and under **two different learners** (RandomForest and LightGBM).

---

## 3. Which biology rules earn their place (leave-one-out)

Δ AUROC on the non-isolated stratum when each rule is removed (negative = the rule helps):

| rule | fires on | effect | verdict |
|---|---|---|---|
| `hydrophobicity_interface` | HuRI 34 %, DRYAD 14 % | −0.043 (DRYAD·RF) | **keeper** — the one rule that consistently helps |
| `colocalization_mismatch` | HuRI 32 % | ~neutral in ablation; direct calibration AUROC 0.88–0.91 on DRYAD | live, being re-measured |
| `evolutionary_coupling_absence` | 0 % | never fires | being removed |
| `string_low_confidence_non_interaction` | 0 % | no data at scale yet | dormant |

The point for the jury: **rules are auditable and individually testable** — we can and do show which
ones pull weight, and drop the ones that don't.

---

## 4. Every command — the catalogue

Three entry points. Everything is one CLI call.

### Generate negatives (the product)
```bash
python -m negaverse.cli                      # SARS-CoV-2 demo → out/negatives.csv + report.html
python -m negaverse.cli --n-train 500 --n-eval 500
python -m negaverse.cli --no-literature      # skip the LLM review
python -m negaverse.cli --judge-remaining    # judge the risky pairs a prior run left over
```

| flag | default | what it does |
|---|---|---|
| `--n-train` / `--n-eval` | 300 / 300 | how many training / benchmark negatives to emit |
| `--no-literature` | off | disable the Claude review even when a key is present |
| `--literature-k` | 40 | max risky pairs sent to the LLM per run (rest stay flagged) |
| `--judge-remaining` | off | resume a prior run and judge its still-unreviewed risky tail |
| `--votes` | 5 | best-of-N majority vote per pair in the LLM review |
| `--provider` / `--model` | auto | LLM backend and model override |
| `--seed`, `--out`, `--no-report` | 0, `out`, off | reproducibility / output dir / skip the dashboard |

### Dashboard on a real dataset
```bash
python -m negaverse.viz --dataset huri       # human PPI (or: sars, dryad)
python -m negaverse.viz --dataset huri --train-selection stacked
```
`--train-selection {stacked, safe, hard, mixture, psm}` — how emitted negatives are chosen (see §5).

### The honest benchmark
```bash
python -m negaverse.bench --gold-test-neg --features spectral        # quick, single-arm
python scripts/bench_corrected.py --dataset huri --max-positives 20000 --seeds 0 1 2 --models rf lgbm
python scripts/bench_corrected.py --dataset huri --injection-test --inject-k 1000   # the §1 proof
python scripts/bench_corrected.py --dataset huri --rule-ablation      # the §3 per-rule table
```

---

## 5. Selection modes — what `--train-selection` does

The pipeline scores every candidate on **safety** (is it really a non-interactor?) and **hardness**
(does it look like a real interaction?). These modes trade those off differently:

| mode | picks | use it when |
|---|---|---|
| **`stacked`** (default) | hard tail **re-ranked by confidence** — pairs every signal agrees are true negatives | **the recommended default** — hard *and* clean |
| `safe` | highest-confidence negatives across the whole pool | you want maximally clean, representative negatives |
| `hard` | topology-hardest tail only | ablation only — **poisons data** (§1), not recommended |
| `mixture` | tunable blend of representative / safe / hard | curriculum experiments |
| `psm` | degree-matched to a clean reference | propensity-matching experiments |

**Why `stacked` wins:** safety authorises the "negative" label; hardness makes it useful; the two are
kept separate. Hard-alone (§1) drags real interactions in; clean-alone loses the hard cases. `stacked`
keeps the pairs that are simultaneously hard *and* unanimously safe.

---

## 6. The hourglass — how a pair becomes a negative

```
   many candidate pairs
        │
   VETO    ── drop anything documented as interacting (BioGRID + IntAct, ~1.5M human pairs)
        │
   SCORE   ── cheap parallel checks: network shape + plain-English biology rules (rules/*.yaml)
        │
   GATE    ── Claude reads ONLY the few most uncertain pairs, returns a reasoned verdict
        │
   two clean sets: a fair BENCHMARK set + a hard TRAINING set
```

Every emitted pair carries its scores, flags, and a plain-English reason. Adding a new biology rule is
**editing a text file** (`rules/*.yaml`) — or hand a paper to the `rule-from-literature` Claude skill and
it writes the rule for you.

---

*Reproduce every number above: see the command blocks in §4 and the "Reproduce" sections of
[`FILTER-EFFECTIVENESS.md`](FILTER-EFFECTIVENESS.md). Numbers are means over seeds 0–2.*
