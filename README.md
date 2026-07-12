# negaverse

**negaverse builds better "these don't go together" examples for biology datasets — and shows its reasoning.**

Built for the Claude: Life Sciences Hackathon.

---

## The problem, in plain terms

To teach a computer which proteins **work together**, you show it examples of pairs that **do** (called *positives*) and pairs that **don't** (called *negatives*).

Positives are collected carefully by scientists. Negatives usually aren't — the common shortcut is to **pair proteins at random** and assume they don't interact. That's risky: some of those "random non-pairs" actually *do* interact, we just haven't tested them yet. Feeding those mistakes to a model teaches it wrong things.

**negaverse replaces random guessing with negatives that are:**
- **matched** to the real examples (so a model can't cheat with shortcuts),
- **confidence-scored** (how sure are we they don't interact?),
- **checked** against everything already known to interact,
- **explained** — every pair comes with the reason it was chosen.

> **Three words to know:** *positive* = a real interacting pair · *negative* = a non-interacting pair · *hard negative* = a non-pair that *looks* like it could interact, so it's a challenging, useful training example.

---

## How it works (the "hourglass")

Candidate pairs flow through three stages, cheap to expensive:

```
   many candidate pairs
        │
   1. QUICK REJECT   ── drop anything already known to interact
        │
   2. SCORE          ── cheap checks in parallel: network shape,
        │               biology rules (e.g. "these live in different
        │               parts of the cell, so they can't meet")
        │
   3. AI REVIEW      ── an LLM reads only the few most uncertain pairs
        │
   two clean sets out:  a fair BENCHMARK set + a challenging TRAINING set
```

Every pair that comes out carries its scores, flags, and a plain reason — nothing is a black box.

---

## What's built so far

| Piece | What it does | Status |
|---|---|---|
| **Pipeline** | The full hourglass: quick-reject → score → AI-review → two output sets | ✅ works end-to-end |
| **Network-shape check** | Judges a pair by how much it *looks like* a real interaction in the network | ✅ |
| **Embedding-manifold view** | A second, *independent* graph view: places each protein by who it interacts with, then flags a pair that looks like the crowd of real interactions (a likely hidden positive). Where this view and the network-shape view **disagree**, the pair is sent to AI review. | ✅ (opt-in) |
| **Biology rules** | Plain-English rules in a text file (`rules/*.yaml`) become checks with **no code** — e.g. co-localization ("different part of the cell ⇒ safe non-pair") | ✅ engine + co-localization live |
| **AI literature review** | An LLM reads the risky pairs (the ones that look like they might really interact) and returns a reasoned verdict; every run reports how many risky pairs it judged vs. left over, and `--judge-remaining` finishes the tail | ✅ (on by default; skipped without an API key) |
| **Known-interaction screening** | Removes any pair documented as interacting in outside databases (IntAct, BioGRID, …) | ✅ live — BioGRID + IntAct built by one script (~1.5M human pairs); vetoes 290 false-negatives on HuRI |
| **Benchmark** | Trains a model on our negatives vs. random ones and measures which is better | ✅ |
| **Dashboard** | A single web page (`out/report.html`) with plain-language charts, made after each run | ✅ |

**Adding a new biology rule is editing a text file — no programming.** See [`rules/AUTHORING.md`](rules/AUTHORING.md), or hand a paper to the `rule-from-literature` Claude skill and it writes the rule for you.

---

## Does it actually work? (the one-number proof)

We hid **1,000 real interactions** in the candidate pool and measured what fraction each strategy
**wrongly labels "negative"** — the exact mistake that poisons training data:

| strategy | HuRI | DRYAD |
|---|---:|---:|
| naive hard-negative mining (the common default) | **74.6 %** | **64.3 %** |
| **negaverse default (`stacked`)** | **0.6 %** | **0.0 %** |

Naive hard mining grabs ~3 of every 4 hidden positives; negaverse lets ~0 through. It also **beats
random downstream** and is the **cleanest** (zero leakage) — across 2 datasets and 2 learners.

→ **Full evidence, all flags, and every command on one page: [`docs/EVALUATION.md`](docs/EVALUATION.md).**
→ **Interactive showcase (open in a browser): [`docs/showcase.html`](docs/showcase.html)** — rotate the 3D maps, hover any risky pair for the LLM verdict. Self-contained; rebuild with `python3 scripts/build_showcase.py`.

---

## Quick start

```bash
pip install -e .            # core
pip install -e ".[llm,viz,bench]"   # + AI review, charts, benchmark
python -m negaverse.cli     # run it → writes out/negatives.csv + out/report.html
open out/report.html        # the dashboard
```

That's it for the demo. To use the AI review, copy `.env.example` to `.env` and add an API key (it auto-skips without one).

<details>
<summary><b>Getting the datasets</b> (not shipped — third-party; all live in gitignored <code>local-docs/</code>)</summary>

```bash
# gold reference non-interactions (Negatome)
mkdir -p local-docs/negatome2 && curl -k -o local-docs/negatome2/combined_stringent.txt \
  https://mips.helmholtz-muenchen.de/proj/ppi/negatome/combined_stringent.txt

# human interaction map for the benchmark (HuRI)
mkdir -p local-docs/huri && curl -kL -o local-docs/huri/HuRI.tsv \
  http://www.interactome-atlas.org/data/HuRI.tsv
```

The SARS-CoV-2 demo interactome (`Network_Table.xlsx`, Gordon et al. 2020) goes under `local-docs/`. Helper scripts build the extra biology annotations (all cached in gitignored `local-docs/`):
- `scripts/build_uniprot_ensembl_map.py` — map Negatome gold negatives into the benchmark's ID space
- `scripts/build_huri_annotations.py` — cell-compartment + chemistry annotations for the benchmark
- `scripts/compute_hydrophobicity.py` — protein chemistry from sequence
- `scripts/build_known_positive_sources.py` — build the BioGRID + IntAct known-interaction screens (downloads the raw human exports, emits UniProt- and Ensembl-space pair files for the SARS and HuRI graphs)
</details>

<details>
<summary><b>Common commands</b></summary>

```bash
python -m negaverse.cli --no-literature        # skip the AI review
python -m negaverse.cli --judge-remaining --out out   # judge any risky pairs a prior run left over
python -m negaverse.viz --dataset huri         # charts on the human dataset
python -m negaverse.viz --dataset huri --train-selection stacked   # how negatives are chosen (default: stacked)
python -m negaverse.bench --gold-test-neg --features spectral      # the fair benchmark
python scripts/bench_corrected.py --dataset huri --injection-test --inject-k 1000   # the proof above
python scripts/build_known_positive_sources.py # build the BioGRID + IntAct known-interaction screens
python scripts/validate_rules.py               # check the biology rules parse
```
</details>

---

## What you get out

```
out/negatives.csv / .jsonl   the negative pairs, with confidence + reasons
out/stats.json               a self-check report (no mistakes leaked in? well-matched?)
out/report.html              the dashboard — open this
out/literature_cards.json    the AI's verdicts on the uncertain pairs
```

The dashboard has: a **map** of pairs (real vs. random vs. our hard vs. risky), a **confidence chart**, the **reasons breakdown**, the **funnel**, and the **AI review** you can expand.

---

## Where we are & what's next

- ✅ **Phase 0 — foundation.** The hourglass, the plug-in filter system, the honest benchmark.
- ✅ **Phase 1 — proteins (mostly done).** Full pipeline on protein–protein data, biology rules firing, the dashboard, and the headline finding above.
- ▶️ **Finishing Phase 1 — beat random for real.** Stack more independent biology signals (protein chemistry, protein-language-model structure, function) onto the co-localization result, and drop in the outside interaction databases. *This is the current focus.*
- ⏭️ **Phase 2 — protein + drug.** Reuse the same machinery for protein–ligand (drug) interactions, to show it generalizes.
- ⏭️ **Phase 3 — polish.** Both modes end-to-end, dashboard, demo.

---

## Project layout

```
negaverse/     the engine (pipeline, filters, benchmark, charts)
rules/         biology rules as plain text (edit these — no code)
scripts/       data-prep + analysis helpers
docs/          design notes + the honest benchmark write-up
local-docs/    downloaded datasets (never committed)
tests/         checks that everything still works
```

More depth: **[`docs/EVALUATION.md`](docs/EVALUATION.md)** (jury one-pager: results + all flags) · [`docs/FILTER-EFFECTIVENESS.md`](docs/FILTER-EFFECTIVENESS.md) (full methodology) · [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) · [`docs/ADDING-A-FILTER.md`](docs/ADDING-A-FILTER.md) · [`rules/AUTHORING.md`](rules/AUTHORING.md)
