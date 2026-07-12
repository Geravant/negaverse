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
python -m negaverse.bench --gold-test-neg --features spectral   # the fair benchmark
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

More depth: [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) · [`docs/BENCHMARK-FINDINGS.md`](docs/BENCHMARK-FINDINGS.md) · [`docs/ADDING-A-FILTER.md`](docs/ADDING-A-FILTER.md) · [`rules/AUTHORING.md`](rules/AUTHORING.md)

## Cheat-sheet: filters, modes, and how to run

### Three things you can run

| Command | What it does |
|---|---|
| `python -m negaverse.cli` | **Generate** a negative dataset for the built-in SARS-CoV-2 graph (writes results + an HTML report). |
| `python -m negaverse.viz --dataset huri` | **Visualise** a run as an interactive report (`out/report.html`). Datasets: `sars`, `huri`, `dryad`. |
| `python scripts/bench_corrected.py` | **Benchmark** how good the chosen negatives are vs. random (the study in `docs/FILTER-EFFECTIVENESS.md`). |

### The filters (what screens each candidate pair)

Every candidate pair passes through these in order. The first two can *reject* a pair; the rest give it a score.

| Filter | Plain-words job |
|---|---|
| `known_positive_veto` | Throws out any pair already recorded as interacting (in the graph or in BioGRID/IntAct). **Always on.** |
| `rule_veto` | Throws out pairs a hard biology rule forbids. |
| `structured` | Down-weights "sticky" hub proteins that interact with everything. |
| `topology` | Judges how *edge-like* a pair looks from the network shape (near an interaction = risky negative). |
| `rules` | Biology rules (co-localisation, hydrophobicity, …) that mark a pair as a safer negative. |
| `manifold`, `sequence_manifold` | Optional extra viewpoints (network surprise; protein-sequence similarity). |
| `literature` | Optional LLM check on the uncertain pairs; drops likely hidden interactions. Needs an API key. |

### Selection modes (how the final training negatives are picked)

Set with `--train-selection` (viz) or `PipelineConfig(train_selection=...)`. Evidence: `docs/FILTER-EFFECTIVENESS.md`.

| Mode | Picks… | Verdict |
|---|---|---|
| `stacked` | hardest pairs, then re-ranked by biology confidence | **default — best & cleanest** |
| `safe` | the pairs we're most confident are non-interactions | ties the default |
| `hard` | only the most edge-like pairs | not recommended — grabs hidden interactions |
| `mixture` | a blend of representative + safe + hard | no better than `stacked` |
| `psm` | pairs matched to the positives' profile, from the clean pool | offered for study; below `stacked` |

### Benchmark flags (`scripts/bench_corrected.py`)

| Flag | Meaning |
|---|---|
| `--dataset huri\|dryad` | which graph |
| `--max-positives N` | cap positives (`0` = use all) |
| `--seeds 0 1 2` | repeat runs for averages |
| `--models rf lgbm` | downstream learner(s) to test with |
| `--rule-ablation` | drop each biology rule one at a time to see its worth |
| `--mixture` / `--psm` / `--eval-match` | compare those selection strategies |
| `--injection-test` `--inject-k K` | plant K known interactions and check which strategy wrongly keeps them |

### Generation flags (`python -m negaverse.cli`)

| Flag | Meaning |
|---|---|
| `--n-train` / `--n-eval` | how many training / evaluation negatives to emit |
| `--no-literature` | skip the LLM check (offline, no API key) |
| `--provider` / `--model` / `--votes` | LLM provider, model, and best-of-N voting |
| `--literature-k` | how many uncertain pairs the LLM reviews |
| `--no-report` | skip the HTML report |
| `--seed` | reproducibility |
