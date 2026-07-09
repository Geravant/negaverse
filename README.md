# negaverse

negaverse helps generate better negative examples for interaction datasets.

It was built for the Claude: Life Sciences Hackathon.

> **Initial prototype.** This is an early, end-to-end skeleton. The three scoring
> methods described below are working but intentionally simple stand-ins — the
> full-strength streams (learned graph embeddings, richer literature retrieval)
> will be wired in shortly. Treat the current scores as a demonstration of the
> pipeline, not the final scoring quality.

Many biology datasets tell us which things do interact — for example, which proteins bind to each other. But machine learning models also need examples of things that do **not** interact.

The problem is that true "non-interactions" are rarely collected carefully. A common shortcut is to randomly pair proteins and pretend they do not interact. That can create noisy training data, because some of those random pairs may actually interact but have not been tested yet.

negaverse tries to make this process safer and more transparent. Instead of creating random negative pairs, it generates negative examples that are:

* matched to the positive examples,
* scored by confidence,
* checked against known interactions,
* and saved with a clear explanation of how each example was created.

Full design notes: [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md).
Pipeline diagram: [`docs/negaverse-architecture-diagram.html`](docs/negaverse-architecture-diagram.html).

---

## What the prototype does

The current prototype works with a SARS-CoV-2 host–pathogen interaction dataset. It uses:

* known viral ↔ human protein interactions as positive examples,
* host ↔ host protein interactions to understand the graph structure,
* and Negatome 2.0 as a reference set of known human protein non-interactions.

The main goal is to generate better candidate negatives for the viral ↔ host interaction space.

---

## Providing input data

The datasets are **not** shipped with this repo — they are third-party and live in `local-docs/`, which is gitignored. Download them yourself and place them at the paths below (or point the loaders at your own files).

**1. Gold negatives — Negatome 2.0** (reference set of known human non-interactions):

```bash
mkdir -p local-docs/negatome2
curl -k -o local-docs/negatome2/combined_stringent.txt \
  https://mips.helmholtz-muenchen.de/proj/ppi/negatome/combined_stringent.txt
```

**2. Positives — SARS-CoV-2 interactome** (the demo dataset). Obtain the `Network_Table.xlsx` from the Gordon et al. host–pathogen interaction map (Nature, 2020; Krogan lab supplementary data) and place it at:

```
local-docs/xlsxUploads_.../sars-cov2-spreadsheets/Network_Table.xlsx
```

Expected columns: `Bait`, `PreyUniprotAcc`, `PreyGeneName`, `is_HumanPPI`.

**3. Human PPI positives — HuRI** (for the downstream benchmark, `python -m negaverse.bench`):

```bash
mkdir -p local-docs/huri
curl -kL -o local-docs/huri/HuRI.tsv http://www.interactome-atlas.org/data/HuRI.tsv
```

A homogeneous human protein–protein interaction map (~52k edges, Ensembl-keyed). Used to test whether negaverse's hard negatives train a better link-prediction model than random negatives (AUROC/AUPRC).

**Custom paths.** Loaders accept a `path=` argument, so you can store the files anywhere:

```python
from negaverse.io import load_sars_cov2_graph, load_negatome_pairs
graph = load_sars_cov2_graph(path="my/Network_Table.xlsx")
gold  = load_negatome_pairs(path="my/negatome.txt")
```

Never commit downloaded data — keep it under `local-docs/` (already gitignored).

---

## How it works

The pipeline follows this flow:

`generate candidate pairs → remove known positives → score candidates → combine scores → create train/eval sets → export results`

**1. Generate candidate pairs.** The system first creates a broad pool of possible protein pairs from the interaction graph. At this stage these are only candidates — some may become useful negatives, others may be removed or marked as risky later.

**2. Remove known positives.** Any pair already listed as a known interaction is removed, so the final negative dataset can't accidentally include confirmed positives.

**3. Score each candidate.** Each remaining candidate is scored from several angles: Does this pair look risky? Does the graph suggest it may be a hidden positive? Is there literature evidence that makes it suspicious? Is it a good match for the positives? A score doesn't claim "this pair definitely does not interact" — it means "based on the available checks, this looks like a safer or riskier negative example."

**4. Combine the scores.** The different scores are combined into one final confidence score. Some methods can also flag a candidate as suspicious or ask the system to avoid using it.

**5. Create two output sets** (from the same run, never mixed):

* **Evaluation set** — matched to the positives, so models can't win with shortcuts like "high-degree proteins are usually positive." Use this for benchmarking.
* **Training set** — harder negatives, closer to the decision boundary. Use this to help models learn more useful signals.

**6. Export results.** The output files include the generated negative pairs, confidence scores, hardness scores, score details, provenance, and warning flags such as `suspected_false_negative`. Written to:

```
out/negatives.csv
out/negatives.jsonl
out/stats.json
```

The literature scorer runs by default and also writes `out/literature_cards.json`. It is automatically skipped if no API key is available (see Setup), so the core pipeline always runs.

---

## Scoring methods

| Method | What it does | Status |
|---|---|---|
| Structured scoring | Removes known positives and applies simple safety rules. | Working (simple) |
| Graph scoring | Checks whether a candidate pair looks similar to known positive interactions in the graph. | Working (simple) |
| Literature scoring | Uses an LLM to review the most uncertain pairs and return a structured risk judgment. | Working, on by default (skipped without a key) |


---

## Evaluation

The pipeline produces a small validation report in `out/stats.json`. It checks things like:

* whether any known positive accidentally appears as a negative,
* whether the evaluation negatives are well matched to the positives,
* whether the training negatives are harder than random negatives,
* and whether reference negatives are handled correctly.

On the demo dataset, the pipeline confirms that no known positives are emitted as negatives, and that the evaluation set is much better matched than random negatives.

---

## Setup

Requires Python 3.10 or newer.

```bash
git clone <repo>
cd negaverse
pip install -e .
```

This installs the core dependencies: numpy, pandas, networkx, scipy, openpyxl.

### Enable LLM literature scoring

Literature scoring runs by default, but needs the LLM extra and an API key. With
neither installed nor configured, the pipeline just skips that step.

```bash
pip install -e ".[llm]"     # adds anthropic + httpx
cp .env.example .env
```

Add one or both API keys to `.env`:

```
ANTHROPIC_API_KEY=sk-ant-...
OPENROUTER_API_KEY=sk-or-...
```

The `.env` file is ignored by git and loaded automatically. Whichever key is
present is used (Anthropic preferred); no flags needed.

---

## Running the pipeline

```bash
# full pipeline — literature scoring runs automatically when a key is present
python -m negaverse.cli

# force a specific backend / model
python -m negaverse.cli --provider openrouter --model anthropic/claude-opus-4-8

# skip the LLM pass entirely
python -m negaverse.cli --no-literature
```

Outputs are saved in `out/`.

### Using it from Python

```python
from negaverse import run_pipeline, PipelineConfig
from negaverse.io import load_sars_cov2_graph

graph = load_sars_cov2_graph()
result = run_pipeline(
    graph,
    PipelineConfig(n_eval=300, n_train=300, match_on_type="viral"),
)
for record in result.records[:3]:
    print(record.u, record.v, record.mode, record.confidence, record.flags)
```

### Running tests

```bash
python -m tests.test_smoke   # pipeline: no leakage, disjoint split, matched eval, harder train
python -m tests.test_llm     # LLM controller: config, key loading, JSON parsing (no model call)
```

---

## Project layout

```
negaverse/
  graph.py          graph data structures
  candidates.py     candidate pair generation
  streams/          scoring methods
  fusion.py         combines scores
  matching.py       creates train and evaluation splits
  pipeline.py       main pipeline
  eval.py           validation metrics
  schema.py         output record format
  io/               data loaders
  llm/              optional LLM literature scoring
  cli.py            command-line runner
docs/               design notes and diagram
local-docs/         local datasets, ignored by git
```

---

## Current status

The project runs end-to-end on the SARS-CoV-2 demo dataset. Working pieces: candidate generation, known-positive filtering, structured scoring, graph scoring, LLM literature scoring (on by default, auto-skipped without a key), train/evaluation split, provenance, and validation reporting.

The scoring methods are simple first versions — proper streams will be wired in shortly. Next steps:

* add a human ↔ human positive interaction dataset,
* make the Negatome benchmark more directly comparable,
* improve graph embeddings,
* and make the scoring system easier to configure.
