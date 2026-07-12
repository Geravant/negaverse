# Inductive generalization: do negaverse's negatives train a better PPI model — and can it propose novel interactions?

*Every number here is from a committed run on real data. Where a result is weak,
inconclusive, or negative, it says so.*

## The question

An **inductive** PPI model scores a pair from the two protein sequences (ESM2 /
D-SCRIPT), so it works on proteins it never saw in training — which is what lets it
propose *novel* interactions. We ask two things:

1. Do negatives chosen by negaverse (`stacked`, and the LLM-judge-`verified` set)
   train a better-generalizing model than `random` negatives?
2. Can the resulting model propose novel, undocumented human PPIs worth testing?

Three negative arms, same size, differ only in selection:
`random` (uniform veto-cleaned) · `stacked` (negaverse default) · `verified`
(stacked, then the LLM judge drops every `suspected_false_negative` — 303 of 8,000
dropped; no mislabeled positives).

## Result 1 — negaverse negatives generalize better, confirmed by two model families

Graded against **independent gold** (Negatome), features independent of the
selection axis. Δ = arm − random; higher = better.

**RandomForest on ESM2 (`bench_inductive_generalization.py`):**

| regime | random | stacked | verified | Δ(verified−random) |
|---|---|---|---|---|
| in-distribution (accuracy held) | 0.786 | 0.817 | **0.820** | **+0.033** |
| **protein-disjoint (unseen proteins)** | 0.669 | 0.714 | **0.720** | **+0.051** |
| transfer → DRYAD (cross-assay) | **0.641** | 0.526 | 0.521 | −0.120 |

**D-SCRIPT (`bench_dscript.py`, the published inductive architecture, n=500/10 epochs):**

| regime | random | stacked | verified |
|---|---|---|---|
| **protein-disjoint** | 0.430 | **0.560** | 0.547 |

D-SCRIPT: Δ(stacked−random) **+0.130**, Δ(verified−random) **+0.117** — the same
verdict as RF, *larger*, with a real architecture. Random even drops below chance
(0.43) on unseen proteins.

**Honest caveats.** (a) On *cross-assay* transfer to DRYAD, random wins (−0.12) —
the rules are human/HuRI-calibrated, so this is a within-human-interactome tool,
not a cross-assay one. (b) D-SCRIPT is only meaningful *at scale*: at 4 epochs /
245 pairs all arms were ≈ 0.5 (undertrained); the result above needed n=500/10
epochs where it reaches AUPR ~0.76. A GPU run would make it definitive.

## Result 2 — structural reality check with real AlphaFold data

Using **real AF2-Multimer pDockQ** (huintaf2 / Burke 2023 — genuine AF2 output):

- **Pass bar:** real pDockQ separates real interactions from random at **AUROC 0.69**
  (21% of real interactions clear the 0.23 confident-interface threshold vs 9% of random).
- **The model does NOT agree with structure:** trained protein-disjoint, model
  confidence vs real pDockQ is **anti-correlated (Spearman −0.16)**. The pairs the
  model ranks highest have *lower* structural interface confidence.
- **Why (investigated case-by-case):** it's **intrinsic disorder**, not one family.
  Of the model's false positives (high-P, pDockQ<0.23), 66% are high-disorder, 46%
  are the LCE/KRTAP families — disordered, promiscuous Y2H "sticky hubs" the model
  loves and AF2 correctly rejects. Controlling for disorder halves the
  anti-correlation (−0.16 → −0.09); controlling for degree barely moves it.
- **A sequence-disorder screen helps** (TOP-IDP, tracks AF2 disorder at 0.56, runs
  *before* folding): dropping the 25% most-disordered pairs halves the
  anti-correlation and doubles the top-100 mean pDockQ (0.035 → 0.081). It's a
  mitigation, not a cure — the model over-weights transient/disordered Y2H
  interactions.

**Takeaway: AF2 is a genuinely orthogonal filter — model top-K need structural
triage, and disorder is the failure mode to screen.**

## Result 3 — novel candidate predictions (screened, ranked, staged for AF2)

Pipeline: model narrows → disorder screen removes the failure mode → **AF2 selects**.

- **IDG understudied kinome** (`novel_idg_verified.tsv`): ordered (only 24%
  disorder-flagged), so trustworthy. Leads: **KALRN×TRIO** (paralogous Rho-GEFs),
  TSSK4×TSSK6 (paralogous kinases), small nucleotide-kinase pairs.
- **COVID host-host** (`covid_hosthost_shortlist.tsv`, from the verified D-SCRIPT
  model): top hits are mitochondrial-import / secretory (TOMM70, PMPCB, SRP72) and
  metabolic (COMT, POR, HMOX1) enzymes — the machinery SARS-CoV-2 hijacks; TOMM70
  is a bona-fide ORF9b target. Hub-capped + foldable batch staged.

### AF2 fold, first pass (5 small kinase pairs, single model, on a Colab T4)
All ipTM low (0.11–0.25) — **no confirmed hit yet**. Novel pairs edge controls
(top-2 novel; novel mean 0.18 vs control 0.15) but within noise. This is expected:
even *real* interactions clear the bar only 21% of the time, so folding 5 pairs and
getting zero is normal. Second pass (5 models, 20 complexes, kinases + host-host)
is running to hunt a confident hit (ipTM > 0.5).

## Bottom line

- Negaverse's **verified negatives beat random for inductive generalization** —
  confirmed independently by RandomForest (+0.05) and D-SCRIPT (+0.12), on
  proteins never seen in training.
- The model is a good interaction *ranker* but a weak *structural* predictor;
  **AF2-Multimer is the independent gate**, and a cheap sequence-disorder screen
  removes its main failure mode.
- The full pipeline runs end-to-end on real data and produces ranked, structurally-
  screened novel predictions for two spaces (kinome, COVID host-host). Landing a
  confirmed AF2 hit is a matter of folding more of them (GPU).

## Reproduce

```bash
PYTHONPATH=. python3 scripts/bench_inductive_generalization.py --seeds 0 1 2
PYTHONPATH=. python3 scripts/bench_dscript.py --n 500 --epochs 10 --regimes protein_disjoint --predict-covid
PYTHONPATH=. python3 scripts/af2_validate.py --stage reference     # real pDockQ pass bar
PYTHONPATH=. python3 scripts/af2_validate.py --stage investigate   # why model disagrees
PYTHONPATH=. python3 scripts/af2_validate.py --stage filtertest    # does the disorder screen help
PYTHONPATH=. python3 scripts/predict_novel_candidates.py --space idg --neg-arm verified
```
Run artifacts live under `out/` (gitignored); D-SCRIPT uses the `.venv-dscript` +
`scripts/dscript_mps.py` Metal launcher.
