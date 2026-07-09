Read both — the project docx (the full 8-step approach + visualizations + benchmarking) and the flowchart PDF. A few things jump out that reframe your questions, so let me ground the answers in what the doc actually commits to versus what's already built.

Quick status anchor — what negaverse **already has** (PPI): bounded candidate generation, known-positive exclusion, a topology/embedding stream (simple Jaccard link-prediction), a gated LLM literature stream (Anthropic/OpenRouter), score fusion, the degree-matched eval + hard train split, and validation metrics (leakage, degree-match KS, hardness). What's **missing** relative to the doc: entity resolution, real structured-DB screening (IntAct/BioGRID/ChEMBL/BindingDB), biological-context + chemical filters, PLI as a modality, the downstream-model benchmark, and all the visualizations.

---

## 1. Scope — is the full stack too ambitious? (Yes.)

The doc's target is 8 filtering steps × 4 relationship types + visualizations + benchmarking. In 5 days that's not realistic — not because any one piece is hard, but because each new *modality* (PLI) and each new *data source* (IntAct, ChEMBL, GO, localization) is a data-wrangling tax that eats days, not hours. Your instinct to cut to **2 relationship types + 4 filters** is the right shape. Here's how I'd triage:

| Component | Verdict | Reasoning |
|---|---|---|
| **PPI vertical** | **Keep — it's your "it works + benchmarks" story** | Mostly built. Negatome (gold) + Krogan COVID (positives) are on disk. Lowest risk, highest polish potential. |
| **PLI vertical** | **Keep but time-box hard** | This is genuinely new code (bipartite graph, RDKit chemistry, different data sources). It's where the *innovation* is (chemical feasibility), but also the biggest sink. De-risk by using **DUD-E / LIT-PCBA** where gold negatives already exist — which lets you validate quickly and lean on RDKit Tanimoto for the chemical stream. But be realistic: PLI is where you'll spend most new effort. |
| **Structured DB screening** | **Keep (funnel it — see Q3)** | Cheap, non-negotiable, prevents leakage. But scope it: query 1–2 sources (IntAct/BioGRID for PPI; ChEMBL for PLI), not the full integrated union. |
| **Topology filter** | **Keep, but simplify — don't build full TPPNI** | The *cheap core* of TPPNI is a few lines: configuration-model expected-edge probability `≈ (k_u·k_v)/2m`, plus **L3** (length-3 paths via sparse A³ — a better PPI predictor than common neighbors, Kovács 2019). Skip the full contrastive-L3 framework; the config-model rank + L3 gets you 80% of the signal. |
| **Chemical / physicochemical feasibility** | **Keep for PLI, drop for PPI** | For PLI this is the differentiator: RDKit ECFP Tanimoto + basic property compatibility is fast and high-signal. For PPI the doc itself hedges ("when annotations are available") — interface/hydrophobicity rules need structures you mostly won't have. Drop it for PPI. |
| **Abstract-level literature** | **Keep — already built** | Your gated LLM controller is exactly this. One refinement: the doc wants *abstract-level screening* (search for "non-binding / not detected / inactive" evidence), which is cheaper and more grounded than the free-form reasoning we do now. Worth adding a retrieval step (even PubMed abstract search) so the LLM cites real sentences, not its own priors. |
| **Evolutionary + subcellular context (step 5)** | **Mostly cut — but keep ONE piece** | Full evo/organism plumbing is expensive for modest payoff. **Exception: subcellular co-localization mismatch is one of the cheapest, highest-signal PPI filters** (two proteins that never share a compartment can't physically bind), and you *have* localization data in the SARS tables (`table_s3`). I'd fold a minimal co-localization check into the PPI "biological context" rather than dropping it entirely. |
| **Full-text review (step 8)** | **Stub only** | Keep the "Review needed?" gate in the flow as a *flag* (`suspected_false_negative` already does this), but don't build real full-text retrieval. It's the classic "escalation path we'd add later." |
| **Protein-RNA / Protein-DNA** | **Cut** | Two more modalities you can't afford. Mention them as "the plugin interface supports these" for the vision. |
| **Self-training loop** | **Cut** | From the old ARCHITECTURE; pure scope risk. |

Net: **PPI (topology + co-localization + literature + benchmark) done well, PLI (topology + chemistry + literature, validated on DUD-E) as the second modality.** Four filters, two verticals, matching your instinct.

---

## 2. Timeline — realistic 5-day SE approach

The critical reframing: **the doc is half software and half demo.** The visualizations and the downstream-model benchmark aren't nice-to-haves — per the doc they *are* the deliverable and the hypothesis test ("do hard negatives beat random?"). A working pipeline with no benchmark and no viz loses a hackathon; a slightly thinner pipeline with a killer AUROC-comparison chart and a UMAP overlap plot wins it.

So prioritize by demo-narrative, not by pipeline completeness:

- **Day 1 — Lock the PPI benchmark (the money shot).** Train a simple RF/GNN on (random negatives) vs (negaverse negatives) using Krogan COVID or a human PPI set + Negatome, and produce the AUROC/AUPRC comparison. This proves the central hypothesis and de-risks everything. It also forces the human↔human positive set we flagged earlier (needed for Negatome to be in-space).
- **Day 2 — Visualizations for PPI.** Degree-distribution overlap, shortest-path KDE (random vs hard vs positive), UMAP of ESM2 embeddings, and the Sankey "% removed per filter." Most are ~30 lines of matplotlib/plotly each and reuse `out/negatives.jsonl` + `stats.json`. High ROI, low risk.
- **Day 3 — PLI modality, minimal.** Bipartite loader, RDKit ECFP + Tanimoto chemical stream, DUD-E as gold. Reuse the existing stream/fusion/split architecture — PLI is a new *plugin*, not a new pipeline. **Freeze scope here** — whatever isn't started by end of day 3 gets cut.
- **Day 4 — Strengthen filters + PLI benchmark.** Add the config-model + L3 topology upgrade and the co-localization check; run the PLI DUD-E benchmark; add abstract-retrieval to the literature stream if time.
- **Day 5 — Integrate, polish demo, buffer.** End-to-end run on both verticals, the visualization dashboard, README/slides. Assume ~1 day slips.

SE approach: **keep the plugin/stream architecture you have.** Every filter is an independent module that emits a sub-score + provenance; add them incrementally; each is time-boxed and independently demoable. Don't refactor for elegance — the skeleton works, extend it.

---

## 3. Scoring strategy — funnel vs parallel-merge

Your funnel instinct is *partly* right, and the flowchart actually already resolves this: it shows a **parallel filtering framework → integrated evidence score → a "Review needed?" gate → expensive full-text only if needed.** That's not pure-funnel and not pure-parallel — it's a **hybrid**, and the hybrid is correct. Concretely, classify filters by type:

- **Cheap binary vetoes → funnel them to the front.** Known-positive / structured-DB screening removes pairs that are *already known to interact*. There's zero value in scoring a known positive, so remove these first, cheaply, before anything else. (Your pipeline already does this as exclusion.) The flowchart draws "known interaction screening" as a co-equal parallel filter — I'd pull it out as a pre-filter instead.
- **Graded biological scorers → run in parallel, merge.** Topology, chemical feasibility, co-localization are all cheap (CPU, sub-millisecond per pair). Run all of them on the survivors and merge into the integrated score. **Do not cascade these**, for two grounded reasons: (a) you lose the per-stream sub-scores that the transparency viz and ablation studies need ("% removed by each rule" / "which stream drove this call"); (b) a cascade makes the final confidence order-dependent and un-ablatable. Parallel-merge is what makes the "integrated evidence score" in the flowchart meaningful.
- **Expensive LLM → funnel/gate to the contested tail.** This is where your cost concern is genuinely load-bearing, and it's exactly what negaverse already does — run the LLM only on near-boundary / conflicting pairs (the "Review needed?" branch), never the full pool.

So the answer is: **funnel at both ends (cheap veto up front, expensive LLM at the back), parallel-merge in the middle.** The cost argument for funneling only really applies to the LLM and the DB screen — the middle graded filters are cheap enough to run on everything, and funneling them would cost you the transparency that's half your demo. The scale numbers back this: graded scoring is ms/pair even at 1M candidates; only the LLM tokens scale with coverage, and those are gated regardless.

---

## 4. Topology representation + skipping no-overlap pairs

**Graph DB vs NetworkX: use in-memory, no graph DB.** Grounded reasoning:

- Your topology operations — degree, common neighbors, L3 paths, configuration-model probabilities, community detection — are **bulk matrix computations, not interactive traversal queries.** Graph DBs (Neo4j) win at "traverse from node X following these edge types"; they add nothing for batch link-prediction math, and cost you ingestion + Cypher + infra you can't spare in 5 days.
- For PPI scale (≤~1M edges, ~20k nodes) everything fits in RAM trivially. Use **NetworkX for convenience/community detection, but `scipy.sparse` for the heavy math** — common neighbors are `A²`, L3 paths come from `A³`, config-model needs only the degree vector. Sparse matmul does these in seconds where per-pair Python loops would crawl.
- For the large PLI bipartite case (up to 20M edges) you wouldn't use NetworkX *or* a graph DB — you'd use sparse adjacency + precomputed embeddings + FAISS for candidate/matching. A graph DB doesn't help there either.

**Skipping topology for no-overlap pairs: don't skip the pair, but short-circuit the expensive computation — and understand *why*.** This is a subtle point worth getting right:

- A pair with no shared network context (no common neighbors, disconnected, no short path) is *topologically implausible as an interaction* → that's a **low topology score = a SAFE (but easy) negative.** The absence of overlap **is** the signal, not a reason to skip. If you silently skip, you lose that "this is a clean easy negative" information.
- The right move: compute a **cheap gate first** (common-neighbor count / same-component check). If it's zero, assign the floor topology score directly and route the pair to the "easy/safe negative" bucket — *without* running the expensive L3/path computation. That's an optimization (huge, since most random pairs have no overlap), not a semantic skip.
- The deeper implication for *hard*-negative mining: the pairs you actually care about are the ones **with** overlap — topologically plausible but non-interacting (near the decision boundary). So topology naturally partitions candidates into "overlap → hard, scrutinize, prioritize for LLM review" and "no overlap → easy, cheap filler." Your `EmbeddingStream` already floors to 0 when there are no shared neighbors; formalizing that as an explicit cheap gate + bucket is both a correctness win and your biggest compute saving.

---

**One tension to flag:** the flowchart already commits to the parallel-plus-gate design (which I'm endorsing), so if the team was debating pure-funnel, the flowchart is the better answer — I'd just move known-interaction screening to a pre-filter veto rather than a co-equal parallel branch.

If it's useful, I can turn this into a concrete 5-day plan doc (task breakdown + owner-agnostic checklist) or start on the highest-leverage Day-1 item — the PPI downstream-model benchmark (random vs negaverse negatives → AUROC/AUPRC), since that's the hypothesis test everything else supports.
