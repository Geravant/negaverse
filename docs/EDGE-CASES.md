# Edge cases

Observed edge cases and how negaverse handles them. Append new entries at the top.

---

## EC-1 — Literature verdict flip-flop on hub-dominated contested pairs

**Symptom.** In single-call mode, the gated LLM literature verdict for certain
contested pairs was unstable across otherwise-identical runs — the same pair
(e.g. `P49643 × orf8`) came back `suspected_false_negative` in one run and
`uncertain` in the next. This made the fused confidence (and therefore whether a
record got the `suspected_false_negative` flag) non-deterministic.

**Root cause.** The contested pairs routed to the gated stage are currently
selected almost entirely by the **structured promiscuity prior**: they all pair a
host protein with **ORF8**, the most promiscuous viral bait (degree 47), whose
prior drives confidence to ~0.5. The topology stream contributes little on this
near-star SARS graph (topo ≈ 0 for most viral–host non-edges). So the contested
set is effectively **mono-signal** — the pairs are genuinely ambiguous to the
model (a promiscuous hub *could* plausibly bind almost anything), and a single
call samples a near-coin-flip between `uncertain` and `suspected_false_negative`.

**Confirmation via voting.** Best-of-5 majority voting exposed the true split:
`P49643 × orf8` was **2 `suspected_false_negative` vs 3 `uncertain`** — a genuine
near-tie, majority-resolving to `uncertain`. The other 7 contested pairs were
unanimous `uncertain` (5/5). So the intermittent `suspected_false_negative` was
minority noise, not a stable judgement.

**Handling (implemented).**
- `LiteratureFilter` uses best-of-N majority voting (default N=5, `--votes`).
  Ties on the top verdict resolve to `uncertain` (don't force a call).
- Each card records `agreement` (winning-verdict fraction) and `vote_counts`, so
  a bare-majority verdict is visibly low-confidence rather than silently trusted.

**How to read it.** Treat a **low `agreement`** (e.g. 0.6) as a signal in its own
right — the model is genuinely undecided, and such pairs are candidates for human
review or a higher vote count, not confident negatives.

**Roadmap implication.** The deeper cause is that the contested set is
promiscuity-driven, not topology-driven, so the LLM has little to adjudicate.
The Phase-1 topology upgrade (configuration-model + L3) and a denser human↔human
graph should produce genuinely near-boundary contested pairs — topologically
plausible but non-interacting — that the literature stream can actually resolve,
raising agreement and making the verdicts informative rather than mostly
`uncertain`. Worth re-checking this edge case after that lands.
