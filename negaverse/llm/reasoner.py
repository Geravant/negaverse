"""Literature-reasoning pass (ARCHITECTURE.md §5.2), gated per §8.5.

Runs the LLM controller on a small set of contested / near-boundary candidate
pairs — never the full pool — and returns a structured, citable judgement of
whether the pair is a true non-interaction. This is the differentiated Claude
step; it degrades gracefully (no key -> the pipeline just skips it).
"""
from __future__ import annotations

import concurrent.futures as cf
from collections import Counter
from dataclasses import dataclass, field, asdict
from typing import Any, Optional

from .base import LLMError
from .controller import LLMController

# Schema is intentionally strict-output friendly: all keys required,
# additionalProperties disabled (Anthropic + OpenRouter json_schema both want this).
CARD_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "verdict": {
            "type": "string",
            "enum": ["safe_negative", "suspected_false_negative", "uncertain"],
        },
        "confidence": {"type": "number"},
        "rationale": {"type": "string"},
        "evidence": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["verdict", "confidence", "rationale", "evidence"],
    "additionalProperties": False,
}

SYSTEM = (
    "You are a structural and molecular biology expert assessing whether a "
    "proposed protein pair is a TRUE non-interaction (a safe negative for "
    "training interaction-prediction models). Weigh subcellular localization, "
    "known complexes, pathway membership, and any literature you are aware of. "
    "A pair you believe likely DOES interact is a suspected false negative. "
    "Cite concrete biological reasons; do not fabricate citations."
)


@dataclass
class LiteratureCard:
    u: str
    v: str
    verdict: str
    confidence: float
    rationale: str
    evidence: list[str]
    model: str
    # populated when the verdict is a majority vote of several LLM calls
    n_votes: int = 1
    agreement: float = 1.0                       # winning-verdict fraction
    vote_counts: dict[str, int] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


class LiteratureReasoner:
    def __init__(self, controller: LLMController, max_tokens: int = 1024):
        self.controller = controller
        self.max_tokens = max_tokens

    def _prompt(self, u: str, v: str, context: Optional[dict]) -> str:
        ctx = ""
        if context:
            lines = [f"- {k}: {val}" for k, val in context.items()]
            ctx = "\nContext:\n" + "\n".join(lines)
        return (
            f"Proposed non-interacting pair: {u} and {v}.{ctx}\n\n"
            "Judge whether this is a safe negative or a suspected false negative, "
            "and return the structured verdict."
        )

    def reason(self, u: str, v: str, context: Optional[dict] = None) -> LiteratureCard:
        obj, resp = self.controller.complete_json(
            self._prompt(u, v, context),
            CARD_SCHEMA,
            system=SYSTEM,
            max_tokens=self.max_tokens,
        )
        return LiteratureCard(
            u=u, v=v,
            verdict=str(obj.get("verdict", "uncertain")),
            confidence=float(obj.get("confidence", 0.0)),
            rationale=str(obj.get("rationale", "")),
            evidence=list(obj.get("evidence", []) or []),
            model=resp.model,
        )

    def reason_vote(self, u: str, v: str, context: Optional[dict] = None,
                    votes: int = 5, max_workers: int = 5) -> LiteratureCard:
        """Best-of-N majority vote: run `votes` independent judgements in
        parallel and aggregate. Cuts the run-to-run variance of a single call.
        Ties on the top verdict resolve to `uncertain` (don't force a call)."""
        if votes <= 1:
            return self.reason(u, v, context)

        cards: list[LiteratureCard] = []
        with cf.ThreadPoolExecutor(max_workers=min(max_workers, votes)) as ex:
            futures = [ex.submit(self.reason, u, v, context) for _ in range(votes)]
            for fut in cf.as_completed(futures):
                try:
                    cards.append(fut.result())
                except LLMError:
                    continue
        if not cards:
            raise LLMError(f"all {votes} literature votes failed for {u}-{v}")

        counts = Counter(c.verdict for c in cards)
        ranked = counts.most_common()
        if len(ranked) > 1 and ranked[0][1] == ranked[1][1]:
            winner = "uncertain"                 # tie on the top verdict -> abstain
        else:
            winner = ranked[0][0]

        winners = [c for c in cards if c.verdict == winner]
        if winners:
            conf = sum(c.confidence for c in winners) / len(winners)
            rep = max(winners, key=lambda c: c.confidence)
        else:                                    # forced-uncertain with no such vote
            conf = 0.5
            rep = max(cards, key=lambda c: c.confidence)
        return LiteratureCard(
            u=u, v=v, verdict=winner, confidence=round(conf, 4),
            rationale=rep.rationale, evidence=rep.evidence, model=rep.model,
            n_votes=len(cards),
            agreement=round(counts.get(winner, 0) / len(cards), 3),
            vote_counts=dict(counts),
        )

    def reason_batch(
        self, pairs: list[tuple[str, str, Optional[dict]]]
    ) -> list[LiteratureCard]:
        """Best-effort over a gated set; a per-pair failure is skipped, not fatal."""
        cards: list[LiteratureCard] = []
        for u, v, ctx in pairs:
            try:
                cards.append(self.reason(u, v, ctx))
            except LLMError:
                continue
        return cards
