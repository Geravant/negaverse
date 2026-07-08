"""Literature-reasoning pass (ARCHITECTURE.md §5.2), gated per §8.5.

Runs the LLM controller on a small set of contested / near-boundary candidate
pairs — never the full pool — and returns a structured, citable judgement of
whether the pair is a true non-interaction. This is the differentiated Claude
step; it degrades gracefully (no key -> the pipeline just skips it).
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
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
