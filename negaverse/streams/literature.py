"""Literature-reasoning filter (ARCHITECTURE.md §5.2) — GATED stage.

Runs the LLM (Anthropic API or OpenRouter, via `negaverse.llm`) on the contested
pairs the pipeline routes to the gated stage, and maps the structured verdict
into the fused confidence:

  * safe_negative           -> high true-negative confidence (= verdict conf)
  * suspected_false_negative -> low  true-negative confidence (= 1 - verdict conf)
                                + a `suspected_false_negative` flag
  * uncertain               -> abstain (no confidence contribution)

Safe by default: `enabled=False` abstains (pure stub, no API calls) so library
use and tests never hit the network. The CLI constructs it with `enabled=True`.
Degrades gracefully — no key / API error -> abstain, pipeline unaffected.
"""
from __future__ import annotations

import os
from typing import Optional

from ..graph import TypedInteractionGraph
from ..schema import StreamScore
from .base import Filter, Stage
from .registry import register

_PROVIDER_ENV = {"anthropic": "ANTHROPIC_API_KEY", "openrouter": "OPENROUTER_API_KEY"}


@register
class LiteratureFilter(Filter):
    name = "literature"
    stage = Stage.GATED

    def __init__(self, enabled: bool = False, provider: str = "auto",
                 model: Optional[str] = None, max_tokens: int = 1024):
        self.enabled = enabled
        self.provider = provider
        self.model = model
        self.max_tokens = max_tokens
        self._reasoner = None
        self._resolved: Optional[str] = None
        self._initialized = False
        self._cache: dict[frozenset, StreamScore] = {}

    def _resolve_provider(self) -> Optional[str]:
        if self.provider == "auto":
            for p, env in _PROVIDER_ENV.items():
                if os.getenv(env):
                    return p
            return None
        return self.provider if os.getenv(_PROVIDER_ENV[self.provider]) else None

    def _ensure(self) -> None:
        if self._initialized:
            return
        self._initialized = True
        if not self.enabled:
            return
        prov = self._resolve_provider()
        if prov is None:
            return
        try:
            from ..llm import LLMConfig, LLMController, LiteratureReasoner
            self._reasoner = LiteratureReasoner(
                LLMController(LLMConfig(provider=prov, model=self.model)),
                max_tokens=self.max_tokens,
            )
            self._resolved = prov
        except Exception:
            self._reasoner = None

    def _skip(self, reason: str) -> StreamScore:
        return StreamScore(self.name, value=None,
                           evidence={"gated_status": "skipped", "reason": reason})

    def score(self, graph: TypedInteractionGraph, u: str, v: str) -> StreamScore:
        self._ensure()
        if self._reasoner is None:
            return self._skip("disabled" if not self.enabled else "no_api_key")
        key = frozenset((u, v))
        if key in self._cache:
            return self._cache[key]

        ctx = {"u_type": graph.node_type.get(u), "v_type": graph.node_type.get(v),
               "u_degree": graph.degree(u), "v_degree": graph.degree(v)}
        try:
            card = self._reasoner.reason(u, v, ctx)
        except Exception:
            sc = self._skip("llm_error")
            self._cache[key] = sc
            return sc

        verdict, conf = card.verdict, float(card.confidence)
        flags: list[str] = []
        if verdict == "safe_negative":
            value: Optional[float] = round(conf, 4)
        elif verdict == "suspected_false_negative":
            value = round(1.0 - conf, 4)
            flags = ["suspected_false_negative"]
        else:  # uncertain
            value = None
        sc = StreamScore(
            self.name, value=value, flags=flags,
            evidence={"gated_status": "reviewed", "verdict": verdict,
                      "verdict_confidence": round(conf, 4), "rationale": card.rationale,
                      "evidence": card.evidence, "model": card.model},
        )
        self._cache[key] = sc
        return sc


# Back-compat alias.
LiteratureStream = LiteratureFilter
