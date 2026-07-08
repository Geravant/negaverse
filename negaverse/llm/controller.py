"""LLMController — one entry point for negaverse's LLM calls.

Selects a backend from `LLMConfig.provider`, exposes `complete()` for free text
and `complete_json()` for schema-constrained structured output, and normalises
JSON parsing across providers (Anthropic's `output_config.format` guarantees
clean JSON; OpenRouter models sometimes wrap it in prose or code fences).
"""
from __future__ import annotations

import json
import re
from typing import Any, Optional

from .base import LLMError, LLMProvider, LLMResponse
from .config import LLMConfig

_FENCE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL)


def _build_provider(config: LLMConfig) -> LLMProvider:
    if config.provider == "anthropic":
        from .anthropic_provider import AnthropicProvider
        return AnthropicProvider(config)
    if config.provider == "openrouter":
        from .openrouter_provider import OpenRouterProvider
        return OpenRouterProvider(config)
    raise LLMError(f"Unknown provider: {config.provider!r}")


def _extract_json(text: str) -> str:
    """Pull the JSON object out of a model response that may include prose or
    a ```json fence (OpenRouter models don't always honour response_format)."""
    m = _FENCE.search(text)
    if m:
        return m.group(1)
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end > start:
        return text[start:end + 1]
    return text


class LLMController:
    def __init__(self, config: Optional[LLMConfig] = None):
        self.config = config or LLMConfig()
        self.provider = _build_provider(self.config)

    @property
    def describe(self) -> str:
        return f"{self.config.provider}:{self.config.resolved_model()}"

    def complete(
        self,
        user: str,
        *,
        system: Optional[str] = None,
        max_tokens: Optional[int] = None,
    ) -> LLMResponse:
        return self.provider.complete(
            system,
            [{"role": "user", "content": user}],
            max_tokens=max_tokens or self.config.max_tokens,
        )

    def complete_json(
        self,
        user: str,
        schema: dict,
        *,
        system: Optional[str] = None,
        max_tokens: Optional[int] = None,
    ) -> tuple[dict[str, Any], LLMResponse]:
        """Return (parsed_object, response). Raises LLMError if the model's
        output can't be parsed as JSON."""
        resp = self.provider.complete(
            system,
            [{"role": "user", "content": user}],
            max_tokens=max_tokens or self.config.max_tokens,
            json_schema=schema,
        )
        try:
            return json.loads(resp.text), resp
        except json.JSONDecodeError:
            try:
                return json.loads(_extract_json(resp.text)), resp
            except json.JSONDecodeError as e:
                raise LLMError(
                    f"Could not parse JSON from {self.describe}: {resp.text[:200]!r}"
                ) from e
