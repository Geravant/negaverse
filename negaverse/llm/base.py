"""Provider-agnostic types for the LLM controller.

The controller powers negaverse's literature-reasoning stream (ARCHITECTURE.md
§5.2): a gated LLM call that reads evidence about a candidate pair and returns a
structured non-interaction judgement. Two backends implement this interface —
the Anthropic API (official SDK) and OpenRouter (OpenAI-compatible HTTP).
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class LLMResponse:
    text: str
    model: str
    provider: str
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    stop_reason: Optional[str] = None
    raw: Any = None

    @property
    def total_tokens(self) -> Optional[int]:
        if self.input_tokens is None or self.output_tokens is None:
            return None
        return self.input_tokens + self.output_tokens


class LLMProvider(ABC):
    """One concrete backend (Anthropic, OpenRouter, ...)."""

    name: str

    @abstractmethod
    def complete(
        self,
        system: Optional[str],
        messages: list[dict[str, str]],
        *,
        max_tokens: int,
        json_schema: Optional[dict] = None,
        **kwargs: Any,
    ) -> LLMResponse:
        """One request. If json_schema is given, the provider constrains the
        response to that schema and `LLMResponse.text` is the JSON string."""


class LLMError(RuntimeError):
    """Raised for auth/config/transport failures the caller should surface."""


class LLMAuthError(LLMError):
    """No usable credentials for the selected provider."""
