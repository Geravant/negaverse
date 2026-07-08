"""Anthropic API backend — uses the official `anthropic` SDK.

Credentials resolve the SDK's normal way: an explicit key, else ANTHROPIC_API_KEY,
else an `ant auth login` profile. Structured requests use `output_config.format`
(json_schema), so the response's first text block is valid JSON.
"""
from __future__ import annotations

from typing import Any, Optional

from .base import LLMAuthError, LLMError, LLMProvider, LLMResponse
from .config import LLMConfig


class AnthropicProvider(LLMProvider):
    name = "anthropic"

    def __init__(self, config: LLMConfig):
        self.config = config
        try:
            import anthropic  # lazy: only required when this backend is used
        except ImportError as e:  # pragma: no cover
            raise LLMError(
                "The Anthropic backend needs the 'anthropic' package: pip install anthropic"
            ) from e
        self._anthropic = anthropic
        key = config.resolved_key()
        kwargs: dict[str, Any] = {
            "timeout": config.timeout,
            "max_retries": config.max_retries,
        }
        if key:
            kwargs["api_key"] = key
        # No key + no ant profile still constructs; the error surfaces on first call.
        self.client = anthropic.Anthropic(**kwargs)

    def complete(
        self,
        system: Optional[str],
        messages: list[dict[str, str]],
        *,
        max_tokens: int,
        json_schema: Optional[dict] = None,
        **kwargs: Any,
    ) -> LLMResponse:
        req: dict[str, Any] = {
            "model": self.config.resolved_model(),
            "max_tokens": max_tokens,
            "messages": messages,
        }
        if system:
            req["system"] = system
        output_config: dict[str, Any] = {}
        if json_schema is not None:
            output_config["format"] = {"type": "json_schema", "schema": json_schema}
        if self.config.effort:
            output_config["effort"] = self.config.effort
            req["thinking"] = {"type": "adaptive"}
        if output_config:
            req["output_config"] = output_config

        try:
            resp = self.client.messages.create(**req)
        except self._anthropic.AuthenticationError as e:
            raise LLMAuthError(
                "Anthropic auth failed. Set ANTHROPIC_API_KEY or run `ant auth login`."
            ) from e
        except self._anthropic.APIError as e:  # rate limit / server / connection
            raise LLMError(f"Anthropic API error: {e}") from e

        if resp.stop_reason == "refusal":
            raise LLMError("Anthropic declined the request (stop_reason=refusal).")

        text = "".join(b.text for b in resp.content if getattr(b, "type", None) == "text")
        usage = getattr(resp, "usage", None)
        return LLMResponse(
            text=text,
            model=resp.model,
            provider=self.name,
            input_tokens=getattr(usage, "input_tokens", None),
            output_tokens=getattr(usage, "output_tokens", None),
            stop_reason=resp.stop_reason,
            raw=resp,
        )
