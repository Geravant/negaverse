"""OpenRouter backend — OpenAI-compatible chat/completions over HTTP.

OpenRouter is not Anthropic-native, so this backend deliberately does NOT use the
`anthropic` SDK; it speaks the OpenAI-compatible wire format with `httpx`. This
lets negaverse route the literature stream to any model OpenRouter exposes
(Claude, GPT, Llama, ...) behind one key.
"""
from __future__ import annotations

import json
import time
from typing import Any, Optional

import httpx

from .base import LLMAuthError, LLMError, LLMProvider, LLMResponse
from .config import LLMConfig, OPENROUTER_BASE_URL


class OpenRouterProvider(LLMProvider):
    name = "openrouter"

    def __init__(self, config: LLMConfig):
        self.config = config
        self.base_url = (config.base_url or OPENROUTER_BASE_URL).rstrip("/")

    def _headers(self) -> dict[str, str]:
        key = self.config.resolved_key()
        if not key:
            raise LLMAuthError("OpenRouter needs OPENROUTER_API_KEY (or config.api_key).")
        return {
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
            "HTTP-Referer": self.config.referer,
            "X-Title": self.config.title,
        }

    def complete(
        self,
        system: Optional[str],
        messages: list[dict[str, str]],
        *,
        max_tokens: int,
        json_schema: Optional[dict] = None,
        **kwargs: Any,
    ) -> LLMResponse:
        chat = ([{"role": "system", "content": system}] if system else []) + list(messages)
        body: dict[str, Any] = {
            "model": self.config.resolved_model(),
            "messages": chat,
            "max_tokens": max_tokens,
        }
        if json_schema is not None:
            body["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": "response", "strict": True, "schema": json_schema},
            }

        url = f"{self.base_url}/chat/completions"
        headers = self._headers()
        last_exc: Optional[Exception] = None
        for attempt in range(self.config.max_retries + 1):
            try:
                r = httpx.post(url, headers=headers, json=body, timeout=self.config.timeout)
            except httpx.HTTPError as e:
                last_exc = e
            else:
                if r.status_code == 401:
                    raise LLMAuthError("OpenRouter rejected the API key (401).")
                if r.status_code in (429, 500, 502, 503, 504):
                    last_exc = LLMError(f"OpenRouter {r.status_code}: {r.text[:200]}")
                elif r.status_code >= 400:
                    raise LLMError(f"OpenRouter {r.status_code}: {r.text[:300]}")
                else:
                    return self._parse(r.json())
            if attempt < self.config.max_retries:
                time.sleep(min(2.0 * (2 ** attempt), 15.0))
        raise LLMError(f"OpenRouter request failed after retries: {last_exc}")

    def _parse(self, data: dict) -> LLMResponse:
        try:
            choice = data["choices"][0]
            text = choice["message"]["content"] or ""
            finish = choice.get("finish_reason")
        except (KeyError, IndexError) as e:
            raise LLMError(f"Unexpected OpenRouter response shape: {json.dumps(data)[:300]}") from e
        usage = data.get("usage") or {}
        return LLMResponse(
            text=text,
            model=data.get("model", self.config.resolved_model()),
            provider=self.name,
            input_tokens=usage.get("prompt_tokens"),
            output_tokens=usage.get("completion_tokens"),
            stop_reason=finish,
            raw=data,
        )
