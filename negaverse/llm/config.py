"""Configuration for the LLM controller.

Providers and their defaults live here so the rest of negaverse only ever sees
an `LLMConfig`. Credentials resolve from the environment at call time (never
hardcode a key): ANTHROPIC_API_KEY for the Anthropic backend, OPENROUTER_API_KEY
for OpenRouter.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Literal, Optional

Provider = Literal["anthropic", "openrouter"]

# Sensible per-provider default models. Anthropic's default follows Anthropic's
# own guidance (Opus 4.8); OpenRouter routes the same family via its slug.
DEFAULT_MODELS: dict[str, str] = {
    "anthropic": "claude-opus-4-8",
    "openrouter": "anthropic/claude-opus-4-8",
}

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


@dataclass
class LLMConfig:
    provider: Provider = "anthropic"
    model: Optional[str] = None            # None -> DEFAULT_MODELS[provider]
    max_tokens: int = 4096
    # Anthropic thinking effort (low|medium|high|xhigh|max); None -> no thinking.
    effort: Optional[str] = None
    api_key: Optional[str] = None          # None -> resolved from env at call time
    base_url: Optional[str] = None         # override the provider endpoint
    timeout: float = 60.0
    max_retries: int = 3
    # OpenRouter attribution headers (optional but recommended by OpenRouter)
    referer: str = "https://github.com/negaverse"
    title: str = "negaverse"

    def resolved_model(self) -> str:
        return self.model or DEFAULT_MODELS[self.provider]

    def resolved_key(self) -> Optional[str]:
        if self.api_key:
            return self.api_key
        env = "ANTHROPIC_API_KEY" if self.provider == "anthropic" else "OPENROUTER_API_KEY"
        return os.getenv(env)
