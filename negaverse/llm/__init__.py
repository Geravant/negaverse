"""LLM controller for negaverse — Anthropic API or OpenRouter behind one interface."""
from .base import LLMResponse, LLMError, LLMAuthError, LLMProvider
from .config import LLMConfig
from .controller import LLMController
from .reasoner import LiteratureReasoner, LiteratureCard, CARD_SCHEMA

__all__ = [
    "LLMController",
    "LLMConfig",
    "LLMResponse",
    "LLMError",
    "LLMAuthError",
    "LLMProvider",
    "LiteratureReasoner",
    "LiteratureCard",
    "CARD_SCHEMA",
]
