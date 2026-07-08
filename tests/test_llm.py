"""Offline tests for the LLM controller — no network, no API key required.

    python -m tests.test_llm
"""
from __future__ import annotations

import os

from negaverse.llm import LLMConfig, LLMController, LLMAuthError, CARD_SCHEMA
from negaverse.llm.controller import _extract_json
from negaverse.llm.config import DEFAULT_MODELS


def test_default_models_per_provider():
    assert LLMConfig(provider="anthropic").resolved_model() == DEFAULT_MODELS["anthropic"]
    assert LLMConfig(provider="openrouter").resolved_model() == DEFAULT_MODELS["openrouter"]
    assert LLMConfig(provider="openrouter", model="openai/gpt-4o").resolved_model() == "openai/gpt-4o"


def test_key_resolution_prefers_explicit():
    cfg = LLMConfig(provider="openrouter", api_key="explicit")
    assert cfg.resolved_key() == "explicit"


def test_extract_json_from_fenced_and_prose():
    assert _extract_json('```json\n{"a": 1}\n```') == '{"a": 1}'
    assert _extract_json('Here is the result: {"a": 1, "b": 2} done.') == '{"a": 1, "b": 2}'


def test_openrouter_without_key_raises_auth():
    # ensure no ambient key
    old = os.environ.pop("OPENROUTER_API_KEY", None)
    try:
        ctrl = LLMController(LLMConfig(provider="openrouter"))
        raised = False
        try:
            ctrl.complete("hello")
        except LLMAuthError:
            raised = True
        assert raised, "expected LLMAuthError without OPENROUTER_API_KEY"
    finally:
        if old is not None:
            os.environ["OPENROUTER_API_KEY"] = old


def test_card_schema_is_strict_output_friendly():
    assert CARD_SCHEMA["additionalProperties"] is False
    assert set(CARD_SCHEMA["required"]) == {"verdict", "confidence", "rationale", "evidence"}


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"PASS  {fn.__name__}")
    print(f"\n{len(fns)} checks passed")
