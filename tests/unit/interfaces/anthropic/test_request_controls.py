from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from llm_proxy.domain.requests import ReasoningDisplay, ReasoningMode
from llm_proxy.interfaces.anthropic.dto import AnthropicMessagesRequest
from llm_proxy.interfaces.anthropic.request_mapper import AnthropicRequestMapper


def body() -> dict:
    return {"model": "local-opus", "max_tokens": 1024, "messages": [{"role": "user", "content": "hello"}]}


def test_maps_observed_adaptive_thinking_request() -> None:
    request = AnthropicMessagesRequest.model_validate({**body(), "thinking": {"type": "adaptive", "display": "omitted"}})

    result = AnthropicRequestMapper().map_request(request)

    assert result.controls.reasoning.mode is ReasoningMode.ADAPTIVE
    assert result.controls.reasoning.display is ReasoningDisplay.OMITTED


def test_maps_enabled_and_disabled_thinking_requests() -> None:
    enabled = AnthropicRequestMapper().map_request(AnthropicMessagesRequest.model_validate({**body(), "thinking": {"type": "enabled", "budget_tokens": 128, "display": "summarized"}}))
    disabled = AnthropicRequestMapper().map_request(AnthropicMessagesRequest.model_validate({**body(), "thinking": {"type": "disabled"}}))

    assert enabled.controls.reasoning.budget_tokens == 128
    assert disabled.controls.reasoning.mode is ReasoningMode.DISABLED
    assert disabled.controls.reasoning.display is ReasoningDisplay.OMITTED


def test_maps_captured_session_context_management_request() -> None:
    fixture = Path("tests/fixtures/claude_code/session-request.json")
    request = AnthropicMessagesRequest.model_validate(json.loads(fixture.read_text()))

    result = AnthropicRequestMapper().map_request(request)

    assert result.controls.context_management.edits[0].kind.value == "clear_thinking_20251015"
    assert result.controls.context_management.edits[0].keep == "all"
    assert result.controls.prompt_cache.ephemeral_blocks == 3


def test_maps_observed_ephemeral_prompt_cache_markers_as_advisory() -> None:
    request = AnthropicMessagesRequest.model_validate({
        **body(),
        "system": [{"type": "text", "text": "one"}, {"type": "text", "text": "two", "cache_control": {"type": "ephemeral"}}],
        "messages": [{"role": "user", "content": [{"type": "text", "text": "hello", "cache_control": {"type": "ephemeral"}}]}],
    })

    result = AnthropicRequestMapper().map_request(request)

    assert result.controls.prompt_cache.ephemeral_blocks == 2


@pytest.mark.parametrize("cache_control", [{"type": "persistent"}, {"type": "ephemeral", "unknown": True}])
def test_rejects_unknown_prompt_cache_contracts(cache_control) -> None:
    with pytest.raises(ValidationError):
        AnthropicMessagesRequest.model_validate({**body(), "messages": [{"role": "user", "content": [{"type": "text", "text": "hello", "cache_control": cache_control}]}]})


def test_maps_observed_tool_result_prompt_cache_marker() -> None:
    request = AnthropicMessagesRequest.model_validate({
        **body(),
        "messages": [{"role": "user", "content": [{"type": "tool_result", "tool_use_id": "call-1", "content": "result", "cache_control": {"type": "ephemeral"}}]}],
    })

    result = AnthropicRequestMapper().map_request(request)

    assert result.controls.prompt_cache.ephemeral_blocks == 1


@pytest.mark.parametrize("thinking", [
    {"type": "unknown", "display": "omitted"},
    {"type": "adaptive", "display": "omitted", "budget_tokens": 1},
    {"type": "enabled", "display": "omitted"},
    {"type": "enabled", "budget_tokens": 1024, "display": "omitted"},
    {"type": "adaptive", "display": "verbose"},
    {"type": "disabled", "unknown": True},
])
def test_rejects_invalid_thinking_contracts(thinking) -> None:
    with pytest.raises(ValidationError):
        AnthropicMessagesRequest.model_validate({**body(), "thinking": thinking})


@pytest.mark.parametrize("context_management", [
    {"edits": []},
    {"edits": [{"type": "clear_thinking_20251015", "keep": "recent"}]},
    {"edits": [{"type": "clear_tool_uses_20250919"}]},
    {"edits": [{"type": "compact_20260101"}]},
    {"edits": [{"type": "clear_thinking_20251015", "keep": "all", "unknown": True}]},
])
def test_rejects_unobserved_or_invalid_context_management_contracts(context_management) -> None:
    with pytest.raises(ValidationError):
        AnthropicMessagesRequest.model_validate({**body(), "context_management": context_management})
