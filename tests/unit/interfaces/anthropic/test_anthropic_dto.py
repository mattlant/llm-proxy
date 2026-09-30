from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from llm_proxy.interfaces.anthropic.dto import AnthropicMessagesRequest


def request_body() -> dict:
    return {"model": "local-opus", "max_tokens": 10, "messages": [{"role": "user", "content": "hello"}]}


def test_accepts_supported_message_and_tool_shapes() -> None:
    request = AnthropicMessagesRequest.model_validate({
        **request_body(),
        "system": [{"type": "text", "text": "first"}, {"type": "text", "text": "second"}],
        "tools": [{"name": "Read", "input_schema": {"oneOf": [{"type": "object"}]}}],
        "tool_choice": {"type": "tool", "name": "Read"},
    })

    assert request.system[1].text == "second"
    assert request.tools[0].input_schema["oneOf"] == [{"type": "object"}]


@pytest.mark.parametrize("update", [
    {"max_tokens": 0},
    {"messages": []},
    {"messages": [{"role": "system", "content": "no"}]},
    {"tools": [{"name": "same", "input_schema": {}}, {"name": "same", "input_schema": {}}]},
    {"unknown": True},
])
def test_rejects_invalid_request_shapes(update) -> None:
    with pytest.raises(ValidationError):
        AnthropicMessagesRequest.model_validate({**request_body(), **update})


def test_rejects_non_json_metadata() -> None:
    with pytest.raises(ValidationError, match="JSON-compatible"):
        AnthropicMessagesRequest.model_validate({**request_body(), "metadata": {"invalid": {"set"}}})


def test_accepts_captured_claude_code_title_output_config() -> None:
    fixture = Path("tests/fixtures/claude_code/title-request.json")
    request = AnthropicMessagesRequest.model_validate(json.loads(fixture.read_text()))

    assert request.output_config.effort == "high"
    assert request.output_config.format.type == "json_schema"


@pytest.mark.parametrize("output_config", [
    {"effort": "ultra"},
    {"format": {"type": "text", "schema": {}}},
    {"format": {"type": "json_schema"}},
    {"format": {"type": "json_schema", "schema": []}},
    {"unexpected": True},
])
def test_rejects_invalid_output_config(output_config) -> None:
    with pytest.raises(ValidationError):
        AnthropicMessagesRequest.model_validate({**request_body(), "output_config": output_config})
