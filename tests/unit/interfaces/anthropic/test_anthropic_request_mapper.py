from __future__ import annotations

import json
from pathlib import Path

import pytest

from llm_proxy.domain.content import ReasoningContent, TextContent, ToolCallContent, ToolResultContent
from llm_proxy.domain.errors import InvalidCompletionRequest
from llm_proxy.domain.messages import MessageRole
from llm_proxy.domain.requests import AutomaticToolChoice, EffortLevel, NamedToolChoice, NoToolChoice
from llm_proxy.interfaces.anthropic.dto import AnthropicMessagesRequest
from llm_proxy.interfaces.anthropic.request_mapper import AnthropicRequestMapper


def test_preserves_system_message_and_tool_round_trip_order() -> None:
    request = AnthropicMessagesRequest.model_validate({
        "model": "local-opus", "max_tokens": 10,
        "system": [{"type": "text", "text": "first"}, {"type": "text", "text": "second"}],
        "messages": [
            {"role": "assistant", "content": [{"type": "text", "text": "calling"}, {"type": "tool_use", "id": "call-1", "name": "Read", "input": {"path": "README"}}]},
            {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "call-1", "content": "contents", "is_error": True}, {"type": "text", "text": "continue"}]},
        ],
        "tools": [{"name": "Read", "input_schema": {"$defs": {"path": {"type": "string"}}}}],
        "tool_choice": {"type": "tool", "name": "Read"},
    })

    result = AnthropicRequestMapper().map_request(request)

    assert [block.text for block in result.messages[0].content if isinstance(block, TextContent)] == ["first", "second"]
    assert isinstance(result.messages[1].content[1], ToolCallContent)
    assert isinstance(result.messages[2].content[0], ToolResultContent)
    assert result.messages[2].role is MessageRole.TOOL
    assert result.messages[2].content[0].is_error is True
    assert isinstance(result.tool_choice, NamedToolChoice)
    assert result.tools[0].input_schema["$defs"]["path"]["type"] == "string"


def test_tool_defaults_and_thinking_policy_are_explicit() -> None:
    base = {"model": "local-opus", "max_tokens": 10, "messages": [{"role": "assistant", "content": [{"type": "thinking", "thinking": "reasoning"}, {"type": "text", "text": "answer"}]}]}

    hidden = AnthropicRequestMapper().map_request(AnthropicMessagesRequest.model_validate(base))
    exposed = AnthropicRequestMapper(expose_thinking=True).map_request(AnthropicMessagesRequest.model_validate(base))
    no_tools = AnthropicRequestMapper().map_request(AnthropicMessagesRequest.model_validate({"model": "local-opus", "max_tokens": 10, "messages": [{"role": "user", "content": "hello"}]}))

    assert not any(isinstance(block, ReasoningContent) for block in hidden.messages[0].content)
    assert any(isinstance(block, ReasoningContent) for block in exposed.messages[0].content)
    assert isinstance(no_tools.tool_choice, NoToolChoice)

    with_tools = AnthropicRequestMapper().map_request(AnthropicMessagesRequest.model_validate({"model": "local-opus", "max_tokens": 10, "messages": [{"role": "user", "content": "hello"}], "tools": [{"name": "Read", "input_schema": {}}]}))
    assert isinstance(with_tools.tool_choice, AutomaticToolChoice)


def test_rejects_signed_thinking_redaction_and_unknown_named_tool() -> None:
    mapper = AnthropicRequestMapper(expose_thinking=True)
    for content in [
        [{"type": "thinking", "thinking": "reasoning", "signature": "signed"}],
        [{"type": "redacted_thinking", "data": "redacted"}],
    ]:
        request = AnthropicMessagesRequest.model_validate({"model": "local-opus", "max_tokens": 10, "messages": [{"role": "assistant", "content": content}]})
        with pytest.raises(InvalidCompletionRequest):
            mapper.map_request(request)
    request = AnthropicMessagesRequest.model_validate({"model": "local-opus", "max_tokens": 10, "messages": [{"role": "user", "content": "hello"}], "tool_choice": {"type": "tool", "name": "missing"}})
    with pytest.raises(InvalidCompletionRequest):
        AnthropicRequestMapper().map_request(request)


def test_maps_sampling_stream_and_defensively_copied_metadata() -> None:
    metadata = {"trace": {"id": "request-1"}}
    request = AnthropicMessagesRequest.model_validate({
        "model": "local-opus", "max_tokens": 42, "temperature": 0.2, "top_p": 0.8, "top_k": 40,
        "stop_sequences": ["first", "second"], "stream": True, "metadata": metadata,
        "messages": [{"role": "user", "content": "hello"}],
    })

    result = AnthropicRequestMapper().map_request(request)
    metadata["trace"]["id"] = "changed"

    assert result.parameters.max_tokens == 42
    assert result.parameters.temperature == 0.2
    assert result.parameters.top_p == 0.8
    assert result.parameters.top_k == 40
    assert result.parameters.stop_sequences == ("first", "second")
    assert result.stream is True
    assert result.metadata == {"trace": {"id": "request-1"}}


def test_splits_interleaved_user_text_and_tool_results_into_provider_mappable_messages() -> None:
    request = AnthropicMessagesRequest.model_validate({
        "model": "local-opus", "max_tokens": 10,
        "messages": [{"role": "user", "content": [
            {"type": "text", "text": "before"},
            {"type": "tool_result", "tool_use_id": "call-1", "content": "result"},
            {"type": "text", "text": "after"},
        ]}],
    })

    result = AnthropicRequestMapper().map_request(request)

    assert [message.role for message in result.messages] == [MessageRole.USER, MessageRole.TOOL, MessageRole.USER]
    assert result.messages[1].content[0].tool_call_id == "call-1"


def test_maps_captured_claude_code_output_controls() -> None:
    fixture = Path("tests/fixtures/claude_code/title-request.json")
    request = AnthropicMessagesRequest.model_validate(json.loads(fixture.read_text()))

    result = AnthropicRequestMapper().map_request(request)

    assert result.controls.effort is EffortLevel.HIGH
    assert result.controls.output_constraint.schema["properties"]["title"]["type"] == "string"
