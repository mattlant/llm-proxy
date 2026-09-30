from __future__ import annotations

import pytest

from llm_proxy.domain.content import ToolCallContent, ToolResultContent
from llm_proxy.domain.errors import InvalidCompletionRequest
from llm_proxy.domain.messages import MessageRole
from llm_proxy.domain.requests import NamedToolChoice
from llm_proxy.interfaces.openai.dto import OpenAIChatCompletionRequest
from llm_proxy.interfaces.openai.request_mapper import OpenAIRequestMapper


def test_maps_messages_tools_choice_and_sampling_parameters() -> None:
    source = OpenAIChatCompletionRequest.model_validate({
        "model": "alias", "messages": [
            {"role": "system", "content": "system"},
            {"role": "assistant", "content": "", "tool_calls": [{"id": "call-1", "type": "function", "function": {"name": "read", "arguments": "{\"path\":\"README\"}"}}]},
            {"role": "tool", "tool_call_id": "call-1", "content": "result"},
        ],
        "tools": [{"type": "function", "function": {"name": "read", "description": "Read", "parameters": {"type": "object", "properties": {"path": {"type": "string"}}}}}],
        "tool_choice": {"type": "function", "function": {"name": "read"}},
        "temperature": 0.2, "top_k": 10, "stop": "DONE", "metadata": {"tenant": "a"},
    })

    result = OpenAIRequestMapper().map_request(source)

    assert result.model == "alias"
    assert isinstance(result.messages[1].content[1], ToolCallContent)
    assert result.messages[1].content[1].arguments == {"path": "README"}
    assert isinstance(result.messages[2].content[0], ToolResultContent)
    assert result.messages[2].content[0].is_error is False
    assert isinstance(result.tool_choice, NamedToolChoice)
    assert result.parameters.stop_sequences == ("DONE",)
    assert result.parameters.top_k == 10
    assert result.metadata == {"tenant": "a"}


def test_rejects_malformed_historical_tool_arguments() -> None:
    source = OpenAIChatCompletionRequest.model_validate({
        "model": "alias", "messages": [{"role": "assistant", "tool_calls": [{"id": "call-1", "type": "function", "function": {"name": "read", "arguments": "[]"}}]}],
    })

    with pytest.raises(InvalidCompletionRequest, match="JSON object"):
        OpenAIRequestMapper().map_request(source)


def test_preserves_developer_message_identity() -> None:
    source = OpenAIChatCompletionRequest.model_validate({
        "model": "alias",
        "messages": [{"role": "developer", "content": "be concise"}],
    })

    result = OpenAIRequestMapper().map_request(source)

    assert result.messages[0].role is MessageRole.DEVELOPER
    assert result.messages[0].content[0].text == "be concise"


def test_preserves_developer_message_order_and_content() -> None:
    source = OpenAIChatCompletionRequest.model_validate({
        "model": "alias",
        "messages": [
            {"role": "system", "content": "system"},
            {"role": "developer", "content": "developer"},
            {"role": "user", "content": "user"},
        ],
    })

    result = OpenAIRequestMapper().map_request(source)

    assert [message.role for message in result.messages] == [MessageRole.SYSTEM, MessageRole.DEVELOPER, MessageRole.USER]
    assert [message.content[0].text for message in result.messages] == ["system", "developer", "user"]


def test_maps_structured_user_text_content_in_order() -> None:
    source = OpenAIChatCompletionRequest.model_validate({
        "model": "alias",
        "messages": [{"role": "user", "content": [
            {"type": "text", "text": "one"},
            {"type": "text", "text": "two"},
        ]}],
    })

    result = OpenAIRequestMapper().map_request(source)

    assert result.messages[0].content[0].text == "onetwo"
