from __future__ import annotations

import pytest

from llm_proxy.domain.content import TextContent, ToolCallContent
from llm_proxy.domain.messages import Message, MessageRole
from llm_proxy.domain.responses import CompletionResponse, FinishReason, Usage


def test_completion_response_represents_tool_use_and_unknown_usage():
    response = CompletionResponse(
        id="response-1",
        model="model",
        message=Message(
            MessageRole.ASSISTANT,
            (ToolCallContent("call-1", "lookup", {"query": "value"}),),
        ),
        finish_reason=FinishReason.TOOL_USE,
        stop_sequence=None,
        usage=Usage(input_tokens=None, output_tokens=None),
    )

    assert response.finish_reason is FinishReason.TOOL_USE
    assert response.usage == Usage(None, None)


def test_completion_response_preserves_arbitrary_stop_sequence():
    response = CompletionResponse(
        "response-1",
        "model",
        Message(MessageRole.ASSISTANT, (TextContent("done"),)),
        FinishReason.STOP_SEQUENCE,
        "CUSTOM_STOP",
        Usage(12, 4),
    )

    assert response.stop_sequence == "CUSTOM_STOP"
    assert response.usage == Usage(12, 4)


def test_negative_usage_counts_are_rejected():
    with pytest.raises(ValueError):
        Usage(-1, 0)
    with pytest.raises(ValueError):
        Usage(0, -1)


def test_response_and_usage_are_frozen():
    usage = Usage(1, 2)
    response = CompletionResponse(
        "response-1",
        "model",
        Message(MessageRole.ASSISTANT, (TextContent("done"),)),
        FinishReason.END_TURN,
        None,
        usage,
    )

    with pytest.raises(AttributeError):
        response.model = "other"  # type: ignore[misc]
    with pytest.raises(AttributeError):
        usage.input_tokens = 4  # type: ignore[misc]
