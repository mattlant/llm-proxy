from __future__ import annotations

import pytest

from llm_proxy.domain.content import ReasoningContent, TextContent, ToolCallContent
from llm_proxy.domain.messages import Message, MessageRole
from llm_proxy.domain.responses import CompletionResponse, FinishReason, Usage
from llm_proxy.domain.requests import ContextManagementResult
from llm_proxy.domain.events import ResponseCompleted
from llm_proxy.configuration.models import InterfaceName
from llm_proxy.application.semantic_support import SemanticSupportResolver
from llm_proxy.interfaces.anthropic.response_mapper import AnthropicResponseMapper


def handling(reason: FinishReason):
    return SemanticSupportResolver().resolve_response(ResponseCompleted(reason, None, Usage(0, 0)), InterfaceName.ANTHROPIC)


def test_maps_ordered_text_tools_alias_and_unknown_usage() -> None:
    response = CompletionResponse(
        "response-1", "upstream", Message(MessageRole.ASSISTANT, (TextContent("first"), ToolCallContent("call-1", "Read", {"path": "README"}), TextContent("second"), ReasoningContent("hidden"))),
        FinishReason.TOOL_USE, None, Usage(None, None), semantic_handling=handling(FinishReason.TOOL_USE),
    )

    result = AnthropicResponseMapper().map_response(response, response_model="local-opus")

    assert result["id"] == "msg_response-1"
    assert result["model"] == "local-opus"
    assert result["content"] == [
        {"type": "text", "text": "first"},
        {"type": "tool_use", "id": "call-1", "name": "Read", "input": {"path": "README"}},
        {"type": "text", "text": "second"},
    ]
    assert result["usage"] == {"input_tokens": 0, "output_tokens": 0}


@pytest.mark.parametrize("reason", list(FinishReason))
def test_maps_all_canonical_stop_reasons(reason: FinishReason) -> None:
    response = CompletionResponse("msg_existing", "upstream", Message(MessageRole.ASSISTANT, (TextContent("done"),)), reason, "STOP" if reason is FinishReason.STOP_SEQUENCE else None, Usage(1, 2), semantic_handling=handling(reason))

    result = AnthropicResponseMapper().map_response(response, response_model="alias")

    assert result["id"] == "msg_existing"
    assert result["stop_reason"] == reason.value
    assert result["stop_sequence"] == ("STOP" if reason is FinishReason.STOP_SEQUENCE else None)


def test_projects_requested_context_management_with_truthful_empty_result() -> None:
    response = CompletionResponse(
        "response-1", "upstream", Message(MessageRole.ASSISTANT, (TextContent("done"),)),
        FinishReason.END_TURN, None, Usage(1, 2), ContextManagementResult(), handling(FinishReason.END_TURN),
    )

    result = AnthropicResponseMapper().map_response(response, response_model="local-opus")

    assert result["context_management"] == {"applied_edits": []}
