from __future__ import annotations

from llm_proxy.domain.content import ReasoningContent, TextContent, ToolCallContent
from llm_proxy.domain.messages import Message, MessageRole
from llm_proxy.domain.responses import CompletionResponse, FinishReason, Usage
from llm_proxy.domain.events import ResponseCompleted
from llm_proxy.configuration.models import InterfaceName
from llm_proxy.application.semantic_support import SemanticSupportResolver
from llm_proxy.interfaces.openai.response_mapper import OpenAIResponseMapper


def handling(reason: FinishReason):
    return SemanticSupportResolver().resolve_response(ResponseCompleted(reason, None, Usage(0, 0)), InterfaceName.OPENAI)


def test_projects_text_tools_and_truthful_usage() -> None:
    response = CompletionResponse(
        "response-1", "upstream", Message(MessageRole.ASSISTANT, (TextContent("hello"), ReasoningContent("hidden"), ToolCallContent("call-1", "read", {"path": "README"}))),
        FinishReason.TOOL_USE, None, Usage(4, 6), semantic_handling=handling(FinishReason.TOOL_USE),
    )

    result = OpenAIResponseMapper().map_response(response, response_model="alias")

    assert result["id"] == "chatcmpl-response-1"
    assert result["model"] == "alias"
    assert result["choices"][0]["message"]["content"] == "hello"
    assert result["choices"][0]["message"]["reasoning_content"] == "hidden"
    assert result["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"] == '{"path":"README"}'
    assert result["choices"][0]["finish_reason"] == "tool_calls"
    assert result["usage"] == {"prompt_tokens": 4, "completion_tokens": 6, "total_tokens": 10}
