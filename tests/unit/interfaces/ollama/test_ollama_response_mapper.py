from __future__ import annotations

from llm_proxy.domain.content import TextContent, ToolCallContent
from llm_proxy.domain.messages import Message, MessageRole
from llm_proxy.domain.responses import CompletionResponse, FinishReason, Usage
from llm_proxy.domain.events import ResponseCompleted
from llm_proxy.configuration.models import InterfaceName
from llm_proxy.application.semantic_support import SemanticSupportResolver
from llm_proxy.interfaces.ollama.response_mapper import OllamaResponseMapper


def handling(reason: FinishReason):
    return SemanticSupportResolver().resolve_response(ResponseCompleted(reason, None, Usage(0, 0)), InterfaceName.OLLAMA)


def test_projects_chat_and_generate_shapes_with_truthful_counts() -> None:
    response = CompletionResponse("id", "upstream", Message(MessageRole.ASSISTANT, (TextContent("hello"), ToolCallContent("call", "read", {"path": "README"}))), FinishReason.TOOL_USE, None, Usage(2, 3), semantic_handling=handling(FinishReason.TOOL_USE))
    mapper = OllamaResponseMapper()

    chat = mapper.map_chat(response, response_model="alias")
    generate = mapper.map_generate(response, response_model="alias")

    assert chat["message"]["tool_calls"][0]["function"]["arguments"] == {"path": "README"}
    assert chat["done_reason"] == "tool_calls"
    assert chat["prompt_eval_count"] == 2
    assert generate["response"] == "hello"
    assert generate["eval_count"] == 3
