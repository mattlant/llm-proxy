from __future__ import annotations

from llm_proxy.domain.content import ToolCallContent, ToolResultContent
from llm_proxy.interfaces.ollama.dto import OllamaChatRequest, OllamaGenerateRequest
from llm_proxy.interfaces.ollama.request_mapper import OllamaRequestMapper


def test_maps_native_chat_tools_options_and_tool_result() -> None:
    source = OllamaChatRequest.model_validate({
        "model": "alias", "messages": [
            {"role": "assistant", "tool_calls": [{"id": "call-1", "function": {"name": "read", "arguments": {"path": "README"}}}]},
            {"role": "tool", "tool_call_id": "call-1", "content": "result"},
        ],
        "tools": [{"type": "function", "function": {"name": "read", "parameters": {"type": "object"}}}],
        "stream": False, "options": {"num_predict": 20, "stop": ["DONE"], "top_k": 40},
    })

    result = OllamaRequestMapper().map_chat(source)

    assert isinstance(result.messages[0].content[0], ToolCallContent)
    assert result.messages[0].content[0].arguments == {"path": "README"}
    assert isinstance(result.messages[1].content[0], ToolResultContent)
    assert result.parameters.max_tokens == 20
    assert result.parameters.stop_sequences == ("DONE",)
    assert result.parameters.top_k == 40


def test_maps_generate_system_and_prompt_to_canonical_messages() -> None:
    result = OllamaRequestMapper().map_generate(OllamaGenerateRequest(model="alias", system="system", prompt="prompt", stream=False))

    assert [message.role.value for message in result.messages] == ["system", "user"]
    assert result.stream is False
