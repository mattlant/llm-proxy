from __future__ import annotations

from llm_proxy.domain.content import (
    ReasoningContent,
    TextContent,
    ToolCallContent,
    ToolResultContent,
)
from llm_proxy.domain.messages import Message, MessageRole
from llm_proxy.domain.requests import CompletionRequest, SamplingParameters
from llm_proxy.domain.responses import CompletionResponse, FinishReason, Usage
from llm_proxy.domain.tools import ToolDefinition


def test_complete_tool_use_conversation_preserves_the_phase_proof_obligation():
    tools = (
        ToolDefinition(
            "read_file",
            "Read a file",
            {
                "type": "object",
                "properties": {"path": {"$ref": "#/$defs/Path"}},
                "$defs": {"Path": {"type": "string"}},
                "required": ["path"],
            },
        ),
        ToolDefinition(
            "write_file",
            "Write a file",
            {"type": "object", "properties": {"path": {"type": "string"}}},
        ),
    )
    messages = (
        Message(MessageRole.SYSTEM, (TextContent("system instructions"),)),
        Message(MessageRole.USER, (TextContent("user request"),)),
        Message(
            MessageRole.ASSISTANT,
            (
                TextContent("I will inspect and update the file."),
                ToolCallContent("call-a", "read_file", {"path": "/tmp/a"}),
                ToolCallContent("call-b", "write_file", {"path": "/tmp/a", "content": "new"}),
            ),
        ),
        Message(
            MessageRole.USER,
            (
                ToolResultContent("call-a", (TextContent("old contents"),)),
                ToolResultContent("call-b", (TextContent("permission denied"),), is_error=True),
            ),
        ),
        Message(
            MessageRole.ASSISTANT,
            (ReasoningContent("The write failed, so I should explain the limitation."), TextContent("I could read the file but could not write it.")),
        ),
    )
    request = CompletionRequest(
        "model",
        messages,
        tools=tools,
        parameters=SamplingParameters(extra={"provider_mode": "extended"}),
    )
    response = CompletionResponse(
        "response-1",
        "model",
        messages[-1],
        FinishReason.TOOL_USE,
        None,
        Usage(42, 19),
    )

    assert [block.text for block in request.messages[0].content if isinstance(block, TextContent)] == [
        "system instructions"
    ]
    assistant_blocks = request.messages[2].content
    assert len(assistant_blocks) == 3
    assert isinstance(assistant_blocks[0], TextContent)
    assert assistant_blocks[0].text == "I will inspect and update the file."
    assert isinstance(assistant_blocks[1], ToolCallContent)
    assert assistant_blocks[1].id == "call-a"
    assert assistant_blocks[1].name == "read_file"
    assert isinstance(assistant_blocks[2], ToolCallContent)
    assert assistant_blocks[2].id == "call-b"
    assert assistant_blocks[2].name == "write_file"
    results = request.messages[3].content
    assert [result.tool_call_id for result in results if isinstance(result, ToolResultContent)] == ["call-a", "call-b"]
    assert results[1].is_error is True  # type: ignore[union-attr]
    final_blocks = response.message.content
    assert isinstance(final_blocks[0], ReasoningContent)
    assert isinstance(final_blocks[1], TextContent)
    assert request.tools[0].input_schema["$defs"] == {"Path": {"type": "string"}}
    assert response.finish_reason is FinishReason.TOOL_USE
    assert response.usage == Usage(42, 19)
