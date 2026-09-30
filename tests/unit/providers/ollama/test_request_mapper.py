from __future__ import annotations

import pytest

from llm_proxy.provider_extensions import CompletionExecution, ProviderExecutionPolicy
from llm_proxy.domain.content import ReasoningContent, TextContent, ToolCallContent, ToolResultContent
from llm_proxy.domain.messages import Message, MessageRole
from llm_proxy.domain.messages import DeveloperRoleMode
from llm_proxy.domain.errors import DeveloperRoleCompatibilityError
from llm_proxy.domain.requests import AutomaticToolChoice, NamedToolChoice, NoToolChoice, RequiredToolChoice, CompletionRequest, SamplingParameters
from llm_proxy.domain.tools import ToolDefinition
from llm_proxy.providers.ollama.request_mapper import OllamaOpenAIRequestMapper


@pytest.fixture
def mapper() -> OllamaOpenAIRequestMapper:
    return OllamaOpenAIRequestMapper()


def execution(request: CompletionRequest, parameters: SamplingParameters) -> CompletionExecution:
    return CompletionExecution(
        request=request,
        upstream_model="hf.co/lordx64/Qwable-v2-GGUF:Q4_K_M",
        provider_instance_name="rtx-3090",
        parameters=parameters,
        response_model=request.model,
    )


def test_maps_complete_request_from_upstream_resolution_and_final_parameters(
    mapper: OllamaOpenAIRequestMapper,
) -> None:
    request = CompletionRequest(
        model="qwable-alias",
        messages=(Message(MessageRole.USER, (TextContent("Read README."),)),),
        tools=(ToolDefinition("Read", "Read a file", {"type": "object"}),),
        tool_choice=AutomaticToolChoice(),
        stream=True,
        metadata={"trace_id": "secret-to-client-only"},
    )
    parameters = SamplingParameters(
        temperature=0.0,
        top_p=0.95,
        top_k=40,
        min_p=0.1,
        repeat_penalty=1.1,
        repeat_last_n=64,
        max_tokens=16384,
        stop_sequences=("DONE",),
        extra={"num_ctx": 32768},
    )
    mapped = mapper.map_request(execution(request, parameters))

    assert mapped["model"] == "hf.co/lordx64/Qwable-v2-GGUF:Q4_K_M"
    assert mapped["messages"] == [{"role": "user", "content": "Read README."}]
    assert mapped["stream"] is True
    assert mapped["stream_options"] == {"include_usage": True}
    assert mapped["temperature"] == 0.0
    assert mapped["top_p"] == 0.95
    assert mapped["top_k"] == 40
    assert mapped["min_p"] == 0.1
    assert mapped["repeat_penalty"] == 1.1
    assert mapped["repeat_last_n"] == 64
    assert mapped["max_tokens"] == 16384
    assert mapped["stop"] == ["DONE"]
    assert mapped["num_ctx"] == 32768
    assert "qwable-alias" not in str(mapped)
    assert "trace_id" not in mapped


def test_omits_unset_parameters_and_stream_options_for_non_streaming(
    mapper: OllamaOpenAIRequestMapper,
) -> None:
    request = CompletionRequest(
        model="qwable",
        messages=(Message(MessageRole.USER, (TextContent("Hello"),)),),
    )
    parameters = SamplingParameters(temperature=0.0, top_k=0, max_tokens=0)

    mapped = mapper.map_request(execution(request, parameters))

    assert mapped == {
        "model": "hf.co/lordx64/Qwable-v2-GGUF:Q4_K_M",
        "messages": [{"role": "user", "content": "Hello"}],
        "stream": False,
        "temperature": 0.0,
        "top_k": 0,
        "max_tokens": 0,
    }


def test_enables_upstream_thinking_only_when_policy_exposes_it(mapper: OllamaOpenAIRequestMapper) -> None:
    request = CompletionRequest("qwable", (Message(MessageRole.USER, (TextContent("Hello"),)),))
    normal = execution(request, SamplingParameters())
    exposed = CompletionExecution(request, normal.upstream_model, normal.provider_instance_name, normal.parameters, normal.response_model, policy=ProviderExecutionPolicy(expose_thinking=True))
    assert "think" not in mapper.map_request(normal)
    assert mapper.map_request(exposed)["think"] is True


def test_rejects_extra_parameter_collisions(mapper: OllamaOpenAIRequestMapper) -> None:
    request = CompletionRequest("qwable", (Message(MessageRole.USER, (TextContent("Hello"),)),))
    parameters = SamplingParameters(extra={"temperature": 0.5})

    with pytest.raises(ValueError, match="temperature"):
        mapper.map_request(execution(request, parameters))


def test_maps_nested_tool_schema_without_mutation(mapper: OllamaOpenAIRequestMapper) -> None:
    schema = {
        "type": "object",
        "properties": {
            "path": {"type": "string", "enum": ["README.md"], "default": "README.md"},
        },
        "required": ["path"],
        "additionalProperties": False,
        "$defs": {"Path": {"type": "string"}},
        "$ref": "#/$defs/Path",
    }
    tool = ToolDefinition("Read", "Read a file", schema)

    mapped = mapper.map_tools((tool,))

    assert mapped == [{"type": "function", "function": {"name": "Read", "description": "Read a file", "parameters": schema}}]
    assert mapped[0]["function"]["parameters"] is not tool.input_schema
    assert mapped[0]["function"]["parameters"]["$defs"] is not tool.input_schema["$defs"]
    assert dict(tool.input_schema) == schema


@pytest.mark.parametrize(
    ("choice", "expected"),
    [
        (AutomaticToolChoice(), "auto"),
        (RequiredToolChoice(), "required"),
        (NoToolChoice(), "none"),
        (NamedToolChoice("Read"), {"type": "function", "function": {"name": "Read"}}),
    ],
)
def test_maps_tool_choice_modes(
    mapper: OllamaOpenAIRequestMapper,
    choice: object,
    expected: object,
) -> None:
    tools = (ToolDefinition("Read", "Read a file", {"type": "object"}),)

    assert mapper.map_tool_choice(choice, tools) == expected  # type: ignore[arg-type]


def test_rejects_named_choice_for_unknown_tool(mapper: OllamaOpenAIRequestMapper) -> None:
    with pytest.raises(ValueError, match="not defined"):
        mapper.map_tool_choice(NamedToolChoice("Write"), (ToolDefinition("Read", "", {}),))


def test_maps_system_and_user_text_in_order(mapper: OllamaOpenAIRequestMapper) -> None:
    messages = (
        Message(MessageRole.SYSTEM, (TextContent("System one"), TextContent("System two"))),
        Message(MessageRole.USER, (TextContent("Read README."),)),
    )

    assert mapper.map_messages(messages) == [
        {"role": "system", "content": "System one\nSystem two"},
        {"role": "user", "content": "Read README."},
    ]


def test_maps_developer_messages_by_explicit_mode(mapper: OllamaOpenAIRequestMapper) -> None:
    messages = (
        Message(MessageRole.SYSTEM, (TextContent("system"),)),
        Message(MessageRole.DEVELOPER, (TextContent("developer one"),)),
        Message(MessageRole.USER, (TextContent("user"),)),
        Message(MessageRole.DEVELOPER, (TextContent("developer two"),)),
    )
    assert mapper.map_messages(messages, DeveloperRoleMode.PRESERVE) == [
        {"role": "system", "content": "system"},
        {"role": "developer", "content": "developer one"},
        {"role": "user", "content": "user"},
        {"role": "developer", "content": "developer two"},
    ]
    assert mapper.map_messages(messages, DeveloperRoleMode.SYSTEM) == [
        {"role": "system", "content": "system"},
        {"role": "system", "content": "developer one"},
        {"role": "user", "content": "user"},
        {"role": "system", "content": "developer two"},
    ]
    with pytest.raises(DeveloperRoleCompatibilityError):
        mapper.map_messages(messages, DeveloperRoleMode.REJECT)


def test_maps_assistant_text_and_tool_calls_without_reasoning(mapper: OllamaOpenAIRequestMapper) -> None:
    message = Message(
        MessageRole.ASSISTANT,
        (
            TextContent("I will read it."),
            ReasoningContent("private reasoning"),
            ToolCallContent("call_a", "Read", {"file_path": "README.md", "line": 1}),
            TextContent("Done selecting the tool."),
            ToolCallContent("call_b", "Stat", {}),
        ),
    )

    assert mapper.map_messages((message,)) == [
        {
            "role": "assistant",
            "content": "I will read it.\nDone selecting the tool.",
            "tool_calls": [
                {
                    "id": "call_a",
                    "type": "function",
                    "function": {"name": "Read", "arguments": '{"file_path":"README.md","line":1}'},
                },
                {
                    "id": "call_b",
                    "type": "function",
                    "function": {"name": "Stat", "arguments": "{}"},
                },
            ],
        }
    ]


def test_maps_tool_results_and_expands_multiple_results(mapper: OllamaOpenAIRequestMapper) -> None:
    message = Message(
        MessageRole.TOOL,
        (
            ToolResultContent("call_a", (TextContent("File contents"),)),
            ToolResultContent("call_b", (TextContent("Permission denied"),), is_error=True),
        ),
    )

    assert mapper.map_messages((message,)) == [
        {"role": "tool", "tool_call_id": "call_a", "content": "File contents"},
        {"role": "tool", "tool_call_id": "call_b", "content": "[tool_error]\nPermission denied"},
    ]


def test_reasoning_only_assistant_message_is_rejected(mapper: OllamaOpenAIRequestMapper) -> None:
    with pytest.raises(ValueError, match="no mappable content"):
        mapper.map_messages((Message(MessageRole.ASSISTANT, (ReasoningContent("hidden"),)),))


@pytest.mark.parametrize(
    "message",
    [
        Message(MessageRole.SYSTEM, (ToolCallContent("call", "Read", {}),)),
        Message(MessageRole.USER, (ToolResultContent("call", (TextContent("result"),)),)),
        Message(MessageRole.ASSISTANT, (ToolResultContent("call", (TextContent("result"),)),)),
        Message(MessageRole.TOOL, (TextContent("not a tool result"),)),
    ],
)
def test_rejects_invalid_role_and_content_combinations(
    mapper: OllamaOpenAIRequestMapper,
    message: Message,
) -> None:
    with pytest.raises(ValueError):
        mapper.map_messages((message,))
