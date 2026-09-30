from __future__ import annotations

import pytest

from llm_proxy.domain.content import ReasoningContent, TextContent, ToolCallContent
from llm_proxy.domain.errors import InvalidToolArgumentsError, ProviderProtocolError
from llm_proxy.domain.responses import FinishReason
from llm_proxy.providers.ollama.response_mapper import OllamaOpenAIResponseMapper


@pytest.fixture
def mapper() -> OllamaOpenAIResponseMapper:
    return OllamaOpenAIResponseMapper()


def response_payload(**overrides: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "id": "chatcmpl-123",
        "model": "hf.co/qwable:Q4_K_M",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": "I will read it."},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 100, "completion_tokens": 25, "total_tokens": 125},
    }
    payload.update(overrides)
    return payload


def test_maps_text_response_and_usage(mapper: OllamaOpenAIResponseMapper) -> None:
    result = mapper.map_response(response_payload())

    assert result.id == "chatcmpl-123"
    assert result.model == "hf.co/qwable:Q4_K_M"
    assert result.message.content == (TextContent("I will read it."),)
    assert result.finish_reason is FinishReason.END_TURN
    assert result.usage.input_tokens == 100
    assert result.usage.output_tokens == 25


def test_maps_text_and_multiple_tool_calls_in_upstream_order(mapper: OllamaOpenAIResponseMapper) -> None:
    payload = response_payload(
        choices=[
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "I will use tools.",
                    "tool_calls": [
                        {"id": "call_a", "type": "function", "function": {"name": "Read", "arguments": '{"path":"README.md"}'}},
                        {"id": "call_b", "type": "function", "function": {"name": "Stat", "arguments": "{}"}},
                    ],
                },
                "finish_reason": "tool_calls",
            }
        ]
    )

    result = mapper.map_response(payload)

    assert isinstance(result.message.content[0], TextContent)
    assert result.message.content[0].text == "I will use tools."
    assert isinstance(result.message.content[1], ToolCallContent)
    assert isinstance(result.message.content[2], ToolCallContent)
    assert result.message.content[1].id == "call_a"
    assert result.message.content[2].name == "Stat"
    assert result.message.content[1].arguments == {"path": "README.md"}
    assert result.finish_reason is FinishReason.TOOL_USE


def test_generates_stable_ids_for_missing_response_and_tool_ids(mapper: OllamaOpenAIResponseMapper) -> None:
    payload = response_payload(
        id=None,
        choices=[
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [{"function": {"name": "Read", "arguments": "{}"}}],
                },
                "finish_reason": "tool_calls",
            }
        ],
    )

    first = mapper.map_response(payload)
    second = mapper.map_response(payload)

    assert first.id == second.id
    assert first.id.startswith("chatcmpl_")
    assert first.message.content[0].id == second.message.content[0].id
    assert first.message.content[0].id.startswith("call_")


def test_missing_usage_is_honest(mapper: OllamaOpenAIResponseMapper) -> None:
    result = mapper.map_response(response_payload(usage=None))

    assert result.usage.input_tokens is None
    assert result.usage.output_tokens is None


@pytest.mark.parametrize(
    ("upstream", "expected"),
    [
        ("stop", FinishReason.END_TURN),
        ("length", FinishReason.MAX_TOKENS),
        ("tool_calls", FinishReason.TOOL_USE),
        ("content_filter", FinishReason.REFUSAL),
        ("refusal", FinishReason.REFUSAL),
    ],
)
def test_maps_known_finish_reasons(
    mapper: OllamaOpenAIResponseMapper,
    upstream: str,
    expected: FinishReason,
) -> None:
    result = mapper.map_response(
        response_payload(choices=[{"index": 0, "message": {"role": "assistant", "content": "ok"}, "finish_reason": upstream}])
    )

    assert result.finish_reason is expected


def test_reasoning_fields_are_not_merged_into_visible_text(mapper: OllamaOpenAIResponseMapper) -> None:
    result = mapper.map_response(
        response_payload(
            choices=[
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "visible", "reasoning_content": "private"},
                    "finish_reason": "stop",
                }
            ]
        )
    )

    assert result.message.content == (TextContent("visible"),)


def test_reasoning_is_preserved_only_when_exposed(mapper: OllamaOpenAIResponseMapper) -> None:
    payload = response_payload(
        choices=[
            {
                "index": 0,
                "message": {"role": "assistant", "content": "visible", "reasoning_content": "private"},
                "finish_reason": "stop",
            }
        ]
    )

    hidden = mapper.map_response(payload, expose_thinking=False)
    exposed = mapper.map_response(payload, expose_thinking=True)

    assert hidden.message.content == (TextContent("visible"),)
    assert exposed.message.content == (ReasoningContent("private"), TextContent("visible"))


def test_thinking_has_precedence_over_legacy_reasoning_aliases(mapper: OllamaOpenAIResponseMapper) -> None:
    result = mapper.map_response(response_payload(choices=[{"index": 0, "message": {"role": "assistant", "content": "visible", "thinking": "native", "reasoning": "legacy", "reasoning_content": "older"}, "finish_reason": "stop"}]), expose_thinking=True)
    assert result.message.content == (ReasoningContent("native"), TextContent("visible"))


def test_literal_think_markup_remains_visible_text(mapper: OllamaOpenAIResponseMapper) -> None:
    result = mapper.map_response(
        response_payload(
            choices=[
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "<think>literal</think>"},
                    "finish_reason": "stop",
                }
            ]
        ),
        expose_thinking=False,
    )

    assert result.message.content == (TextContent("<think>literal</think>"),)


@pytest.mark.parametrize("arguments", ["not-json", "[]", "1", "true", "null", '"text"'])
def test_rejects_malformed_or_non_object_tool_arguments(
    mapper: OllamaOpenAIResponseMapper,
    arguments: str,
) -> None:
    payload = response_payload(
        choices=[
            {
                "index": 0,
                "message": {"role": "assistant", "content": None, "tool_calls": [{"id": "call", "function": {"name": "Read", "arguments": arguments}}]},
                "finish_reason": "tool_calls",
            }
        ]
    )

    with pytest.raises(InvalidToolArgumentsError):
        mapper.map_response(payload)


def test_rejects_empty_or_multiple_choices(mapper: OllamaOpenAIResponseMapper) -> None:
    with pytest.raises(ProviderProtocolError, match="one choice"):
        mapper.map_response(response_payload(choices=[]))
    with pytest.raises(ProviderProtocolError, match="multiple"):
        mapper.map_response(response_payload(choices=[{}, {}]))


def test_rejects_unknown_finish_reason(mapper: OllamaOpenAIResponseMapper) -> None:
    with pytest.raises(ProviderProtocolError, match="unknown"):
        mapper.map_response(response_payload(choices=[{"index": 0, "message": {"content": "ok"}, "finish_reason": "mystery"}]))
