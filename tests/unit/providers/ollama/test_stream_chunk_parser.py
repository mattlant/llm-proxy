from __future__ import annotations

import pytest

from llm_proxy.domain.errors import ProviderProtocolError
from llm_proxy.domain.events import (
    ReasoningCompleted,
    ReasoningDelta,
    ReasoningStarted,
    ResponseCompleted,
    ResponseStarted,
    TextCompleted,
    TextDelta,
    TextStarted,
    ToolCallArgumentsDelta,
    ToolCallCompleted,
    ToolCallStarted,
)
from llm_proxy.domain.errors import InvalidToolArgumentsError
from llm_proxy.domain.responses import FinishReason
from llm_proxy.providers.ollama.stream_chunk_parser import OpenAIStreamParser
from llm_proxy.providers.ollama.stream_parser import SseRecord


def record(data: str) -> SseRecord:
    return SseRecord(None, data)


def test_maps_one_text_chunk_to_complete_lifecycle() -> None:
    parser = OpenAIStreamParser()

    events = parser.feed(record('{"id":"chatcmpl-1","model":"qwable","choices":[{"delta":{"role":"assistant","content":"hello"},"finish_reason":null}]}'))
    events += parser.feed(record('{"choices":[{"delta":{"content":null},"finish_reason":"stop"}]}'))

    assert events == (
        ResponseStarted("chatcmpl-1", "qwable"),
        TextStarted("text_0"),
        TextDelta("text_0", "hello"),
        TextCompleted("text_0"),
        ResponseCompleted(FinishReason.END_TURN, None, usage(events[-1])),
    )


def usage(event: ResponseCompleted):
    return event.usage


def test_maps_text_over_many_chunks_and_final_usage() -> None:
    parser = OpenAIStreamParser()

    events = []
    events.extend(parser.feed(record('{"id":"id","model":"model","choices":[{"delta":{"role":"assistant"},"finish_reason":null}]}')))
    events.extend(parser.feed(record('{"choices":[{"delta":{"content":"Hel"},"finish_reason":null}]}')))
    events.extend(parser.feed(record('{"choices":[{"delta":{"content":"lo"},"finish_reason":null}]}')))
    events.extend(parser.feed(record('{"choices":[],"usage":{"prompt_tokens":7,"completion_tokens":2}}')))
    events.extend(parser.feed(record('{"choices":[{"delta":{},"finish_reason":"length"}]}')))

    assert events[0] == ResponseStarted("id", "model")
    assert events[1:4] == [TextStarted("text_0"), TextDelta("text_0", "Hel"), TextDelta("text_0", "lo")]
    assert events[-2:] == [TextCompleted("text_0"), ResponseCompleted(FinishReason.MAX_TOKENS, None, events[-1].usage)]
    assert events[-1].usage.input_tokens == 7
    assert events[-1].usage.output_tokens == 2


def test_role_only_chunk_starts_response_without_text() -> None:
    parser = OpenAIStreamParser()

    assert parser.feed(record('{"id":"id","model":"model","choices":[{"delta":{"role":"assistant"},"finish_reason":null}]}')) == (
        ResponseStarted("id", "model"),
    )


def test_done_and_clean_eof_complete_without_finish_reason() -> None:
    parser = OpenAIStreamParser()
    parser.feed(record('{"id":"id","model":"model","choices":[{"delta":{"content":"hello"},"finish_reason":null}]}'))

    done_events = parser.feed(record("[DONE]"))
    assert isinstance(done_events[-1], ResponseCompleted)
    assert done_events[-1].finish_reason is FinishReason.END_TURN

    clean = OpenAIStreamParser()
    clean.feed(record('{"id":"id","model":"model","choices":[{"delta":{"content":"hello"},"finish_reason":null}]}'))
    assert clean.finish()[-1].finish_reason is FinishReason.END_TURN


def test_accepts_trailing_usage_and_done_after_finish_reason() -> None:
    parser = OpenAIStreamParser()

    events = parser.feed(record('{"id":"id","model":"model","choices":[{"delta":{"content":"hello"},"finish_reason":null}]}'))
    events += parser.feed(record('{"choices":[{"delta":{},"finish_reason":"stop"}]}'))
    completion = events[-1]
    assert isinstance(completion, ResponseCompleted)

    assert parser.feed(record('{"choices":[],"usage":{"prompt_tokens":7,"completion_tokens":2}}')) == ()
    assert parser.feed(record("[DONE]")) == ()
    assert completion.usage.input_tokens == 7
    assert completion.usage.output_tokens == 2


def test_clean_eof_after_finish_reason_is_idempotent() -> None:
    parser = OpenAIStreamParser()
    parser.feed(record('{"id":"id","model":"model","choices":[{"delta":{},"finish_reason":"stop"}]}'))

    assert parser.finish() == ()


def test_rejects_duplicate_finish_and_non_usage_chunk_after_completion() -> None:
    parser = OpenAIStreamParser()
    parser.feed(record('{"id":"id","model":"model","choices":[{"delta":{},"finish_reason":"stop"}]}'))

    with pytest.raises(ProviderProtocolError, match="after completion"):
        parser.feed(record('{"choices":[{"delta":{},"finish_reason":"stop"}]}'))

    completed = OpenAIStreamParser()
    completed.feed(record('{"id":"id","model":"model","choices":[{"delta":{},"finish_reason":"stop"}]}'))
    with pytest.raises(ProviderProtocolError, match="after completion"):
        completed.feed(record('{"choices":[{"delta":{"content":"late"},"finish_reason":null}]}'))


@pytest.mark.parametrize(
    "payload",
    [
        '{"choices":[]}',
        '{"choices":[],"usage":null}',
        '{"choices":[],"usage":[]}',
    ],
)
def test_rejects_invalid_trailing_usage_after_completion(payload: str) -> None:
    parser = OpenAIStreamParser()
    parser.feed(record('{"id":"id","model":"model","choices":[{"delta":{},"finish_reason":"stop"}]}'))

    with pytest.raises(ProviderProtocolError, match="after completion"):
        parser.feed(record(payload))


def test_missing_response_id_is_deterministic() -> None:
    payload = record('{"model":"model","choices":[{"delta":{"role":"assistant"},"finish_reason":null}]}')
    first = OpenAIStreamParser().feed(payload)[0]
    second = OpenAIStreamParser().feed(payload)[0]

    assert isinstance(first, ResponseStarted)
    assert isinstance(second, ResponseStarted)
    assert first.response_id == second.response_id


def test_maps_tool_only_lifecycle_and_concatenates_arguments_before_parsing() -> None:
    parser = OpenAIStreamParser()
    events = []
    events.extend(parser.feed(record('{"id":"id","model":"model","choices":[{"delta":{"role":"assistant","tool_calls":[{"index":0,"id":"call-1","function":{"name":"Read"}}]},"finish_reason":null}]}')))
    events.extend(parser.feed(record('{"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":"{\\"path\\":"}}]},"finish_reason":null}]}')))
    events.extend(parser.feed(record('{"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":"\\"README.md\\"}"}}]},"finish_reason":"tool_calls"}]}')))

    assert events[0] == ResponseStarted("id", "model")
    assert events[1] == ToolCallStarted("tool_0", "call-1", "Read")
    assert events[2] == ToolCallArgumentsDelta("tool_0", "call-1", '{"path":')
    assert events[3] == ToolCallArgumentsDelta("tool_0", "call-1", '"README.md"}')
    assert events[4] == ToolCallCompleted("tool_0", "call-1", {"path": "README.md"})
    assert events[5].finish_reason is FinishReason.TOOL_USE


def test_maps_parallel_tools_in_index_order_at_completion() -> None:
    parser = OpenAIStreamParser()
    parser.feed(record('{"id":"id","model":"model","choices":[{"delta":{"tool_calls":[{"index":1,"id":"b","function":{"name":"Write","arguments":"{}"}},{"index":0,"id":"a","function":{"name":"Read","arguments":"{}"}}]},"finish_reason":null}]}'))

    events = parser.feed(record('{"choices":[{"delta":{},"finish_reason":"tool_calls"}]}'))

    assert [event.tool_call_id for event in events if isinstance(event, ToolCallCompleted)] == ["a", "b"]


def test_text_then_tool_closes_text_before_tool_completion() -> None:
    parser = OpenAIStreamParser()
    parser.feed(record('{"id":"id","model":"model","choices":[{"delta":{"content":"before"},"finish_reason":null}]}'))
    parser.feed(record('{"choices":[{"delta":{"tool_calls":[{"index":0,"id":"a","function":{"name":"Read","arguments":"{}"}}]},"finish_reason":null}]}'))

    events = parser.feed(record('{"choices":[{"delta":{},"finish_reason":"tool_calls"}]}'))

    assert isinstance(events[0], TextCompleted)
    assert isinstance(events[-2], ToolCallCompleted)


@pytest.mark.parametrize("arguments", ["not-json", "[]"])
def test_rejects_malformed_or_non_object_tool_arguments(arguments: str) -> None:
    parser = OpenAIStreamParser()
    parser.feed(record('{"id":"id","model":"model","choices":[{"delta":{"tool_calls":[{"index":0,"id":"a","function":{"name":"Read","arguments":"' + arguments + '"}}]},"finish_reason":null}]}'))

    with pytest.raises(InvalidToolArgumentsError):
        parser.feed(record('{"choices":[{"delta":{},"finish_reason":"tool_calls"}]}'))


def test_reasoning_is_suppressed_when_thinking_is_hidden() -> None:
    parser = OpenAIStreamParser(expose_thinking=False)

    events = parser.feed(record('{"id":"id","model":"model","choices":[{"delta":{"reasoning_content":"private","content":"visible"},"finish_reason":"stop"}]}'))

    assert all(not isinstance(event, (ReasoningStarted, ReasoningDelta, ReasoningCompleted)) for event in events)
    assert any(isinstance(event, TextDelta) and event.text == "visible" for event in events)


def test_reasoning_is_separate_and_closes_before_response_completion_when_exposed() -> None:
    parser = OpenAIStreamParser(expose_thinking=True)
    events = parser.feed(record('{"id":"id","model":"model","choices":[{"delta":{"reasoning":"think ","content":"visible"},"finish_reason":null}]}'))
    events += parser.feed(record('{"choices":[{"delta":{"reasoning_content":"more"},"finish_reason":"stop"}]}'))

    assert events[1:5] == (
        TextStarted("text_0"),
        TextDelta("text_0", "visible"),
        ReasoningStarted("reasoning_0"),
        ReasoningDelta("reasoning_0", "think "),
    )
    assert isinstance(events[-3], TextCompleted)
    assert events[-2] == ReasoningCompleted("reasoning_0")
    assert isinstance(events[-1], ResponseCompleted)


def test_native_thinking_has_precedence_over_legacy_aliases() -> None:
    parser = OpenAIStreamParser(expose_thinking=True)
    events = parser.feed(record('{"id":"id","model":"model","choices":[{"delta":{"thinking":"native","reasoning":"legacy"},"finish_reason":"stop"}]}'))
    assert ReasoningDelta("reasoning_0", "native") in events
    assert ReasoningDelta("reasoning_0", "legacy") not in events
