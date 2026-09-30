from __future__ import annotations

import pytest

from llm_proxy.domain.events import (
    ReasoningCompleted,
    ReasoningDelta,
    ReasoningStarted,
    ResponseCompleted,
    ResponseFailed,
    ResponseStarted,
    TextCompleted,
    TextDelta,
    TextStarted,
    ToolCallArgumentsDelta,
    ToolCallCompleted,
    ToolCallStarted,
)
from llm_proxy.domain.errors import ProviderUnavailableError
from llm_proxy.domain.responses import FinishReason, Usage


def test_text_only_event_sequence_is_semantic_and_ordered():
    events = (
        ResponseStarted("response-1", "model"),
        TextStarted("block-1"),
        TextDelta("block-1", "hello"),
        TextDelta("block-1", " world"),
        TextCompleted("block-1"),
        ResponseCompleted(FinishReason.END_TURN, None, Usage(None, 2)),
    )

    assert [type(event) for event in events] == [
        ResponseStarted,
        TextStarted,
        TextDelta,
        TextDelta,
        TextCompleted,
        ResponseCompleted,
    ]
    assert [event.text for event in events if isinstance(event, TextDelta)] == ["hello", " world"]


def test_tool_only_and_concurrent_tool_call_sequences_preserve_identity():
    events = (
        ResponseStarted("response-1", "model"),
        ToolCallStarted("block-1", "call-a", "read"),
        ToolCallStarted("block-2", "call-b", "write"),
        ToolCallArgumentsDelta("block-1", "call-a", '{"path":'),
        ToolCallArgumentsDelta("block-2", "call-b", '{"path":'),
        ToolCallArgumentsDelta("block-1", "call-a", '"/a"}'),
        ToolCallArgumentsDelta("block-2", "call-b", '"/b"}'),
        ToolCallCompleted("block-1", "call-a", {"path": "/a"}),
        ToolCallCompleted("block-2", "call-b", {"path": "/b"}),
        ResponseCompleted(FinishReason.TOOL_USE, None, Usage(4, None)),
    )

    assert [(event.block_id, event.tool_call_id) for event in events if isinstance(event, ToolCallStarted)] == [
        ("block-1", "call-a"),
        ("block-2", "call-b"),
    ]
    assert [event.fragment for event in events if isinstance(event, ToolCallArgumentsDelta)] == [
        '{"path":',
        '{"path":',
        '"/a"}',
        '"/b"}',
    ]


def test_text_followed_by_tool_and_reasoning_followed_by_visible_text_remain_distinct():
    events = (
        TextStarted("text-1"),
        TextDelta("text-1", "before tool"),
        TextCompleted("text-1"),
        ToolCallStarted("tool-1", "call-1", "lookup"),
        ToolCallArgumentsDelta("tool-1", "call-1", "{}"),
        ToolCallCompleted("tool-1", "call-1", {}),
        ReasoningStarted("reasoning-1"),
        ReasoningDelta("reasoning-1", "private thought"),
        ReasoningCompleted("reasoning-1"),
        TextStarted("text-2"),
        TextDelta("text-2", "visible answer"),
        TextCompleted("text-2"),
    )

    assert [event.text for event in events if isinstance(event, ReasoningDelta)] == ["private thought"]
    assert [event.text for event in events if isinstance(event, TextDelta)] == ["before tool", "visible answer"]


def test_response_failure_and_unknown_usage_are_representable():
    failure = ResponseFailed(ProviderUnavailableError("provider failed", provider="provider"))
    completed = ResponseCompleted(FinishReason.MAX_TOKENS, None, Usage(None, None))

    assert str(failure.error) == "provider failed"
    assert failure.error.provider == "provider"
    assert completed.usage.input_tokens is None
    assert completed.usage.output_tokens is None


@pytest.mark.parametrize(
    "factory",
    [
        lambda: TextStarted(""),
        lambda: TextDelta("block", ""),
        lambda: ToolCallStarted("block", "", "tool"),
        lambda: ToolCallArgumentsDelta("block", "call", ""),
        lambda: ReasoningDelta("", "thought"),
    ],
)
def test_events_reject_invalid_ids_and_empty_fragments(factory):
    with pytest.raises(ValueError):
        factory()


def test_completed_tool_arguments_are_defensively_copied():
    arguments = {"nested": {"value": 1}}
    event = ToolCallCompleted("block", "call", arguments)
    arguments["nested"]["value"] = 2

    assert event.arguments["nested"] == {"value": 1}
