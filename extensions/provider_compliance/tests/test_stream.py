from __future__ import annotations

import pytest

from llm_proxy.provider_extensions import (
    FinishReason, ResponseCompleted, ResponseStarted, TextCompleted, TextDelta,
    TextStarted, ToolCallArgumentsDelta, ToolCallCompleted, ToolCallStarted, Usage,
)

from llm_proxy_provider_compliance import ProviderComplianceViolation, observe_stream


async def _events(*events):
    for event in events:
        yield event


async def test_observe_stream_accepts_canonical_text_lifecycle() -> None:
    events = await observe_stream("sample", _events(
        ResponseStarted("id", "model"), TextStarted("text"), TextDelta("text", "hello"),
        TextCompleted("text"), ResponseCompleted(FinishReason.END_TURN, None, Usage(1, 2)),
    ))
    assert len(events) == 5


async def test_observe_stream_accepts_fragmented_tool_lifecycle() -> None:
    events = await observe_stream("sample", _events(
        ResponseStarted("id", "model"), ToolCallStarted("tool", "call", "weather"),
        ToolCallArgumentsDelta("tool", "call", '{"city":'), ToolCallArgumentsDelta("tool", "call", '"Toronto"}'),
        ToolCallCompleted("tool", "call", {"city": "Toronto"}),
        ResponseCompleted(FinishReason.TOOL_USE, None, Usage(1, 2)),
    ))
    assert len(events) == 6


@pytest.mark.parametrize(
    "events, match",
    [
        ((_events(TextStarted("text")),), "before ResponseStarted"),
        ((_events(ResponseStarted("id", "model"), TextDelta("text", "x")),), "without open text block"),
        ((_events(ResponseStarted("id", "model"), ResponseCompleted(FinishReason.END_TURN, None, Usage(1, 2)), TextStarted("text")),), "event after ResponseCompleted"),
        ((_events(ResponseStarted("id", "model"), TextStarted("text")),), "missing ResponseCompleted"),
        ((_events(ResponseStarted("id", "model"), ToolCallStarted("tool", "one", "weather"), ToolCallArgumentsDelta("tool", "two", "{}")),), "tool call id differs"),
    ],
)
async def test_observe_stream_reports_capability_specific_violations(events, match: str) -> None:
    with pytest.raises(ProviderComplianceViolation, match=match):
        await observe_stream("broken-stream", events[0])
