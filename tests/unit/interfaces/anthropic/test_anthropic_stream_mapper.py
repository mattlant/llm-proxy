from __future__ import annotations

import pytest

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
    ResponseFailed,
)
from llm_proxy.domain.errors import ProviderUnavailableError
from llm_proxy.domain.responses import FinishReason, Usage
from llm_proxy.application.semantic_support import SemanticSupportResolver
from llm_proxy.configuration.models import InterfaceName


def completed(reason: FinishReason, usage: Usage, context_management=None):
    from llm_proxy.domain.events import ResponseCompleted
    event = ResponseCompleted(reason, None, usage, context_management)
    return ResponseCompleted(reason, None, usage, context_management, SemanticSupportResolver().resolve_response(event, InterfaceName.ANTHROPIC))
from llm_proxy.domain.requests import ContextManagementResult
from llm_proxy.interfaces.anthropic.stream_mapper import serialize_sse
from llm_proxy.interfaces.anthropic.stream_mapper import AnthropicStreamMapper


def test_serializes_compact_deterministic_utf8_sse() -> None:
    result = serialize_sse("content_block_delta", {"type": "content_block_delta", "delta": {"text": "héllo"}, "index": 0})

    assert result == 'event: content_block_delta\ndata: {"delta":{"text":"héllo"},"index":0,"type":"content_block_delta"}\n\n'.encode()


@pytest.mark.parametrize("event", ["message_start", "content_block_start", "content_block_stop", "message_delta", "message_stop", "ping", "error"])
def test_supports_each_anthropic_sse_event(event: str) -> None:
    assert serialize_sse(event, {"type": event}).startswith(f"event: {event}\n".encode())


def test_rejects_unknown_event_names() -> None:
    with pytest.raises(ValueError, match="unsupported"):
        serialize_sse("provider_chunk", {})


async def test_projects_text_and_tool_events_to_anthropic_transcript() -> None:
    async def events():
        yield ResponseStarted("response-1", "upstream")
        yield TextStarted("text-1")
        yield TextDelta("text-1", "héllo")
        yield TextCompleted("text-1")
        yield ToolCallStarted("tool-1", "call-1", "Read")
        yield ToolCallArgumentsDelta("tool-1", "call-1", '{"path":')
        yield ToolCallArgumentsDelta("tool-1", "call-1", '"README"}')
        yield ToolCallCompleted("tool-1", "call-1", {"path": "README"})
        yield completed(FinishReason.TOOL_USE, Usage(2, 3))

    transcript = b"".join([chunk async for chunk in AnthropicStreamMapper().map_stream(events(), response_model="local-opus")]).decode()

    assert 'event: message_start\ndata: {"message":{"content":[],"id":"msg_response-1"' in transcript
    assert '"type":"text_delta"' in transcript
    assert '"partial_json":"{\\"path\\":"' in transcript
    assert '"stop_reason":"tool_use"' in transcript
    assert transcript.endswith('event: message_stop\ndata: {"type":"message_stop"}\n\n')


async def test_suppresses_reasoning_and_closes_stream_on_cancellation() -> None:
    closed = False

    async def events():
        nonlocal closed
        try:
            yield ResponseStarted("response", "upstream")
            yield ReasoningStarted("reasoning")
            yield ReasoningDelta("reasoning", "hidden")
            yield ReasoningCompleted("reasoning")
            yield TextStarted("text")
            yield TextDelta("text", "visible")
        finally:
            closed = True

    stream = AnthropicStreamMapper().map_stream(events(), response_model="alias")
    await anext(stream)
    await stream.aclose()

    assert closed is True


async def test_emits_error_without_message_stop_after_stream_has_started() -> None:
    async def events():
        yield ResponseStarted("response", "upstream")
        yield ResponseFailed(ProviderUnavailableError("internal provider detail"))

    transcript = b"".join([chunk async for chunk in AnthropicStreamMapper().map_stream(events(), response_model="alias")]).decode()

    assert 'event: error\ndata: {"error":{"message":"Upstream provider is unavailable","type":"overloaded_error"},"type":"error"}\n\n' in transcript
    assert "message_stop" not in transcript


async def test_ignores_late_failure_after_message_stop() -> None:
    async def events():
        yield ResponseStarted("response", "upstream")
        yield completed(FinishReason.END_TURN, Usage(1, 1))
        yield ResponseFailed(ProviderUnavailableError("late"))

    transcript = b"".join([chunk async for chunk in AnthropicStreamMapper().map_stream(events(), response_model="alias")]).decode()

    assert transcript.endswith('event: message_stop\ndata: {"type":"message_stop"}\n\n')
    assert "event: error" not in transcript


async def test_projects_context_management_in_final_message_delta() -> None:
    async def events():
        yield ResponseStarted("response", "upstream")
        yield completed(FinishReason.END_TURN, Usage(1, 1), ContextManagementResult())

    transcript = b"".join([chunk async for chunk in AnthropicStreamMapper().map_stream(events(), response_model="alias")]).decode()

    assert '"context_management":{"applied_edits":[]}' in transcript
    assert transcript.index('"context_management"') < transcript.index("event: message_stop")
