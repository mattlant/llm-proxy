from __future__ import annotations

import json
import asyncio
from collections.abc import AsyncIterator, Mapping
from typing import Any

from llm_proxy.domain.events import (
    CompletionEvent,
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

from .error_mapper import AnthropicErrorMapper
from .response_mapper import AnthropicResponseMapper
from .stream_state import AnthropicStreamState

_SUPPORTED_EVENTS = frozenset({
    "message_start",
    "content_block_start",
    "content_block_delta",
    "content_block_stop",
    "message_delta",
    "message_stop",
    "ping",
    "error",
})


def serialize_sse(event_name: str, payload: Mapping[str, Any]) -> bytes:
    if event_name not in _SUPPORTED_EVENTS:
        raise ValueError(f"unsupported Anthropic SSE event '{event_name}'")
    if not isinstance(payload, Mapping):
        raise TypeError("SSE payload must be a mapping")
    data = json.dumps(dict(payload), ensure_ascii=False, separators=(",", ":"), sort_keys=True)
    return f"event: {event_name}\ndata: {data}\n\n".encode("utf-8")


class AnthropicStreamMapper:
    def __init__(self, error_mapper: AnthropicErrorMapper | None = None) -> None:
        self._error_mapper = error_mapper or AnthropicErrorMapper()

    async def map_stream(self, events: AsyncIterator[CompletionEvent], *, response_model: str) -> AsyncIterator[bytes]:
        state = AnthropicStreamState()
        try:
            async for event in events:
                if isinstance(event, ResponseStarted):
                    response_id = AnthropicResponseMapper.response_id(event.response_id)
                    state.start_message(response_id, response_model)
                    yield serialize_sse("message_start", {
                        "type": "message_start",
                        "message": {
                            "id": response_id,
                            "type": "message",
                            "role": "assistant",
                            "model": response_model,
                            "content": [],
                            "stop_reason": None,
                            "stop_sequence": None,
                            "usage": {"input_tokens": 0, "output_tokens": 0},
                        },
                    })
                elif isinstance(event, TextStarted):
                    block = state.start_block(event.block_id, "text")
                    yield serialize_sse("content_block_start", {"type": "content_block_start", "index": block.index, "content_block": {"type": "text", "text": ""}})
                elif isinstance(event, TextDelta):
                    block = state.append_delta(event.block_id, event.text)
                    yield serialize_sse("content_block_delta", {"type": "content_block_delta", "index": block.index, "delta": {"type": "text_delta", "text": event.text}})
                elif isinstance(event, TextCompleted):
                    block = state.stop_block(event.block_id)
                    yield serialize_sse("content_block_stop", {"type": "content_block_stop", "index": block.index})
                elif isinstance(event, ToolCallStarted):
                    block = state.start_block(event.block_id, "tool_use", tool_call_id=event.tool_call_id, tool_name=event.name)
                    yield serialize_sse("content_block_start", {"type": "content_block_start", "index": block.index, "content_block": {"type": "tool_use", "id": event.tool_call_id, "name": event.name, "input": {}}})
                elif isinstance(event, ToolCallArgumentsDelta):
                    block = state.append_delta(event.block_id, event.fragment)
                    yield serialize_sse("content_block_delta", {"type": "content_block_delta", "index": block.index, "delta": {"type": "input_json_delta", "partial_json": event.fragment}})
                elif isinstance(event, ToolCallCompleted):
                    block = state._require_open_block(event.block_id)
                    if not block.fragments:
                        encoded = json.dumps(dict(event.arguments), ensure_ascii=False, separators=(",", ":"), sort_keys=True)
                        state.append_delta(event.block_id, encoded)
                        yield serialize_sse("content_block_delta", {"type": "content_block_delta", "index": block.index, "delta": {"type": "input_json_delta", "partial_json": encoded}})
                    elif json.loads("".join(block.fragments)) != dict(event.arguments):
                        raise ValueError("tool argument stream does not match completed arguments")
                    state.stop_block(event.block_id)
                    yield serialize_sse("content_block_stop", {"type": "content_block_stop", "index": block.index})
                elif isinstance(event, ResponseCompleted):
                    AnthropicResponseMapper.validate_semantic_handling(event)
                    state.complete_message()
                    payload = {
                        "type": "message_delta",
                        "delta": {"stop_reason": event.finish_reason.value, "stop_sequence": event.stop_sequence},
                        "usage": {"output_tokens": event.usage.output_tokens or 0},
                    }
                    if event.context_management is not None:
                        payload["context_management"] = {"applied_edits": [{"type": edit.kind.value} for edit in event.context_management.applied_edits]}
                    yield serialize_sse("message_delta", payload)
                    yield serialize_sse("message_stop", {"type": "message_stop"})
                    return
                elif isinstance(event, ResponseFailed):
                    _, payload = self._error_mapper.map_error(event.error)
                    yield serialize_sse("error", payload)
                    return
                elif isinstance(event, (ReasoningStarted, ReasoningDelta, ReasoningCompleted)):
                    continue
        except asyncio.CancelledError:
            raise
        except Exception as error:
            if not state.started:
                raise
            _, payload = self._error_mapper.map_error(error)
            yield serialize_sse("error", payload)
        finally:
            close = getattr(events, "aclose", None)
            if close is not None:
                await close()
