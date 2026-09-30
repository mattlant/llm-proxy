from __future__ import annotations

import json
import time
from collections.abc import AsyncIterator
from typing import Any

from llm_proxy.domain.events import (
    CompletionEvent,
    ResponseCompleted,
    ResponseFailed,
    ResponseStarted,
    ReasoningDelta,
    TextDelta,
    ToolCallArgumentsDelta,
    ToolCallStarted,
)

from .error_mapper import OpenAIErrorMapper
from .response_mapper import OpenAIResponseMapper


class OpenAIStreamMapper:
    def __init__(self, error_mapper: OpenAIErrorMapper | None = None) -> None:
        self._error_mapper = error_mapper or OpenAIErrorMapper()

    async def map_stream(self, events: AsyncIterator[CompletionEvent], *, response_model: str) -> AsyncIterator[bytes]:
        tool_indexes: dict[str, int] = {}
        started = False
        response_completed = False
        completion: ResponseCompleted | None = None
        response_id = ""
        created = int(time.time())
        try:
            async for event in events:
                if response_completed:
                    continue
                if isinstance(event, ResponseStarted):
                    started = True
                    response_id = OpenAIResponseMapper.response_id(event.response_id)
                    yield self._record(response_id, created, response_model, {"role": "assistant"})
                elif isinstance(event, TextDelta):
                    yield self._record(response_id, created, response_model, {"content": event.text})
                elif isinstance(event, ReasoningDelta):
                    yield self._record(response_id, created, response_model, {"reasoning_content": event.text})
                elif isinstance(event, ToolCallStarted):
                    index = len(tool_indexes)
                    tool_indexes[event.block_id] = index
                    yield self._record(response_id, created, response_model, {"tool_calls": [{"index": index, "id": event.tool_call_id, "type": "function", "function": {"name": event.name}}]})
                elif isinstance(event, ToolCallArgumentsDelta):
                    index = tool_indexes[event.block_id]
                    yield self._record(response_id, created, response_model, {"tool_calls": [{"index": index, "function": {"arguments": event.fragment}}]})
                elif isinstance(event, ResponseCompleted):
                    OpenAIResponseMapper.validate_semantic_handling(event)
                    choice: dict[str, Any] = {"index": 0, "delta": {}, "finish_reason": OpenAIResponseMapper._finish_reasons[event.finish_reason]}
                    result: dict[str, Any] = {"id": response_id, "object": "chat.completion.chunk", "created": created, "model": response_model, "choices": [choice]}
                    yield self._sse(result)
                    completion = event
                    response_completed = True
                elif isinstance(event, ResponseFailed):
                    _, payload = self._error_mapper.map_error(event.error)
                    yield self._sse(payload)
                    return
            if completion is not None:
                usage = OpenAIResponseMapper.map_usage(completion.usage)
                if usage is not None:
                    yield self._sse({"id": response_id, "object": "chat.completion.chunk", "created": created, "model": response_model, "choices": [], "usage": usage})
                yield b"data: [DONE]\n\n"
        except Exception as error:
            if not started:
                raise
            _, payload = self._error_mapper.map_error(error)
            yield self._sse(payload)
        finally:
            close = getattr(events, "aclose", None)
            if close is not None:
                await close()

    @staticmethod
    def _record(response_id: str, created: int, model: str, delta: dict[str, Any]) -> bytes:
        return OpenAIStreamMapper._sse({"id": response_id, "object": "chat.completion.chunk", "created": created, "model": model, "choices": [{"index": 0, "delta": delta, "finish_reason": None}]})

    @staticmethod
    def _sse(payload: dict[str, Any]) -> bytes:
        return f"data: {json.dumps(payload, separators=(',', ':'))}\n\n".encode("utf-8")
