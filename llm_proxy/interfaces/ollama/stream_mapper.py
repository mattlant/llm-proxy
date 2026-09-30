from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Any

from llm_proxy.domain.events import CompletionEvent, ResponseCompleted, ResponseFailed, TextDelta, ToolCallCompleted, ToolCallStarted

from .error_mapper import OllamaErrorMapper
from .response_mapper import OllamaResponseMapper


class OllamaStreamMapper:
    def __init__(self, error_mapper: OllamaErrorMapper | None = None) -> None:
        self._error_mapper = error_mapper or OllamaErrorMapper()

    async def map_chat(self, events: AsyncIterator[CompletionEvent], *, response_model: str) -> AsyncIterator[bytes]:
        async for record in self._records(events, response_model=response_model, generate=False):
            yield self._line(record)

    async def map_generate(self, events: AsyncIterator[CompletionEvent], *, response_model: str) -> AsyncIterator[bytes]:
        async for record in self._records(events, response_model=response_model, generate=True):
            yield self._line(record)

    async def _records(self, events: AsyncIterator[CompletionEvent], *, response_model: str, generate: bool) -> AsyncIterator[dict[str, Any]]:
        mapper = OllamaResponseMapper()
        tool_names: dict[str, str] = {}
        try:
            async for event in events:
                if isinstance(event, TextDelta):
                    yield self._fragment(response_model, mapper.created_at(), event.text, generate)
                elif isinstance(event, ToolCallStarted):
                    tool_names[event.block_id] = event.name
                elif isinstance(event, ToolCallCompleted) and not generate:
                    yield {"model": response_model, "created_at": mapper.created_at(), "message": {"role": "assistant", "content": "", "tool_calls": [{"function": {"name": tool_names.get(event.block_id, ""), "arguments": dict(event.arguments)}}]}, "done": False}
                elif isinstance(event, ResponseCompleted):
                    OllamaResponseMapper.validate_semantic_handling(event)
                    result: dict[str, Any] = {"model": response_model, "created_at": mapper.created_at(), "done": True, "done_reason": mapper._done_reasons[event.finish_reason]}
                    result.update(mapper.usage(event.usage))
                    if generate:
                        result["response"] = ""
                    else:
                        result["message"] = {"role": "assistant", "content": ""}
                    yield result
                elif isinstance(event, ResponseFailed):
                    _, error = self._error_mapper.map_error(event.error)
                    yield error
                    return
        except Exception as error:
            _, payload = self._error_mapper.map_error(error)
            yield payload
        finally:
            close = getattr(events, "aclose", None)
            if close is not None:
                await close()

    @staticmethod
    def _fragment(model: str, created_at: str, text: str, generate: bool) -> dict[str, Any]:
        base: dict[str, Any] = {"model": model, "created_at": created_at, "done": False}
        if generate:
            base["response"] = text
        else:
            base["message"] = {"role": "assistant", "content": text}
        return base

    @staticmethod
    def _line(payload: dict[str, Any]) -> bytes:
        return f"{json.dumps(payload, separators=(',', ':'))}\n".encode("utf-8")
