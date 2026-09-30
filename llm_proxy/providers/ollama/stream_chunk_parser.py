from __future__ import annotations

import hashlib
import json
from typing import Any

from llm_proxy.domain.errors import InvalidToolArgumentsError, ProviderProtocolError
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
from llm_proxy.domain.responses import FinishReason, Usage

from .stream_parser import SseRecord
from .protocol_mapping import map_finish_reason, map_usage_count
from .stream_state import ToolCallStreamState, ToolCallStreamStateStore


class OpenAIStreamParser:
    def __init__(self, *, expose_thinking: bool = False) -> None:
        self._started = False
        self._semantic_completed = False
        self._protocol_completed = False
        self._text_started = False
        self._finish_reason: FinishReason | None = None
        self._response_id: str | None = None
        self._model: str | None = None
        self._input_tokens: int | None = None
        self._output_tokens: int | None = None
        self._completion_event: ResponseCompleted | None = None
        self._tools: ToolCallStreamStateStore | None = None
        self._expose_thinking = expose_thinking
        self._reasoning_started = False

    def feed(self, record: SseRecord) -> tuple[Any, ...]:
        if self._protocol_completed:
            return ()
        if record.data == "[DONE]":
            events = self._complete()
            self._protocol_completed = True
            return events
        if not record.data:
            return ()
        try:
            payload = json.loads(record.data)
        except json.JSONDecodeError as exc:
            raise ProviderProtocolError("upstream stream data is not valid JSON") from exc
        if not isinstance(payload, dict):
            raise ProviderProtocolError("upstream stream data must be an object")

        if self._semantic_completed:
            choices = payload.get("choices")
            usage = payload.get("usage")
            if choices != [] or not isinstance(usage, dict):
                raise ProviderProtocolError("stream chunk received after completion")
            self._update_usage(usage, [])
            if self._completion_event is not None:
                object.__setattr__(self._completion_event, "usage", Usage(self._input_tokens, self._output_tokens))
            return ()

        events: list[Any] = []
        if not self._started:
            self._start(payload, events)
        self._update_usage(payload.get("usage"), events)
        choices = payload.get("choices", [])
        if not isinstance(choices, list):
            raise ProviderProtocolError("upstream stream choices must be a list")
        if choices:
            if len(choices) != 1 or not isinstance(choices[0], dict):
                raise ProviderProtocolError("upstream stream must contain at most one choice")
            self._process_choice(choices[0], events)
        elif payload.get("usage") is None:
            raise ProviderProtocolError("upstream stream chunk is missing choices")
        return tuple(events)

    def finish(self) -> tuple[Any, ...]:
        if self._semantic_completed:
            return ()
        return self._complete()

    def _start(self, payload: dict[str, Any], events: list[Any]) -> None:
        response_id = payload.get("id")
        if response_id is not None and not isinstance(response_id, str):
            raise ProviderProtocolError("upstream stream response id must be a string")
        model = payload.get("model")
        if model is not None and not isinstance(model, str):
            raise ProviderProtocolError("upstream stream model must be a string")
        if not response_id:
            seed = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
            response_id = f"stream_{hashlib.sha256(seed.encode()).hexdigest()[:16]}"
        self._response_id = response_id
        self._model = model or "unknown"
        self._tools = ToolCallStreamStateStore(self._response_id)
        self._started = True
        events.append(ResponseStarted(self._response_id, self._model))

    def _update_usage(self, usage: Any, events: list[Any]) -> None:
        if usage is None:
            return
        if not isinstance(usage, dict):
            raise ProviderProtocolError("upstream stream usage must be an object")
        self._input_tokens = map_usage_count(usage.get("prompt_tokens"), "prompt_tokens", self._input_tokens)
        self._output_tokens = map_usage_count(usage.get("completion_tokens"), "completion_tokens", self._output_tokens)

    def _process_choice(self, choice: dict[str, Any], events: list[Any]) -> None:
        delta = choice.get("delta", {})
        if not isinstance(delta, dict):
            raise ProviderProtocolError("upstream stream delta must be an object")
        text = delta.get("content")
        if text is not None:
            if not isinstance(text, str):
                raise ProviderProtocolError("upstream stream text delta must be a string or null")
            if text:
                if not self._text_started:
                    self._text_started = True
                    events.append(TextStarted("text_0"))
                events.append(TextDelta("text_0", text))
        self._process_reasoning(delta, events)
        self._process_tool_calls(delta.get("tool_calls", []), events)
        finish_reason = choice.get("finish_reason")
        if finish_reason is not None:
            mapped = map_finish_reason(finish_reason)
            if self._finish_reason is not None:
                raise ProviderProtocolError("duplicate upstream stream finish reason")
            self._finish_reason = mapped
            events.extend(self._complete())

    def _process_tool_calls(self, raw_tool_calls: Any, events: list[Any]) -> None:
        if raw_tool_calls is None:
            raw_tool_calls = []
        if not isinstance(raw_tool_calls, list):
            raise ProviderProtocolError("upstream stream tool_calls must be a list")
        if self._tools is None:
            raise ProviderProtocolError("tool state is unavailable before response start")
        for position, raw_tool_call in enumerate(raw_tool_calls):
            if not isinstance(raw_tool_call, dict):
                raise ProviderProtocolError("upstream stream tool call must be an object")
            index = raw_tool_call.get("index", position)
            state = self._tools.state_for(index)
            has_identity_fragment = False
            if "id" in raw_tool_call and raw_tool_call["id"] is not None:
                if not isinstance(raw_tool_call["id"], str):
                    raise ProviderProtocolError("upstream stream tool call ID must be a string")
                self._tools.append_id(index, raw_tool_call["id"])
                has_identity_fragment = True
            function = raw_tool_call.get("function", {})
            if not isinstance(function, dict):
                raise ProviderProtocolError("upstream stream tool function must be an object")
            if "name" in function and function["name"] is not None:
                if not isinstance(function["name"], str):
                    raise ProviderProtocolError("upstream stream tool name must be a string")
                self._tools.append_name(index, function["name"])
                has_identity_fragment = True
            argument_fragment = function.get("arguments")
            if argument_fragment is not None:
                if not isinstance(argument_fragment, str):
                    raise ProviderProtocolError("upstream stream tool arguments must be a string")
                if argument_fragment:
                    self._tools.append_arguments(index, argument_fragment)
            if not state.started and not has_identity_fragment:
                self._start_tool(state, events)
            elif state.started and argument_fragment:
                events.append(ToolCallArgumentsDelta(state.block_id, state.tool_call_id or "", argument_fragment))

    def _process_reasoning(self, delta: dict[str, Any], events: list[Any]) -> None:
        if not self._expose_thinking:
            return
        for field in ("thinking", "reasoning", "reasoning_content"):
            value = delta.get(field)
            if value is None:
                continue
            if not isinstance(value, str):
                raise ProviderProtocolError(f"upstream stream {field} must be a string or null")
            if not value:
                continue
            if not self._reasoning_started:
                self._reasoning_started = True
                events.append(ReasoningStarted("reasoning_0"))
            events.append(ReasoningDelta("reasoning_0", value))
            break

    def _start_tool(self, state: ToolCallStreamState, events: list[Any]) -> None:
        if self._tools is None:
            raise ProviderProtocolError("tool state is unavailable before response start")
        state = self._tools.start_if_ready(state.upstream_index)
        events.append(ToolCallStarted(state.block_id, state.tool_call_id or "", state.name or ""))
        if state.arguments:
            events.append(ToolCallArgumentsDelta(state.block_id, state.tool_call_id or "", state.arguments))

    def _complete_tools(self, events: list[Any]) -> bool:
        if self._tools is None or not self._tools.states:
            return False
        for index in sorted(self._tools.states):
            state = self._tools.states[index]
            if not state.started:
                self._start_tool(state, events)
            try:
                arguments = json.loads(state.arguments) if state.arguments else {}
            except json.JSONDecodeError as exc:
                raise InvalidToolArgumentsError("upstream stream tool arguments are not valid JSON") from exc
            if not isinstance(arguments, dict):
                raise InvalidToolArgumentsError("upstream stream tool arguments must decode to an object")
            state.mark_completed()
            events.append(ToolCallCompleted(state.block_id, state.tool_call_id or "", arguments))
        return True

    def _complete(self) -> tuple[Any, ...]:
        if self._semantic_completed:
            return ()
        if not self._started:
            return ()
        events: list[Any] = []
        if self._text_started:
            events.append(TextCompleted("text_0"))
        has_tools = self._complete_tools(events)
        if has_tools and self._finish_reason is None:
            self._finish_reason = FinishReason.TOOL_USE
        if self._reasoning_started:
            events.append(ReasoningCompleted("reasoning_0"))
        self._finish_reason = self._finish_reason or FinishReason.END_TURN
        self._completion_event = ResponseCompleted(
            self._finish_reason,
            None,
            Usage(self._input_tokens, self._output_tokens),
        )
        events.append(self._completion_event)
        self._semantic_completed = True
        return tuple(events)
