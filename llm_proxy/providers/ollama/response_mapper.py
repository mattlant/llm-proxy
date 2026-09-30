from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from typing import Any

from llm_proxy.domain.content import ReasoningContent, TextContent, ToolCallContent
from llm_proxy.domain.errors import InvalidToolArgumentsError, ProviderProtocolError
from llm_proxy.domain.messages import Message, MessageRole
from llm_proxy.domain.responses import CompletionResponse, FinishReason, Usage

from .protocol_mapping import map_finish_reason, map_usage_count


class OllamaOpenAIResponseMapper:
    def map_response(self, payload: Mapping[str, Any], *, expose_thinking: bool = False) -> CompletionResponse:
        if not isinstance(payload, Mapping):
            raise ProviderProtocolError("upstream response must be an object")

        response_id = self._response_id(payload)
        model = payload.get("model")
        if not isinstance(model, str) or not model:
            raise ProviderProtocolError("upstream response is missing a model")

        choices = payload.get("choices")
        if not isinstance(choices, list) or not choices:
            raise ProviderProtocolError("upstream response must contain one choice")
        if len(choices) != 1:
            raise ProviderProtocolError("multiple upstream choices are unsupported")
        choice = choices[0]
        if not isinstance(choice, Mapping):
            raise ProviderProtocolError("upstream choice must be an object")

        message = choice.get("message")
        if not isinstance(message, Mapping):
            raise ProviderProtocolError("upstream choice is missing a message")
        canonical_message = self._map_message(message, response_id, choice.get("index", 0), expose_thinking)
        finish_reason = map_finish_reason(choice.get("finish_reason"))
        usage = self._usage(payload.get("usage"))

        return CompletionResponse(
            id=response_id,
            model=model,
            message=canonical_message,
            finish_reason=finish_reason,
            stop_sequence=None,
            usage=usage,
        )

    def _map_message(
        self,
        message: Mapping[str, Any],
        response_id: str,
        choice_index: Any,
        expose_thinking: bool,
    ) -> Message:
        if not isinstance(choice_index, int) or isinstance(choice_index, bool):
            raise ProviderProtocolError("upstream choice index must be an integer")
        content: list[ReasoningContent | TextContent | ToolCallContent] = []
        for field in ("thinking", "reasoning", "reasoning_content"):
            reasoning = message.get(field)
            if reasoning is not None:
                if not isinstance(reasoning, str):
                    raise ProviderProtocolError(f"upstream {field} must be a string or null")
                if expose_thinking and reasoning:
                    content.append(ReasoningContent(reasoning))
                break
        text = message.get("content")
        if text is not None:
            if not isinstance(text, str):
                raise ProviderProtocolError("upstream message content must be a string or null")
            if text:
                content.append(TextContent(text))

        tool_calls = message.get("tool_calls", [])
        if tool_calls is None:
            tool_calls = []
        if not isinstance(tool_calls, list):
            raise ProviderProtocolError("upstream tool_calls must be a list")
        for call_index, tool_call in enumerate(tool_calls):
            content.append(self._map_tool_call(tool_call, response_id, choice_index, call_index))
        if not content:
            raise ProviderProtocolError("upstream assistant message has no content")
        return Message(MessageRole.ASSISTANT, tuple(content))

    def _map_tool_call(
        self,
        tool_call: Any,
        response_id: str,
        choice_index: int,
        call_index: int,
    ) -> ToolCallContent:
        if not isinstance(tool_call, Mapping):
            raise ProviderProtocolError("upstream tool call must be an object")
        function = tool_call.get("function")
        if not isinstance(function, Mapping):
            raise ProviderProtocolError("upstream tool call is missing a function")
        name = function.get("name")
        if not isinstance(name, str) or not name:
            raise ProviderProtocolError("upstream tool call is missing a name")
        arguments = function.get("arguments")
        if not isinstance(arguments, str):
            raise InvalidToolArgumentsError("upstream tool arguments must be a JSON string")
        try:
            decoded = json.loads(arguments)
        except (TypeError, json.JSONDecodeError) as exc:
            raise InvalidToolArgumentsError("upstream tool arguments are not valid JSON") from exc
        if not isinstance(decoded, Mapping):
            raise InvalidToolArgumentsError("upstream tool arguments must decode to an object")

        tool_id = tool_call.get("id")
        if tool_id is None or tool_id == "":
            tool_id = self._fallback_tool_id(response_id, choice_index, call_index)
        elif not isinstance(tool_id, str):
            raise ProviderProtocolError("upstream tool call id must be a string")
        return ToolCallContent(tool_id, name, decoded)

    def _response_id(self, payload: Mapping[str, Any]) -> str:
        response_id = payload.get("id")
        if response_id not in (None, ""):
            if not isinstance(response_id, str):
                raise ProviderProtocolError("upstream response id must be a non-empty string")
            return response_id
        canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
        digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]
        return f"chatcmpl_{digest}"

    @staticmethod
    def _fallback_tool_id(response_id: str, choice_index: int, call_index: int) -> str:
        seed = f"{response_id}:{choice_index}:{call_index}"
        digest = hashlib.sha256(seed.encode("utf-8")).hexdigest()[:16]
        return f"call_{digest}"

    def _usage(self, value: Any) -> Usage:
        if value is None:
            return Usage(None, None)
        if not isinstance(value, Mapping):
            raise ProviderProtocolError("upstream usage must be an object")
        input_tokens = map_usage_count(value.get("prompt_tokens"), "prompt_tokens")
        output_tokens = map_usage_count(value.get("completion_tokens"), "completion_tokens")
        return Usage(input_tokens, output_tokens)
