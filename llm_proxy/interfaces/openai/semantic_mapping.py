from __future__ import annotations

import json
from collections.abc import Mapping

from llm_proxy.domain.content import TextContent, ToolCallContent, ToolResultContent
from llm_proxy.domain.errors import InvalidCompletionRequest
from llm_proxy.domain.messages import Message, MessageRole
from llm_proxy.domain.tools import ToolDefinition

from .dto import (
    OpenAIAssistantMessage,
    OpenAIDeveloperMessage,
    OpenAISystemMessage,
    OpenAIToolMessage,
    OpenAIUserMessage,
)


class OpenAISemanticMapper:
    def map_dto_messages(self, sources: object) -> tuple[Message, ...]:
        return tuple(self._map_dto_message(source) for source in sources)

    def map_wire_messages(self, source: object) -> tuple[Message, ...]:
        if not isinstance(source, list):
            return ()
        mapped = []
        for item in source:
            message = self._map_wire_message(item)
            if message is not None:
                mapped.append(message)
        return tuple(mapped)

    def map_dto_tools(self, sources: object) -> tuple[ToolDefinition, ...]:
        return tuple(ToolDefinition(self._required_function_name(source.function), source.function.description or "", source.function.parameters) for source in sources or ())

    def map_wire_tool_names(self, source: object) -> tuple[str, ...]:
        if not isinstance(source, list):
            return ()
        return tuple(name for item in source if isinstance(item, Mapping) and item.get("type") == "function" and (name := self._function_name(item.get("function"))) is not None)

    def _map_dto_message(self, source: object) -> Message:
        if isinstance(source, OpenAISystemMessage):
            return Message(MessageRole.SYSTEM, (TextContent(source.content),))
        if isinstance(source, OpenAIDeveloperMessage):
            return Message(MessageRole.DEVELOPER, (TextContent(source.content),))
        if isinstance(source, OpenAIUserMessage):
            return Message(MessageRole.USER, (TextContent(source.content if isinstance(source.content, str) else "".join(part.text for part in source.content)),))
        if isinstance(source, OpenAIToolMessage):
            return Message(MessageRole.TOOL, (ToolResultContent(source.tool_call_id, (TextContent(source.content),), is_error=False),))
        if isinstance(source, OpenAIAssistantMessage):
            content = []
            if source.content is not None:
                content.append(TextContent(source.content))
            content.extend(ToolCallContent(call.id, self._required_function_name(call.function), self._parse_arguments(call.function.arguments)) for call in source.tool_calls or ())
            return Message(MessageRole.ASSISTANT, tuple(content))
        raise InvalidCompletionRequest("unsupported OpenAI message")

    def _map_wire_message(self, source: object) -> Message | None:
        if not isinstance(source, Mapping) or not isinstance(source.get("role"), str):
            return None
        role, content = source["role"], source.get("content")
        if role in {"system", "developer"} and isinstance(content, str):
            return Message(MessageRole(role), (TextContent(content),))
        if role == "user":
            text = content if isinstance(content, str) else self._wire_user_text(content)
            return Message(MessageRole.USER, (TextContent(text),)) if text is not None else None
        if role == "tool" and isinstance(source.get("tool_call_id"), str) and source["tool_call_id"] and isinstance(content, str):
            return Message(MessageRole.TOOL, (ToolResultContent(source["tool_call_id"], (TextContent(content),), is_error=False),))
        if role == "assistant":
            blocks = []
            if isinstance(content, str):
                blocks.append(TextContent(content))
            calls = source.get("tool_calls")
            if isinstance(calls, list):
                for call in calls:
                    mapped = self._wire_tool_call(call)
                    if mapped is not None:
                        blocks.append(mapped)
            return Message(MessageRole.ASSISTANT, tuple(blocks)) if blocks else None
        return None

    @staticmethod
    def _wire_user_text(content: object) -> str | None:
        if not isinstance(content, list) or not content or not all(isinstance(item, Mapping) and item.get("type") == "text" and isinstance(item.get("text"), str) for item in content):
            return None
        return "".join(item["text"] for item in content)

    def _wire_tool_call(self, source: object) -> ToolCallContent | None:
        if not isinstance(source, Mapping) or not isinstance(source.get("id"), str) or not source["id"]:
            return None
        name = self._function_name(source.get("function"))
        arguments = source.get("function", {}).get("arguments") if isinstance(source.get("function"), Mapping) else None
        if name is None or not isinstance(arguments, str):
            return None
        try:
            return ToolCallContent(source["id"], name, self._parse_arguments(arguments))
        except InvalidCompletionRequest:
            return None

    @staticmethod
    def _function_name(function: object) -> str | None:
        name = function.get("name") if isinstance(function, Mapping) else getattr(function, "name", None)
        return name if isinstance(name, str) and name else None

    def _required_function_name(self, function: object) -> str:
        name = self._function_name(function)
        if name is None:
            raise InvalidCompletionRequest("OpenAI function name must be non-empty")
        return name

    @staticmethod
    def _parse_arguments(raw: str) -> Mapping[str, object]:
        try:
            value = json.loads(raw)
        except json.JSONDecodeError as error:
            raise InvalidCompletionRequest("tool call arguments must be valid JSON") from error
        if not isinstance(value, Mapping):
            raise InvalidCompletionRequest("tool call arguments must be a JSON object")
        return value
