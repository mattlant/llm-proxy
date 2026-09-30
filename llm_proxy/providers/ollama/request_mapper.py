from __future__ import annotations

import json
import copy
from typing import Any

from llm_proxy.provider_extensions import CompletionExecution
from llm_proxy.domain.content import (
    ReasoningContent,
    TextContent,
    ToolCallContent,
    ToolResultContent,
)
from llm_proxy.domain.messages import Message, MessageRole
from llm_proxy.domain.messages import DeveloperRoleMode
from llm_proxy.domain.errors import DeveloperRoleCompatibilityError
from llm_proxy.domain.requests import (
    AutomaticToolChoice,
    NamedToolChoice,
    NoToolChoice,
    RequiredToolChoice,
    ToolChoice,
)
from llm_proxy.domain.tools import ToolDefinition
from .parameters import OllamaParameterProjection


class OllamaOpenAIRequestMapper:

    def map_request(self, execution: CompletionExecution) -> dict[str, Any]:
        request = execution.request
        parameters = execution.parameters
        body: dict[str, Any] = {
            "model": execution.upstream_model,
            "messages": self.map_messages(request.messages, execution.policy.developer_role_mode),
            "stream": request.stream,
        }
        if request.tools:
            body["tools"] = self.map_tools(request.tools)
            body["tool_choice"] = self.map_tool_choice(request.tool_choice, request.tools)
        elif not isinstance(request.tool_choice, AutomaticToolChoice):
            body["tool_choice"] = self.map_tool_choice(request.tool_choice, ())
        self._map_parameters(body, parameters)
        self._map_output_constraint(body, execution)
        if execution.policy.expose_thinking:
            body["think"] = True
        if request.stream:
            body["stream_options"] = {"include_usage": True}
        return body

    @staticmethod
    def _map_output_constraint(body: dict[str, Any], execution: CompletionExecution) -> None:
        constraint = execution.request.controls.output_constraint
        if constraint is None:
            return
        schema = OllamaOpenAIRequestMapper._mutable_json(constraint.schema)
        if execution.policy.structured_output_mode == "openai_json_schema":
            body["response_format"] = {"type": "json_schema", "json_schema": {"name": "response", "schema": schema}}
            return
        if execution.policy.structured_output_mode == "ollama_format":
            body["format"] = schema
            return
        raise ValueError("selected model does not support JSON schema structured output")

    @staticmethod
    def _mutable_json(value: Any) -> Any:
        if isinstance(value, dict) or hasattr(value, "items"):
            return {key: OllamaOpenAIRequestMapper._mutable_json(item) for key, item in value.items()}
        if isinstance(value, tuple):
            return [OllamaOpenAIRequestMapper._mutable_json(item) for item in value]
        return value

    def _map_parameters(self, body: dict[str, Any], parameters: Any) -> None:
        values = OllamaParameterProjection().project(parameters)
        try:
            body.update(values)
            json.dumps(body)
        except (TypeError, ValueError) as exc:
            raise ValueError("extra parameters must be JSON-compatible") from exc

    def map_tools(self, tools: tuple[ToolDefinition, ...]) -> list[dict[str, Any]]:
        return [
            {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": copy.deepcopy(dict(tool.input_schema)),
                },
            }
            for tool in tools
        ]

    def map_tool_choice(
        self,
        tool_choice: ToolChoice,
        tools: tuple[ToolDefinition, ...],
    ) -> str | dict[str, Any]:
        if isinstance(tool_choice, AutomaticToolChoice):
            return "auto"
        if isinstance(tool_choice, RequiredToolChoice):
            return "required"
        if isinstance(tool_choice, NoToolChoice):
            return "none"
        if isinstance(tool_choice, NamedToolChoice):
            if tool_choice.name not in {tool.name for tool in tools}:
                raise ValueError(f"named tool choice is not defined: {tool_choice.name}")
            return {
                "type": "function",
                "function": {"name": tool_choice.name},
            }
        raise TypeError("tool_choice must be a ToolChoice")

    def map_messages(
        self,
        messages: tuple[Message, ...],
        developer_role_mode: DeveloperRoleMode = DeveloperRoleMode.PRESERVE,
    ) -> list[dict[str, Any]]:
        return [
            outbound
            for message in messages
            for outbound in self._map_message(message, developer_role_mode)
        ]

    def _map_message(self, message: Message, developer_role_mode: DeveloperRoleMode) -> list[dict[str, Any]]:
        if message.role in (MessageRole.SYSTEM, MessageRole.USER):
            return [self._map_text_message(message)]
        if message.role is MessageRole.DEVELOPER:
            if developer_role_mode is DeveloperRoleMode.REJECT:
                raise DeveloperRoleCompatibilityError()
            mapped = self._map_text_message(message)
            if developer_role_mode is DeveloperRoleMode.SYSTEM:
                mapped["role"] = MessageRole.SYSTEM.value
            return [mapped]
        if message.role is MessageRole.ASSISTANT:
            return [self._map_assistant_message(message)]
        if message.role is MessageRole.TOOL:
            if not all(isinstance(block, ToolResultContent) for block in message.content):
                raise ValueError("tool messages may contain only tool results")
            return [self._map_tool_result(block) for block in message.content]
        raise ValueError(f"unsupported message role: {message.role.value}")

    def _map_text_message(self, message: Message) -> dict[str, Any]:
        text = self._visible_text(message.content, allow_tool_calls=False)
        if not text:
            raise ValueError(f"{message.role.value} message has no visible text")
        return {"role": message.role.value, "content": text}

    def _map_assistant_message(self, message: Message) -> dict[str, Any]:
        text_parts: list[str] = []
        tool_calls: list[dict[str, Any]] = []
        for block in message.content:
            if isinstance(block, TextContent):
                text_parts.append(block.text)
            elif isinstance(block, ToolCallContent):
                tool_calls.append(self._map_tool_call(block))
            elif isinstance(block, ReasoningContent):
                continue
            else:
                raise ValueError("assistant messages may not contain tool results")

        if not text_parts and not tool_calls:
            raise ValueError("assistant message has no mappable content")
        mapped: dict[str, Any] = {"role": MessageRole.ASSISTANT.value}
        if text_parts:
            mapped["content"] = "\n".join(text_parts)
        if tool_calls:
            mapped["tool_calls"] = tool_calls
        return mapped

    def _map_tool_result(self, block: ToolResultContent) -> dict[str, Any]:
        text = self._visible_text(block.content, allow_tool_calls=False)
        if block.is_error:
            # OpenAI-compatible tool messages have no structural error flag.
            text = f"[tool_error]\n{text}"
        return {
            "role": MessageRole.TOOL.value,
            "tool_call_id": block.tool_call_id,
            "content": text,
        }

    def _map_tool_call(self, block: ToolCallContent) -> dict[str, Any]:
        return {
            "id": block.id,
            "type": "function",
            "function": {
                "name": block.name,
                "arguments": json.dumps(dict(block.arguments), ensure_ascii=False, separators=(",", ":")),
            },
        }

    def _visible_text(self, blocks: tuple[Any, ...], *, allow_tool_calls: bool) -> str:
        parts: list[str] = []
        for block in blocks:
            if isinstance(block, TextContent):
                parts.append(block.text)
            elif isinstance(block, ReasoningContent):
                continue
            elif isinstance(block, ToolCallContent) and allow_tool_calls:
                continue
            else:
                raise ValueError("content contains a block unsupported for text mapping")
        return "\n".join(parts)
