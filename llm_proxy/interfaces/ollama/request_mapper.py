from __future__ import annotations

from llm_proxy.domain.content import TextContent, ToolCallContent, ToolResultContent
from llm_proxy.domain.messages import Message, MessageRole
from llm_proxy.domain.requests import AutomaticToolChoice, CompletionRequest, SamplingParameters
from llm_proxy.domain.tools import ToolDefinition

from .dto import OllamaChatRequest, OllamaGenerateRequest, OllamaMessage, OllamaOptions
from .parameters import parameters_from_options


class OllamaRequestMapper:
    def map_chat(self, source: OllamaChatRequest) -> CompletionRequest:
        return CompletionRequest(
            model=source.model,
            messages=tuple(self._map_message(message, index) for index, message in enumerate(source.messages)),
            tools=tuple(ToolDefinition(tool.function.name, tool.function.description or "", tool.function.parameters) for tool in source.tools or ()),
            tool_choice=AutomaticToolChoice(),
            parameters=self._map_options(source.options),
            stream=source.stream,
        )

    def map_generate(self, source: OllamaGenerateRequest) -> CompletionRequest:
        messages = []
        if source.system is not None:
            messages.append(Message(MessageRole.SYSTEM, (TextContent(source.system),)))
        messages.append(Message(MessageRole.USER, (TextContent(source.prompt),)))
        return CompletionRequest(
            model=source.model,
            messages=tuple(messages),
            parameters=self._map_options(source.options),
            stream=source.stream,
        )

    @staticmethod
    def _map_options(source: OllamaOptions | None) -> SamplingParameters:
        return parameters_from_options(source)

    @staticmethod
    def _map_message(source: OllamaMessage, message_index: int) -> Message:
        if source.role == "system":
            return Message(MessageRole.SYSTEM, (TextContent(source.content),))
        if source.role == "user":
            return Message(MessageRole.USER, (TextContent(source.content),))
        if source.role == "tool":
            call_id = source.tool_call_id or f"tool-{message_index}"
            return Message(MessageRole.TOOL, (ToolResultContent(call_id, (TextContent(source.content),), is_error=False),))
        content = [TextContent(source.content)] if source.content else []
        content.extend(
            ToolCallContent(call.id or f"call-{message_index}-{index}", call.function.name, call.function.arguments)
            for index, call in enumerate(source.tool_calls or ())
        )
        return Message(MessageRole.ASSISTANT, tuple(content))
