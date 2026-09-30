from __future__ import annotations

from llm_proxy.domain.content import ReasoningContent, TextContent, ToolCallContent, ToolResultContent
from llm_proxy.domain.errors import InvalidCompletionRequest
from llm_proxy.domain.messages import Message, MessageRole
from llm_proxy.domain.requests import AutomaticToolChoice, CompletionRequest, ContextEditKind, ContextEditRequest, ContextManagementRequest, EffortLevel, ExecutionControls, JsonSchemaOutputConstraint, NamedToolChoice, NoToolChoice, PromptCachePreference, ReasoningDisplay, ReasoningMode, ReasoningPreference, RequiredToolChoice
from .parameters import parameters_from_source
from llm_proxy.domain.tools import ToolDefinition

from .dto import (
    AnthropicAnyToolChoice,
    AnthropicAutoToolChoice,
    AnthropicClearThinkingContextEdit,
    AnthropicContentBlock,
    AnthropicMessagesRequest,
    AnthropicNamedToolChoice,
    AnthropicNoToolChoice,
    AnthropicRedactedThinkingBlock,
    AnthropicAdaptiveThinkingConfig,
    AnthropicDisabledThinkingConfig,
    AnthropicEnabledThinkingConfig,
    AnthropicTextBlock,
    AnthropicThinkingBlock,
    AnthropicToolResultBlock,
    AnthropicToolUseBlock,
    AnthropicUnsupportedDocumentBlock,
    AnthropicUnsupportedImageBlock,
)


class AnthropicRequestMapper:
    def __init__(self, *, expose_thinking: bool = False) -> None:
        self._expose_thinking = expose_thinking

    def map_request(self, source: AnthropicMessagesRequest) -> CompletionRequest:
        messages: list[Message] = []
        if source.system is not None:
            system_blocks = (AnthropicTextBlock(type="text", text=source.system),) if isinstance(source.system, str) else tuple(source.system)
            messages.append(Message(MessageRole.SYSTEM, tuple(TextContent(block.text) for block in system_blocks)))
        for message in source.messages:
            messages.extend(self._map_messages(message.role, message.content))
        tools = tuple(ToolDefinition(tool.name, tool.description, tool.input_schema) for tool in source.tools)
        return CompletionRequest(
            model=source.model,
            messages=tuple(messages),
            tools=tools,
            tool_choice=self._map_tool_choice(source.tool_choice, tools),
            parameters=parameters_from_source(source),
            stream=source.stream,
            metadata=source.metadata or {},
            controls=self._map_controls(source),
        )

    @staticmethod
    def _map_controls(source: AnthropicMessagesRequest) -> ExecutionControls:
        output_config = source.output_config
        return ExecutionControls(
            effort=(EffortLevel(output_config.effort) if output_config is not None and output_config.effort is not None else None),
            output_constraint=(
                JsonSchemaOutputConstraint(output_config.format.json_schema)
                if output_config is not None and output_config.format is not None
                else None
            ),
            reasoning=AnthropicRequestMapper._map_reasoning(source.thinking),
            context_management=AnthropicRequestMapper._map_context_management(source),
            prompt_cache=AnthropicRequestMapper._map_prompt_cache(source),
        )

    @staticmethod
    def _map_prompt_cache(source: AnthropicMessagesRequest) -> PromptCachePreference | None:
        system_blocks = () if source.system is None or isinstance(source.system, str) else source.system
        count = sum(block.cache_control is not None for block in system_blocks)
        for message in source.messages:
            if isinstance(message.content, list):
                count += sum(isinstance(block, AnthropicTextBlock) and block.cache_control is not None for block in message.content)
                count += sum(isinstance(block, AnthropicToolResultBlock) and block.cache_control is not None for block in message.content)
        return PromptCachePreference(count) if count else None

    @staticmethod
    def _map_context_management(source: AnthropicMessagesRequest) -> ContextManagementRequest | None:
        if source.context_management is None:
            return None
        edits = []
        for edit in source.context_management.edits:
            if isinstance(edit, AnthropicClearThinkingContextEdit):
                edits.append(ContextEditRequest(ContextEditKind.CLEAR_THINKING, edit.keep))
            else:
                raise InvalidCompletionRequest("unsupported context-management edit")
        return ContextManagementRequest(tuple(edits))

    @staticmethod
    def _map_reasoning(source: AnthropicAdaptiveThinkingConfig | AnthropicEnabledThinkingConfig | AnthropicDisabledThinkingConfig | None) -> ReasoningPreference | None:
        if source is None:
            return None
        if isinstance(source, AnthropicAdaptiveThinkingConfig):
            return ReasoningPreference(ReasoningMode.ADAPTIVE, ReasoningDisplay(source.display))
        if isinstance(source, AnthropicEnabledThinkingConfig):
            return ReasoningPreference(ReasoningMode.ENABLED, ReasoningDisplay(source.display), source.budget_tokens)
        if isinstance(source, AnthropicDisabledThinkingConfig):
            return ReasoningPreference(ReasoningMode.DISABLED, ReasoningDisplay(source.display))
        raise InvalidCompletionRequest("unsupported thinking configuration")

    def _map_messages(self, role: str, source: str | list[AnthropicContentBlock]) -> tuple[Message, ...]:
        if role != "user" or isinstance(source, str):
            return (self._map_message(role, source),)

        mapped: list[Message] = []
        content: list = []
        for block in source:
            if isinstance(block, AnthropicToolResultBlock):
                if content:
                    mapped.append(Message(MessageRole.USER, tuple(content)))
                    content = []
                mapped.append(Message(MessageRole.TOOL, (self._map_block(block),)))
            elif self._include_block(block):
                content.append(self._map_block(block))
        if content:
            mapped.append(Message(MessageRole.USER, tuple(content)))
        if not mapped:
            raise InvalidCompletionRequest("message has no supported content blocks")
        return tuple(mapped)

    def _map_message(self, role: str, source: str | list[AnthropicContentBlock]) -> Message:
        blocks = (AnthropicTextBlock(type="text", text=source),) if isinstance(source, str) else tuple(source)
        content = tuple(self._map_block(block) for block in blocks if self._include_block(block))
        if not content:
            raise InvalidCompletionRequest("message has no supported content blocks")
        return Message(MessageRole.USER if role == "user" else MessageRole.ASSISTANT, content)

    def _include_block(self, block: AnthropicContentBlock) -> bool:
        return not isinstance(block, AnthropicThinkingBlock) or self._expose_thinking

    def _map_block(self, block: AnthropicContentBlock):
        if isinstance(block, AnthropicTextBlock):
            return TextContent(block.text)
        if isinstance(block, AnthropicToolUseBlock):
            return ToolCallContent(block.id, block.name, block.input)
        if isinstance(block, AnthropicToolResultBlock):
            values = (AnthropicTextBlock(type="text", text=block.content),) if isinstance(block.content, str) else tuple(block.content)
            return ToolResultContent(block.tool_use_id, tuple(TextContent(item.text) for item in values), is_error=block.is_error)
        if isinstance(block, AnthropicThinkingBlock):
            if block.signature is not None:
                raise InvalidCompletionRequest("signed thinking blocks are not supported")
            return ReasoningContent(block.thinking)
        if isinstance(block, AnthropicRedactedThinkingBlock):
            raise InvalidCompletionRequest("redacted_thinking blocks are not supported")
        if isinstance(block, (AnthropicUnsupportedImageBlock, AnthropicUnsupportedDocumentBlock)):
            raise InvalidCompletionRequest(f"{block.type} blocks are not supported")
        raise InvalidCompletionRequest("unsupported content block")

    @staticmethod
    def _map_tool_choice(source, tools: tuple[ToolDefinition, ...]):
        if source is None:
            return AutomaticToolChoice() if tools else NoToolChoice()
        if isinstance(source, AnthropicAutoToolChoice):
            return AutomaticToolChoice()
        if isinstance(source, AnthropicAnyToolChoice):
            return RequiredToolChoice()
        if isinstance(source, AnthropicNoToolChoice):
            return NoToolChoice()
        if isinstance(source, AnthropicNamedToolChoice):
            if source.name not in {tool.name for tool in tools}:
                raise InvalidCompletionRequest("tool_choice references an undefined tool")
            return NamedToolChoice(source.name)
        raise InvalidCompletionRequest("unsupported tool_choice")
