from __future__ import annotations

from llm_proxy.domain.requests import (
    AutomaticToolChoice,
    CompletionRequest,
    NamedToolChoice,
    NoToolChoice,
    RequiredToolChoice,
)
from .parameters import parameters_from_source
from .dto import (
    OpenAIChatCompletionRequest,
    OpenAINamedToolChoice,
)
from .semantic_mapping import OpenAISemanticMapper


class OpenAIRequestMapper:
    def map_request(self, source: OpenAIChatCompletionRequest) -> CompletionRequest:
        return CompletionRequest(
            model=source.model,
            messages=OpenAISemanticMapper().map_dto_messages(source.messages),
            tools=OpenAISemanticMapper().map_dto_tools(source.tools),
            tool_choice=self._map_tool_choice(source.tool_choice),
            parameters=parameters_from_source(source),
            stream=source.stream,
            metadata=source.metadata or {},
        )

    @staticmethod
    def _map_tool_choice(source: object):
        if source is None or source == "auto":
            return AutomaticToolChoice()
        if source == "required":
            return RequiredToolChoice()
        if source == "none":
            return NoToolChoice()
        if isinstance(source, OpenAINamedToolChoice):
            return NamedToolChoice(source.function.name)
        raise InvalidCompletionRequest("unsupported tool_choice")
