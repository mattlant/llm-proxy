from __future__ import annotations

import copy
from dataclasses import dataclass
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Mapping

from .responses import FinishReason, Usage
from .semantic_support import EffectiveSemanticHandling
from .requests import ContextManagementResult

if TYPE_CHECKING:
    from .errors import CompletionError


def _require_non_empty(value: str, field_name: str) -> None:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field_name} must be non-empty")


def _copy_mapping(value: Mapping[str, Any], field_name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{field_name} must be a mapping")
    return MappingProxyType(copy.deepcopy(dict(value)))


@dataclass(frozen=True, slots=True)
class ResponseStarted:
    response_id: str
    model: str

    def __post_init__(self) -> None:
        _require_non_empty(self.response_id, "response id")
        _require_non_empty(self.model, "model")


@dataclass(frozen=True, slots=True)
class TextStarted:
    block_id: str

    def __post_init__(self) -> None:
        _require_non_empty(self.block_id, "block id")


@dataclass(frozen=True, slots=True)
class TextDelta:
    block_id: str
    text: str

    def __post_init__(self) -> None:
        _require_non_empty(self.block_id, "block id")
        _require_non_empty(self.text, "text delta")


@dataclass(frozen=True, slots=True)
class TextCompleted:
    block_id: str

    def __post_init__(self) -> None:
        _require_non_empty(self.block_id, "block id")


@dataclass(frozen=True, slots=True)
class ToolCallStarted:
    block_id: str
    tool_call_id: str
    name: str

    def __post_init__(self) -> None:
        _require_non_empty(self.block_id, "block id")
        _require_non_empty(self.tool_call_id, "tool call id")
        _require_non_empty(self.name, "tool call name")


@dataclass(frozen=True, slots=True)
class ToolCallArgumentsDelta:
    block_id: str
    tool_call_id: str
    fragment: str

    def __post_init__(self) -> None:
        _require_non_empty(self.block_id, "block id")
        _require_non_empty(self.tool_call_id, "tool call id")
        _require_non_empty(self.fragment, "tool arguments fragment")


@dataclass(frozen=True, slots=True)
class ToolCallCompleted:
    block_id: str
    tool_call_id: str
    arguments: Mapping[str, Any]

    def __post_init__(self) -> None:
        _require_non_empty(self.block_id, "block id")
        _require_non_empty(self.tool_call_id, "tool call id")
        object.__setattr__(self, "arguments", _copy_mapping(self.arguments, "tool call arguments"))


@dataclass(frozen=True, slots=True)
class ReasoningStarted:
    block_id: str

    def __post_init__(self) -> None:
        _require_non_empty(self.block_id, "block id")


@dataclass(frozen=True, slots=True)
class ReasoningDelta:
    block_id: str
    text: str

    def __post_init__(self) -> None:
        _require_non_empty(self.block_id, "block id")
        _require_non_empty(self.text, "reasoning delta")


@dataclass(frozen=True, slots=True)
class ReasoningCompleted:
    block_id: str

    def __post_init__(self) -> None:
        _require_non_empty(self.block_id, "block id")


@dataclass(frozen=True, slots=True)
class ResponseCompleted:
    finish_reason: FinishReason
    stop_sequence: str | None
    usage: Usage
    context_management: ContextManagementResult | None = None
    semantic_handling: tuple[EffectiveSemanticHandling, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.finish_reason, FinishReason):
            raise TypeError("finish_reason must be a FinishReason")
        if not isinstance(self.usage, Usage):
            raise TypeError("usage must be Usage")
        if self.context_management is not None and not isinstance(self.context_management, ContextManagementResult):
            raise TypeError("context_management must be ContextManagementResult or null")
        semantic_handling = tuple(self.semantic_handling)
        if not all(isinstance(item, EffectiveSemanticHandling) for item in semantic_handling):
            raise TypeError("semantic_handling must contain EffectiveSemanticHandling values")
        object.__setattr__(self, "semantic_handling", semantic_handling)


@dataclass(frozen=True, slots=True)
class ResponseFailed:
    error: CompletionError


CompletionEvent = (
    ResponseStarted
    | TextStarted
    | TextDelta
    | TextCompleted
    | ToolCallStarted
    | ToolCallArgumentsDelta
    | ToolCallCompleted
    | ReasoningStarted
    | ReasoningDelta
    | ReasoningCompleted
    | ResponseCompleted
    | ResponseFailed
)
