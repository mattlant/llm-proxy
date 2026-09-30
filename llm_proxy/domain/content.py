from __future__ import annotations

import copy
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Mapping


def _require_non_empty(value: str, field_name: str) -> None:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field_name} must be non-empty")


def _copy_mapping(value: Mapping[str, Any], field_name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{field_name} must be a mapping")
    return MappingProxyType(copy.deepcopy(dict(value)))


@dataclass(frozen=True, slots=True)
class TextContent:
    text: str


@dataclass(frozen=True, slots=True)
class ReasoningContent:
    text: str


@dataclass(frozen=True, slots=True)
class ToolCallContent:
    id: str
    name: str
    arguments: Mapping[str, Any]

    def __post_init__(self) -> None:
        _require_non_empty(self.id, "tool call id")
        _require_non_empty(self.name, "tool call name")
        object.__setattr__(self, "arguments", _copy_mapping(self.arguments, "tool call arguments"))


@dataclass(frozen=True, slots=True)
class ToolResultContent:
    tool_call_id: str
    content: tuple[ContentBlock, ...]
    is_error: bool = False

    def __post_init__(self) -> None:
        _require_non_empty(self.tool_call_id, "tool result id")
        object.__setattr__(self, "content", tuple(self.content))


ContentBlock = TextContent | ReasoningContent | ToolCallContent | ToolResultContent
