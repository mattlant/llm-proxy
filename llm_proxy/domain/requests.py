from __future__ import annotations

import copy
from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import Any, Mapping, TypeAlias

from .messages import Message
from .tools import ToolDefinition


try:
    from enum import StrEnum
except ImportError:  # pragma: no cover - compatibility for Python 3.10
    class StrEnum(str, Enum):
        pass


def _copy_mapping(value: Mapping[str, Any], field_name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{field_name} must be a mapping")
    return MappingProxyType(copy.deepcopy(dict(value)))


JsonValue: TypeAlias = None | bool | int | float | str | Mapping[str, "JsonValue"] | tuple["JsonValue", ...]


def _freeze_json(value: Any, field_name: str) -> JsonValue:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Mapping):
        if not all(isinstance(key, str) for key in value):
            raise TypeError(f"{field_name} keys must be strings")
        return MappingProxyType({key: _freeze_json(item, field_name) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_json(item, field_name) for item in value)
    raise TypeError(f"{field_name} must contain only JSON-compatible values")


class EffortLevel(StrEnum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    XHIGH = "xhigh"
    MAX = "max"


class ReasoningMode(StrEnum):
    DISABLED = "disabled"
    ADAPTIVE = "adaptive"
    ENABLED = "enabled"


class ReasoningDisplay(StrEnum):
    SUMMARIZED = "summarized"
    OMITTED = "omitted"


class ContextEditKind(StrEnum):
    CLEAR_THINKING = "clear_thinking_20251015"


@dataclass(frozen=True, slots=True)
class PromptCachePreference:
    ephemeral_blocks: int

    def __post_init__(self) -> None:
        if not isinstance(self.ephemeral_blocks, int) or isinstance(self.ephemeral_blocks, bool) or self.ephemeral_blocks <= 0:
            raise ValueError("ephemeral_blocks must be a positive integer")


@dataclass(frozen=True, slots=True)
class ReasoningPreference:
    mode: ReasoningMode
    display: ReasoningDisplay
    budget_tokens: int | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.mode, ReasoningMode):
            raise TypeError("mode must be a ReasoningMode")
        if not isinstance(self.display, ReasoningDisplay):
            raise TypeError("display must be a ReasoningDisplay")
        if self.budget_tokens is not None and self.budget_tokens <= 0:
            raise ValueError("budget_tokens must be positive when supplied")
        if self.mode is ReasoningMode.ENABLED and self.budget_tokens is None:
            raise ValueError("enabled reasoning requires budget_tokens")
        if self.mode in (ReasoningMode.DISABLED, ReasoningMode.ADAPTIVE) and self.budget_tokens is not None:
            raise ValueError(f"{self.mode.value} reasoning does not accept budget_tokens")


@dataclass(frozen=True, slots=True)
class JsonSchemaOutputConstraint:
    schema: Mapping[str, JsonValue]

    def __post_init__(self) -> None:
        if not isinstance(self.schema, Mapping):
            raise TypeError("schema must be a mapping")
        object.__setattr__(self, "schema", _freeze_json(self.schema, "schema"))


@dataclass(frozen=True, slots=True)
class ContextEditRequest:
    kind: ContextEditKind
    keep: str

    def __post_init__(self) -> None:
        if not isinstance(self.kind, ContextEditKind):
            raise TypeError("kind must be a ContextEditKind")
        if self.keep != "all":
            raise ValueError("keep must be 'all'")


@dataclass(frozen=True, slots=True)
class ContextManagementRequest:
    edits: tuple[ContextEditRequest, ...]

    def __post_init__(self) -> None:
        edits = tuple(self.edits)
        if not edits:
            raise ValueError("context management must contain at least one edit")
        if not all(isinstance(edit, ContextEditRequest) for edit in edits):
            raise TypeError("context management edits must be ContextEditRequest values")
        object.__setattr__(self, "edits", edits)


@dataclass(frozen=True, slots=True)
class ContextManagementResult:
    applied_edits: tuple[ContextEditRequest, ...] = ()

    def __post_init__(self) -> None:
        edits = tuple(self.applied_edits)
        if not all(isinstance(edit, ContextEditRequest) for edit in edits):
            raise TypeError("applied edits must be ContextEditRequest values")
        object.__setattr__(self, "applied_edits", edits)


@dataclass(frozen=True, slots=True)
class ExecutionControls:
    effort: EffortLevel | None = None
    reasoning: ReasoningPreference | None = None
    output_constraint: JsonSchemaOutputConstraint | None = None
    context_management: ContextManagementRequest | None = None
    prompt_cache: PromptCachePreference | None = None

    def __post_init__(self) -> None:
        if self.effort is not None and not isinstance(self.effort, EffortLevel):
            raise TypeError("effort must be an EffortLevel")
        if self.reasoning is not None and not isinstance(self.reasoning, ReasoningPreference):
            raise TypeError("reasoning must be a ReasoningPreference")
        if self.output_constraint is not None and not isinstance(self.output_constraint, JsonSchemaOutputConstraint):
            raise TypeError("output_constraint must be a JsonSchemaOutputConstraint")
        if self.context_management is not None and not isinstance(self.context_management, ContextManagementRequest):
            raise TypeError("context_management must be a ContextManagementRequest")
        if self.prompt_cache is not None and not isinstance(self.prompt_cache, PromptCachePreference):
            raise TypeError("prompt_cache must be a PromptCachePreference")


@dataclass(frozen=True, slots=True)
class SamplingParameters:
    temperature: float | None = None
    top_p: float | None = None
    top_k: int | None = None
    min_p: float | None = None
    repeat_penalty: float | None = None
    repeat_last_n: int | None = None
    max_tokens: int | None = None
    stop_sequences: tuple[str, ...] = ()
    extra: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name in ("top_k", "repeat_last_n", "max_tokens"):
            value = getattr(self, name)
            if value is not None and value < 0:
                raise ValueError(f"{name} must not be negative")
        object.__setattr__(self, "stop_sequences", tuple(self.stop_sequences))
        object.__setattr__(self, "extra", _copy_mapping(self.extra, "extra"))


@dataclass(frozen=True, slots=True)
class AutomaticToolChoice:
    pass


@dataclass(frozen=True, slots=True)
class RequiredToolChoice:
    pass


@dataclass(frozen=True, slots=True)
class NoToolChoice:
    pass


@dataclass(frozen=True, slots=True)
class NamedToolChoice:
    name: str

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("tool choice name must be non-empty")


ToolChoice = AutomaticToolChoice | RequiredToolChoice | NoToolChoice | NamedToolChoice


@dataclass(frozen=True, slots=True)
class CompletionRequest:
    model: str
    messages: tuple[Message, ...]
    tools: tuple[ToolDefinition, ...] = ()
    tool_choice: ToolChoice = field(default_factory=AutomaticToolChoice)
    parameters: SamplingParameters = field(default_factory=SamplingParameters)
    stream: bool = False
    metadata: Mapping[str, Any] = field(default_factory=dict)
    controls: ExecutionControls = field(default_factory=ExecutionControls)

    def __post_init__(self) -> None:
        if not isinstance(self.model, str) or not self.model:
            raise ValueError("model must be non-empty")
        object.__setattr__(self, "messages", tuple(self.messages))
        object.__setattr__(self, "tools", tuple(self.tools))
        if not isinstance(self.tool_choice, (AutomaticToolChoice, RequiredToolChoice, NoToolChoice, NamedToolChoice)):
            raise TypeError("tool_choice must be a ToolChoice")
        if not isinstance(self.parameters, SamplingParameters):
            raise TypeError("parameters must be SamplingParameters")
        if not isinstance(self.controls, ExecutionControls):
            raise TypeError("controls must be an ExecutionControls")
        object.__setattr__(self, "metadata", _copy_mapping(self.metadata, "metadata"))
