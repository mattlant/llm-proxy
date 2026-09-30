from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from .messages import Message
from .requests import ContextManagementResult
from .semantic_support import EffectiveSemanticHandling


try:
    from enum import StrEnum
except ImportError:  # pragma: no cover - compatibility for Python 3.10
    class StrEnum(str, Enum):
        pass


class FinishReason(StrEnum):
    END_TURN = "end_turn"
    MAX_TOKENS = "max_tokens"
    STOP_SEQUENCE = "stop_sequence"
    TOOL_USE = "tool_use"
    PAUSE_TURN = "pause_turn"
    REFUSAL = "refusal"


@dataclass(frozen=True, slots=True)
class Usage:
    input_tokens: int | None
    output_tokens: int | None

    def __post_init__(self) -> None:
        for name in ("input_tokens", "output_tokens"):
            value = getattr(self, name)
            if value is not None and value < 0:
                raise ValueError(f"{name} must not be negative")


@dataclass(frozen=True, slots=True)
class CompletionResponse:
    id: str
    model: str
    message: Message
    finish_reason: FinishReason
    stop_sequence: str | None
    usage: Usage
    context_management: ContextManagementResult | None = None
    semantic_handling: tuple[EffectiveSemanticHandling, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.id, str) or not self.id:
            raise ValueError("response id must be non-empty")
        if not isinstance(self.model, str) or not self.model:
            raise ValueError("model must be non-empty")
        if not isinstance(self.message, Message):
            raise TypeError("message must be a Message")
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
