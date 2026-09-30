"""Immutable canonical diagnostics for semantic support resolution."""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


try:
    from enum import StrEnum
except ImportError:  # pragma: no cover - compatibility for Python 3.10
    class StrEnum(str, Enum):
        pass


class SemanticIdentity(StrEnum):
    CONTEXT_MANAGEMENT = "context_management"
    EFFORT = "effort"
    PROMPT_CACHE = "prompt_cache"
    STRUCTURED_OUTPUT = "structured_output"
    REASONING = "reasoning"
    FINISH_REASON = "finish_reason"


class SemanticHandlingOutcome(StrEnum):
    SUPPORTED = "supported"
    REJECTED = "rejected"
    IGNORED_NOOP = "ignored_noop"
    ADVISORY = "advisory"
    LOSSY = "lossy"
    UNKNOWN = "unknown"


class SemanticResolutionScope(StrEnum):
    STATIC = "static"
    ROUTE_RESOLVED = "route_resolved"
    RUNTIME_OBSERVED = "runtime_observed"


class SemanticSupportPhase(StrEnum):
    REQUEST = "request"
    RESPONSE = "response"


@dataclass(frozen=True, slots=True)
class EffectiveSemanticHandling:
    semantic: SemanticIdentity
    phase: SemanticSupportPhase
    scope: SemanticResolutionScope
    outcome: SemanticHandlingOutcome
    reason: str

    def __post_init__(self) -> None:
        for name, enum_type in (
            ("semantic", SemanticIdentity),
            ("phase", SemanticSupportPhase),
            ("scope", SemanticResolutionScope),
            ("outcome", SemanticHandlingOutcome),
        ):
            if not isinstance(getattr(self, name), enum_type):
                raise TypeError(f"{name} must be a {enum_type.__name__}")
        if not isinstance(self.reason, str) or not self.reason.strip():
            raise ValueError("reason must be non-blank")


__all__ = [
    "EffectiveSemanticHandling",
    "SemanticHandlingOutcome",
    "SemanticIdentity",
    "SemanticResolutionScope",
    "SemanticSupportPhase",
]
