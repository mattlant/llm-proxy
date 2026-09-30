from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Mapping, Protocol, runtime_checkable

from .contracts import JsonValue, frozen_mapping

SIDE_EFFECT_FAMILY = "side_effect"
SIDE_EFFECT_VERSION = 1

class SideEffectExecutionMode(str, Enum):
    CANONICAL = "canonical"
    PRESERVE = "preserve"

class SideEffectFailureCategory(str, Enum):
    UNAVAILABLE = "unavailable"
    REJECTED = "rejected"
    FAILED = "failed"

class SideEffectError(Exception):
    def __init__(self, category: SideEffectFailureCategory) -> None:
        self.category = category
        super().__init__(category.value)

@dataclass(frozen=True, slots=True)
class SideEffectContext:
    request_id: str
    trace_transaction_id: str | None
    binding_id: str
    source_interface: str
    execution_mode: SideEffectExecutionMode
    requested_model: str
    canonical_model: str
    provider_instance: str
    upstream_model: str
    stream: bool

@runtime_checkable
class SideEffectCapability(Protocol):
    def validate_arguments(self, arguments: Mapping[str, JsonValue]) -> None: ...
    async def invoke(self, context: SideEffectContext, arguments: Mapping[str, JsonValue]) -> None: ...

def validate_side_effect(implementation: object) -> None:
    if not isinstance(implementation, SideEffectCapability):
        raise TypeError("side effect implementation does not implement SideEffectCapability")
