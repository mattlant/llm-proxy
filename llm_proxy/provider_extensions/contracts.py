"""Supported provider-extension SDK contracts."""
from __future__ import annotations

import copy
from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import Any, Protocol, runtime_checkable

from .execution import CompletionExecution, CompletionGateway, ProviderExecutionPolicy, SourceInterface
from llm_proxy.domain.events import CompletionEvent
from llm_proxy.domain.responses import CompletionResponse
from llm_proxy.domain.content import TextContent
from llm_proxy.domain.errors import ProviderUnavailableError
from llm_proxy.domain.events import ResponseCompleted, ResponseStarted, TextCompleted, TextDelta, TextStarted, ToolCallArgumentsDelta, ToolCallCompleted, ToolCallStarted
from llm_proxy.domain.messages import Message, MessageRole
from llm_proxy.domain.responses import FinishReason, Usage
from llm_proxy.domain.semantic_support import SemanticIdentity


PROVIDER_SDK_VERSION = "0.5.0"

try:
    from enum import StrEnum
except ImportError:  # pragma: no cover
    class StrEnum(str, Enum):
        pass


JsonValue = None | bool | int | float | str | tuple["JsonValue", ...] | Mapping[str, "JsonValue"]


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


@dataclass(frozen=True, slots=True)
class SdkCompatibility:
    minimum: str
    maximum: str | None = None
    core_compatibility: str | None = None

    def __post_init__(self) -> None:
        for name in ("minimum", "maximum", "core_compatibility"):
            value = getattr(self, name)
            if value is not None and (not isinstance(value, str) or not value.strip()):
                raise ValueError(f"{name} must be a non-blank string or null")


@dataclass(frozen=True, slots=True)
class ProviderExtensionMetadata:
    extension_id: str
    display_name: str
    package_name: str
    package_version: str

    def __post_init__(self) -> None:
        for name in ("extension_id", "display_name", "package_name", "package_version"):
            if not isinstance(getattr(self, name), str) or not getattr(self, name).strip():
                raise ValueError(f"{name} must be a non-blank string")


@dataclass(frozen=True, slots=True)
class ProviderCapabilities:
    completion: bool = True
    streaming: bool = False
    native_tools: bool = False
    parallel_tools: bool = False
    reasoning: bool = False
    structured_output: bool = False
    usage: bool = False
    cancellation: bool = True
    health: bool = False
    management_commands: bool = False
    model_listing: bool = False
    wire_protocols: tuple["WireProtocolCapability", ...] = ()

    def __post_init__(self) -> None:
        if not all(isinstance(getattr(self, name), bool) for name in self.__dataclass_fields__ if name != "wire_protocols"):
            raise TypeError("provider capabilities must be booleans")
        if self.parallel_tools and not self.native_tools:
            raise ValueError("parallel_tools requires native_tools")
        from .wire import WireProtocolCapability
        protocols = tuple(self.wire_protocols)
        if not all(isinstance(item, WireProtocolCapability) for item in protocols):
            raise TypeError("wire_protocols must contain WireProtocolCapability values")
        identities = [(item.protocol_id, item.version, item.endpoint) for item in protocols]
        if len(identities) != len(set(identities)):
            raise ValueError("duplicate wire protocol capability")
        object.__setattr__(self, "wire_protocols", protocols)


@dataclass(frozen=True, slots=True)
class ProviderInstanceConfig:
    """Immutable JSON-compatible configuration owned by the extension."""

    instance_name: str
    extension_id: str
    config: Mapping[str, JsonValue] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name in ("instance_name", "extension_id"):
            if not isinstance(getattr(self, name), str) or not getattr(self, name).strip():
                raise ValueError(f"{name} must be a non-blank string")
        if not isinstance(self.config, Mapping):
            raise TypeError("config must be a mapping")
        object.__setattr__(self, "config", _freeze_json(copy.deepcopy(dict(self.config)), "config"))


class ProviderHealthStatus(StrEnum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"


@dataclass(frozen=True, slots=True)
class ProviderHealth:
    status: ProviderHealthStatus
    diagnostic: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.status, ProviderHealthStatus):
            raise TypeError("status must be a ProviderHealthStatus")
        if not isinstance(self.diagnostic, str):
            raise TypeError("diagnostic must be a string")


class ProviderFactory(Protocol):
    def create(self, config: ProviderInstanceConfig) -> CompletionGateway:
        ...


@dataclass(frozen=True, slots=True)
class ProviderListedModel:
    upstream_model: str

    def __post_init__(self) -> None:
        if not isinstance(self.upstream_model, str) or not self.upstream_model.strip():
            raise ValueError("upstream_model must be non-blank")


class ProviderListingFailureCategory(StrEnum):
    UNAVAILABLE = "unavailable"
    TIMEOUT = "timeout"
    INVALID_RESPONSE = "invalid_response"


@dataclass(frozen=True, slots=True)
class ProviderListingError(Exception):
    provider: str
    category: ProviderListingFailureCategory
    message: str

    def __post_init__(self) -> None:
        if not isinstance(self.provider, str) or not self.provider.strip() or not isinstance(self.message, str) or not self.message.strip():
            raise ValueError("provider and message must be non-blank")
        if not isinstance(self.category, ProviderListingFailureCategory):
            raise TypeError("category must be a ProviderListingFailureCategory")


@runtime_checkable
class ProviderModelListingGateway(Protocol):
    async def list_models(self) -> tuple[ProviderListedModel, ...]:
        ...


@runtime_checkable
class ProviderHealthGateway(Protocol):
    async def health(self) -> ProviderHealth:
        ...


class ProviderExtension(Protocol):
    metadata: ProviderExtensionMetadata
    compatibility: SdkCompatibility
    capabilities: ProviderCapabilities
    factory: ProviderFactory


class ProviderSemanticDisposition(StrEnum):
    IMPLEMENTED = "implemented"
    UNSUPPORTED = "unsupported"


@dataclass(frozen=True, slots=True)
class ProviderSemanticDeclaration:
    """Optional provider implementation facts, not application outcomes."""

    support: Mapping[SemanticIdentity, ProviderSemanticDisposition]

    def __post_init__(self) -> None:
        if not isinstance(self.support, Mapping):
            raise TypeError("support must be a mapping")
        normalized = dict(self.support)
        if set(normalized) != {SemanticIdentity.REASONING}:
            raise ValueError("provider semantic declarations may contain only reasoning")
        if not isinstance(normalized[SemanticIdentity.REASONING], ProviderSemanticDisposition):
            raise TypeError("reasoning declaration must be a ProviderSemanticDisposition")
        object.__setattr__(self, "support", MappingProxyType(normalized))


@runtime_checkable
class ProviderSemanticExtension(Protocol):
    semantic_support: ProviderSemanticDeclaration


def provider_semantic_declaration(extension: object) -> ProviderSemanticDeclaration | None:
    """Return and validate one optional semantic declaration."""
    try:
        declaration = getattr(extension, "semantic_support")
    except AttributeError:
        return None
    if not isinstance(declaration, ProviderSemanticDeclaration):
        raise TypeError("semantic_support must be ProviderSemanticDeclaration")
    return declaration

__all__ = [
    "PROVIDER_SDK_VERSION", "CompletionEvent", "CompletionExecution", "CompletionGateway", "ProviderExecutionPolicy", "SourceInterface",
    "CompletionResponse", "ProviderCapabilities", "ProviderExtension", "ProviderExtensionMetadata",
    "ProviderFactory", "ProviderHealth", "ProviderHealthGateway", "ProviderHealthStatus", "ProviderInstanceConfig", "ProviderListedModel", "ProviderListingError", "ProviderListingFailureCategory", "ProviderModelListingGateway",
    "SdkCompatibility", "FinishReason", "Message", "MessageRole", "ResponseCompleted",
    "ProviderSemanticDeclaration", "ProviderSemanticDisposition", "ProviderSemanticExtension", "provider_semantic_declaration",
    "ResponseStarted", "TextCompleted", "TextContent", "TextDelta", "TextStarted", "ToolCallContent", "ToolCallStarted", "ToolCallArgumentsDelta", "ToolCallCompleted", "Usage",
    "ProviderUnavailableError",
]
