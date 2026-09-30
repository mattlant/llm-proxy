from __future__ import annotations

import copy
from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Protocol, runtime_checkable

from llm_proxy.configuration.models import InterfaceName
from llm_proxy.domain.messages import Message
from llm_proxy.domain.requests import ExecutionControls, SamplingParameters
from llm_proxy.provider_extensions import WirePatch, WireProtocolCapability, WireRequest


@dataclass(frozen=True, slots=True)
class WireCapabilityIdentity:
    protocol_id: str
    version: str
    endpoint: str

    def __post_init__(self) -> None:
        if not all(isinstance(getattr(self, name), str) and getattr(self, name).strip() for name in ("protocol_id", "version", "endpoint")):
            raise ValueError("wire capability identity must be non-blank")
        if not self.endpoint.startswith("/"):
            raise ValueError("wire capability endpoint must start with '/'")

    @classmethod
    def from_capability(cls, capability: WireProtocolCapability) -> WireCapabilityIdentity:
        if not isinstance(capability, WireProtocolCapability):
            raise TypeError("capability must be a WireProtocolCapability")
        return cls(capability.protocol_id, capability.version, capability.endpoint)

    @property
    def route_label(self) -> str:
        return f"{self.protocol_id}@{self.version}"


@dataclass(frozen=True, slots=True)
class InboundWirePlanningRequest:
    wire_request: WireRequest
    interface: InterfaceName
    requested_model: str
    stream: bool
    parameters: SamplingParameters
    messages: tuple[Message, ...]
    tool_names: tuple[str, ...]
    metadata: Mapping[str, object]
    controls: ExecutionControls = field(default_factory=ExecutionControls)
    inbound_context: object | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.wire_request, WireRequest) or not isinstance(self.interface, InterfaceName):
            raise TypeError("wire_request and interface must be valid")
        if not isinstance(self.requested_model, str) or not self.requested_model.strip() or not isinstance(self.stream, bool):
            raise ValueError("wire planning model and stream are invalid")
        if not isinstance(self.parameters, SamplingParameters) or not isinstance(self.controls, ExecutionControls):
            raise TypeError("parameters and controls must be valid")
        object.__setattr__(self, "messages", tuple(self.messages))
        object.__setattr__(self, "tool_names", tuple(self.tool_names))
        if not isinstance(self.metadata, Mapping):
            raise TypeError("metadata must be a mapping")
        object.__setattr__(self, "metadata", MappingProxyType(copy.deepcopy(dict(self.metadata))))


@dataclass(frozen=True, slots=True)
class WireMutationProjection:
    patch: WirePatch | None
    fallback_reason: str | None

    def __post_init__(self) -> None:
        if (self.patch is None) == (self.fallback_reason is None):
            raise ValueError("exactly one mutation projection result is required")
        if self.patch is not None and not isinstance(self.patch, WirePatch):
            raise TypeError("patch must be a WirePatch")
        if self.fallback_reason is not None and (not isinstance(self.fallback_reason, str) or not self.fallback_reason.strip()):
            raise ValueError("fallback_reason must be non-blank")


@dataclass(frozen=True, slots=True)
class WireMutationDecision:
    upstream_model: str
    parameters: SamplingParameters
    stream: bool
    requires_stream_usage: bool

    def __post_init__(self) -> None:
        if not isinstance(self.upstream_model, str) or not self.upstream_model.strip():
            raise ValueError("upstream_model must be non-blank")
        if not isinstance(self.parameters, SamplingParameters) or not isinstance(self.stream, bool) or not isinstance(self.requires_stream_usage, bool):
            raise TypeError("mutation decision values are invalid")


@runtime_checkable
class InterfaceWireMutationProjector(Protocol):
    def project(self, request: WireRequest, decision: WireMutationDecision) -> WireMutationProjection: ...


class PrivateWirePlanningRegistry:
    def __init__(self, projectors: Mapping[WireCapabilityIdentity, InterfaceWireMutationProjector]) -> None:
        if not isinstance(projectors, Mapping):
            raise TypeError("projectors must be a mapping")
        copied = {}
        for identity, projector in projectors.items():
            if not isinstance(identity, WireCapabilityIdentity) or not isinstance(projector, InterfaceWireMutationProjector):
                raise TypeError("projectors must map capability identities to projectors")
            copied[identity] = projector
        self._projectors = MappingProxyType(copied)

    def get(self, capability: WireProtocolCapability) -> InterfaceWireMutationProjector | None:
        return self._projectors.get(WireCapabilityIdentity.from_capability(capability))
