"""Protocol-preserving provider execution contracts."""
from __future__ import annotations

import copy
import json
from collections.abc import AsyncIterator, Mapping
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from llm_proxy.domain.messages import Message
from llm_proxy.domain.requests import SamplingParameters
from .execution import ProviderExecutionPolicy

HeaderPairs = tuple[tuple[bytes, bytes], ...]
JsonValue = object


@dataclass(frozen=True, slots=True)
class WireProtocolCapability:
    protocol_id: str
    version: str
    endpoint: str
    request_body: bool = True
    request_headers: bool = True
    response_body: bool = True
    response_headers: bool = True
    streaming: bool = True
    requires_stream_usage: bool = False

    def __post_init__(self) -> None:
        if not all(isinstance(getattr(self, name), str) and getattr(self, name).strip() for name in ("protocol_id", "version", "endpoint")):
            raise ValueError("wire capability identity must be non-blank")
        if not self.endpoint.startswith("/"):
            raise ValueError("wire capability endpoint must start with '/'")
        if not all(isinstance(getattr(self, name), bool) for name in self.__dataclass_fields__ if name not in {"protocol_id", "version", "endpoint"}):
            raise TypeError("wire capability flags must be booleans")


@dataclass(frozen=True, slots=True)
class WireRequestProjection:
    requested_model: str
    stream: bool
    parameters: SamplingParameters
    messages: tuple[Message, ...] = ()
    tool_names: tuple[str, ...] = ()
    metadata: Mapping[str, JsonValue] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if not isinstance(self.requested_model, str) or not self.requested_model.strip() or not isinstance(self.stream, bool):
            raise ValueError("wire projection model and stream are invalid")
        if not isinstance(self.parameters, SamplingParameters):
            raise TypeError("parameters must be SamplingParameters")
        object.__setattr__(self, "messages", tuple(self.messages))
        object.__setattr__(self, "tool_names", tuple(self.tool_names))
        object.__setattr__(self, "metadata", dict(self.metadata or {}))


@dataclass(frozen=True, slots=True)
class WireRequest:
    capability: WireProtocolCapability
    body: bytes
    headers: HeaderPairs
    payload: Mapping[str, JsonValue]
    projection: WireRequestProjection


@dataclass(frozen=True, slots=True)
class WirePatchOperation:
    pointer: str
    value: JsonValue
    create_missing_parent_object: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.pointer, str) or not self.pointer.startswith("/"):
            raise ValueError("wire patch pointer must be a JSON pointer")


@dataclass(frozen=True, slots=True)
class WirePatch:
    operations: tuple[WirePatchOperation, ...] = ()

    def apply(self, request: WireRequest) -> bytes:
        if not self.operations:
            return request.body
        value = copy.deepcopy(dict(request.payload))
        for operation in self.operations:
            parts = [part.replace("~1", "/").replace("~0", "~") for part in operation.pointer[1:].split("/")]
            target = value
            for index, part in enumerate(parts[:-1]):
                if not isinstance(target, dict):
                    raise ValueError(f"wire patch parent is not an object: {operation.pointer}")
                if part not in target:
                    if operation.create_missing_parent_object and index == len(parts) - 2:
                        target[part] = {}
                    else:
                        raise ValueError(f"wire patch parent is missing: {operation.pointer}")
                target = target[part]
            if not isinstance(target, dict):
                raise ValueError(f"wire patch parent is not an object: {operation.pointer}")
            target[parts[-1]] = operation.value
        return json.dumps(value, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


@dataclass(frozen=True, slots=True)
class WireExecution:
    request: WireRequest
    provider_instance_name: str
    upstream_model: str
    policy: ProviderExecutionPolicy
    patch: WirePatch = WirePatch()


@dataclass(frozen=True, slots=True)
class WireResponse:
    status_code: int
    headers: HeaderPairs
    body: bytes


class WireStream:
    def __init__(self, status_code: int, headers: HeaderPairs, iterator: AsyncIterator[bytes], close) -> None:
        self.status_code, self.headers, self._iterator, self._close = status_code, headers, iterator, close
        self._closed = False

    def __aiter__(self) -> AsyncIterator[bytes]:
        return self._iterator

    async def aclose(self) -> None:
        if not self._closed:
            self._closed = True
            await self._close()


@runtime_checkable
class WireGateway(Protocol):
    async def complete_wire(self, execution: WireExecution) -> WireResponse: ...
    async def stream_wire(self, execution: WireExecution) -> WireStream: ...
