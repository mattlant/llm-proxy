from __future__ import annotations

import copy
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Mapping, Protocol, TypeAlias

EXTENSION_SDK_VERSION = "1.0.0"
JsonValue: TypeAlias = None | bool | int | float | str | tuple["JsonValue", ...] | Mapping[str, "JsonValue"]


def freeze_json(value: object) -> JsonValue:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, (list, tuple)):
        return tuple(freeze_json(item) for item in value)
    if isinstance(value, Mapping):
        if not all(isinstance(key, str) for key in value):
            raise ValueError("JSON mappings require string keys")
        return MappingProxyType({key: freeze_json(item) for key, item in value.items()})
    raise ValueError("value must be JSON-compatible")


def frozen_mapping(value: Mapping[str, object]) -> Mapping[str, JsonValue]:
    frozen = freeze_json(copy.deepcopy(dict(value)))
    assert isinstance(frozen, Mapping)
    return frozen


def _nonblank(value: object, name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be non-blank")


@dataclass(frozen=True, slots=True)
class ExtensionMetadata:
    extension_id: str
    display_name: str
    package_name: str
    package_version: str

    def __post_init__(self) -> None:
        for name in ("extension_id", "display_name", "package_name", "package_version"):
            _nonblank(getattr(self, name), name)


@dataclass(frozen=True, slots=True)
class ExtensionSdkCompatibility:
    minimum: str
    maximum: str | None = None

    def __post_init__(self) -> None:
        _nonblank(self.minimum, "minimum")
        if self.maximum is not None:
            _nonblank(self.maximum, "maximum")


@dataclass(frozen=True, slots=True)
class ExtensionInstanceConfig:
    instance_name: str
    extension_id: str
    config: Mapping[str, JsonValue] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _nonblank(self.instance_name, "instance_name")
        _nonblank(self.extension_id, "extension_id")
        if not isinstance(self.config, Mapping):
            raise TypeError("config must be a mapping")
        object.__setattr__(self, "config", frozen_mapping(self.config))


@dataclass(frozen=True, slots=True)
class CapabilityRegistration:
    family: str
    family_version: int
    name: str
    implementation: object

    def __post_init__(self) -> None:
        _nonblank(self.family, "family")
        _nonblank(self.name, "name")
        if isinstance(self.family_version, bool) or not isinstance(self.family_version, int) or self.family_version <= 0:
            raise ValueError("family_version must be a positive integer")


class ExtensionInstance(Protocol):
    @property
    def registrations(self) -> tuple[CapabilityRegistration, ...]: ...
    async def aclose(self) -> None: ...


class ExtensionFactory(Protocol):
    def create(self, config: ExtensionInstanceConfig) -> ExtensionInstance: ...


class Extension(Protocol):
    metadata: ExtensionMetadata
    compatibility: ExtensionSdkCompatibility
    factory: ExtensionFactory
