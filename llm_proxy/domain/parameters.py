"""Internal ownership of canonical and compatibility sampling parameters."""
from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any


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


def _thaw_json(value: JsonValue) -> Any:
    if isinstance(value, Mapping):
        return {key: _thaw_json(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw_json(item) for item in value]
    return value


@dataclass(frozen=True, slots=True)
class ParameterDefinition:
    name: str
    value_type: type | tuple[type, ...]
    validate: Callable[[Any], None]
    numeric_for_logging: bool = False

    def normalize(self, value: Any) -> JsonValue:
        if not isinstance(value, self.value_type) or isinstance(value, bool) and self.value_type is not bool:
            raise ValueError(f"{self.name} has invalid type")
        self.validate(value)
        return _freeze_json(value, self.name)


def _non_negative(value: Any) -> None:
    if value < 0:
        raise ValueError("must not be negative")


def _any(_: Any) -> None:
    pass


class ParameterCatalog:
    def __init__(self, definitions: tuple[ParameterDefinition, ...]) -> None:
        entries = {definition.name: definition for definition in definitions}
        if len(entries) != len(definitions):
            raise ValueError("duplicate canonical parameter identity")
        self._definitions = MappingProxyType(entries)

    def require(self, name: str) -> ParameterDefinition:
        try:
            return self._definitions[name]
        except KeyError as error:
            raise ValueError(f"unknown canonical parameter: {name}") from error

    def normalize(self, name: str, value: Any) -> JsonValue:
        return self.require(name).normalize(value)

    def is_numeric(self, name: str) -> bool:
        return self.require(name).numeric_for_logging

    @property
    def names(self) -> frozenset[str]:
        return frozenset(self._definitions)


CORE_PARAMETER_CATALOG = ParameterCatalog((
    ParameterDefinition("temperature", (int, float), _any, True),
    ParameterDefinition("top_p", (int, float), _any, True),
    ParameterDefinition("top_k", int, _non_negative, True),
    ParameterDefinition("min_p", (int, float), _any, True),
    ParameterDefinition("repeat_penalty", (int, float), _any, True),
    ParameterDefinition("repeat_last_n", int, _non_negative, True),
    ParameterDefinition("max_tokens", int, _non_negative, True),
    ParameterDefinition("n", int, _non_negative, True),
    ParameterDefinition("stop_sequences", tuple, lambda value: None),
))


@dataclass(frozen=True, slots=True)
class CompatibilityParameters:
    values: Mapping[str, JsonValue] = field(default_factory=dict)
    removed: frozenset[str] = frozenset()

    def __post_init__(self) -> None:
        if not all(isinstance(key, str) for key in self.values):
            raise TypeError("compatibility parameter keys must be strings")
        if not all(isinstance(key, str) for key in self.removed):
            raise TypeError("removed compatibility parameter keys must be strings")
        if set(self.values).intersection(self.removed):
            raise ValueError("compatibility parameters cannot be assigned and removed")
        object.__setattr__(self, "values", MappingProxyType({key: _freeze_json(value, "compatibility parameters") for key, value in self.values.items()}))
        object.__setattr__(self, "removed", frozenset(self.removed))


@dataclass(frozen=True, slots=True)
class ParameterOverlay:
    canonical: Mapping[str, JsonValue] = field(default_factory=dict)
    compatibility: Mapping[str, JsonValue] = field(default_factory=dict)
    remove_canonical: frozenset[str] = frozenset()
    remove_compatibility: frozenset[str] = frozenset()

    def __post_init__(self) -> None:
        if set(self.canonical).intersection(self.remove_canonical) or set(self.compatibility).intersection(self.remove_compatibility):
            raise ValueError("parameter overlay cannot assign and remove the same name")


@dataclass(frozen=True, slots=True)
class EffectiveParameters:
    canonical: Mapping[str, JsonValue] = field(default_factory=dict)
    compatibility: CompatibilityParameters = field(default_factory=CompatibilityParameters)
    catalog: ParameterCatalog = CORE_PARAMETER_CATALOG

    def __post_init__(self) -> None:
        object.__setattr__(self, "canonical", MappingProxyType({name: self.catalog.normalize(name, value) for name, value in self.canonical.items()}))
        if not isinstance(self.compatibility, CompatibilityParameters):
            raise TypeError("compatibility must be CompatibilityParameters")

    def apply(self, overlay: ParameterOverlay) -> "EffectiveParameters":
        if not isinstance(overlay, ParameterOverlay):
            raise TypeError("overlay must be ParameterOverlay")
        canonical = dict(self.canonical)
        compatibility = dict(self.compatibility.values)
        removed_canonical = set(overlay.remove_canonical)
        removed_compatibility = set(self.compatibility.removed).union(overlay.remove_compatibility)
        for name in removed_canonical:
            self.catalog.require(name)
            canonical.pop(name, None)
        for name in removed_compatibility:
            compatibility.pop(name, None)
        for name, value in overlay.canonical.items():
            canonical[name] = self.catalog.normalize(name, value)
            removed_canonical.discard(name)
        for name, value in overlay.compatibility.items():
            compatibility[name] = _freeze_json(value, "compatibility parameters")
            removed_compatibility.discard(name)
        return EffectiveParameters(canonical, CompatibilityParameters(compatibility, frozenset(removed_compatibility)), self.catalog)

    @classmethod
    def from_legacy(cls, parameters: Any, catalog: ParameterCatalog = CORE_PARAMETER_CATALOG) -> "EffectiveParameters":
        canonical = {name: getattr(parameters, name) for name in catalog.names if name != "stop_sequences" and hasattr(parameters, name) and getattr(parameters, name) is not None}
        if parameters.stop_sequences:
            canonical["stop_sequences"] = tuple(parameters.stop_sequences)
        return cls(canonical, CompatibilityParameters(dict(parameters.extra)), catalog)

    def to_legacy(self):
        from .requests import SamplingParameters
        values = dict(self.canonical)
        stop = tuple(values.pop("stop_sequences", ()))
        legacy_fields = SamplingParameters.__dataclass_fields__
        compatibility = _thaw_json(self.compatibility.values)
        compatibility.update({name: _thaw_json(value) for name, value in values.items() if name not in legacy_fields})
        return SamplingParameters(
            stop_sequences=stop,
            extra=compatibility,
            **{name: value for name, value in values.items() if name in legacy_fields},
        )
