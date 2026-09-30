from __future__ import annotations

import inspect
import logging
from dataclasses import dataclass
from importlib import metadata
from typing import Callable, Iterable, Mapping

from .compatibility import validate_extension_sdk_compatibility
from .contracts import CapabilityRegistration, Extension, ExtensionInstanceConfig

ENTRY_POINT_GROUP = "llm_proxy.extensions"
log = logging.getLogger(__name__)

class ExtensionRegistryError(RuntimeError): pass

@dataclass(frozen=True, slots=True)
class ExtensionDiagnostic:
    source: str
    category: str

@dataclass(frozen=True, slots=True)
class CapabilityFamilyAdapter:
    family: str
    version: int
    validate: Callable[[object], None]

class ExtensionCatalog:
    def __init__(self, entries: Mapping[str, Extension], diagnostics: tuple[ExtensionDiagnostic, ...] = ()) -> None:
        self.entries = dict(entries)
        self.diagnostics = diagnostics
    def get(self, extension_id: str) -> Extension:
        try: return self.entries[extension_id]
        except KeyError as error: raise ExtensionRegistryError(f"extension '{extension_id}' is unavailable") from error

def discover_extensions(builtins: Iterable[tuple[str, object]] = (), entry_points=None) -> ExtensionCatalog:
    raw = list(builtins)
    points = metadata.entry_points().select(group=ENTRY_POINT_GROUP) if entry_points is None else entry_points
    raw.extend((getattr(point, "name", "entry point"), point) for point in points)
    entries, diagnostics = {}, []
    for source, item in sorted(raw, key=lambda value: value[0]):
        try:
            extension = item.load() if hasattr(item, "load") else item
            compatibility = validate_extension_sdk_compatibility(extension.compatibility)
            if not compatibility.compatible:
                diagnostics.append(ExtensionDiagnostic(source, "incompatible")); continue
            extension_id = extension.metadata.extension_id
            if extension_id in entries: raise ExtensionRegistryError(f"duplicate extension ID '{extension_id}'")
            if not callable(extension.factory.create): raise TypeError("extension factory is invalid")
            entries[extension_id] = extension
        except ExtensionRegistryError: raise
        except Exception:
            diagnostics.append(ExtensionDiagnostic(source, "invalid"))
    return ExtensionCatalog(entries, tuple(diagnostics))

class ExtensionInstanceRegistry:
    def __init__(self, instances, registrations) -> None:
        self._instances, self._registrations, self._closed = tuple(instances), dict(registrations), False
    def get(self, instance: str, family: str, version: int, capability: str) -> object:
        try: return self._registrations[(instance, family, version, capability)]
        except KeyError as error: raise ExtensionRegistryError("capability is unavailable") from error
    def has(self, instance: str, family: str, version: int, capability: str) -> bool:
        return (instance, family, version, capability) in self._registrations
    def validate_arguments(self, instance: str, family: str, version: int, capability: str, arguments) -> None:
        self.get(instance, family, version, capability).validate_arguments(arguments)
    async def aclose(self) -> None:
        if self._closed: return
        self._closed = True
        for instance in reversed(self._instances):
            try: await instance.aclose()
            except Exception: log.warning("Extension cleanup failed: type=%s", type(instance).__name__)

async def activate_extension_instances(catalog: ExtensionCatalog, configs, families: Iterable[CapabilityFamilyAdapter]) -> ExtensionInstanceRegistry:
    families = tuple(families)
    adapters = {(family.family, family.version): family for family in families}
    if len(adapters) != len(families): raise ExtensionRegistryError("duplicate capability family adapter")
    instances, registrations = [], {}
    current = None
    try:
        for config in configs:
            if not config.enabled: continue
            current = catalog.get(config.extension_id).factory.create(ExtensionInstanceConfig(config.name, config.extension_id, config.config))
            for registration in current.registrations:
                adapter = adapters.get((registration.family, registration.family_version))
                if adapter is None: raise ExtensionRegistryError("unsupported capability family")
                adapter.validate(registration.implementation)
                key = (config.name, registration.family, registration.family_version, registration.name)
                if key in registrations: raise ExtensionRegistryError("duplicate capability registration")
                registrations[key] = registration.implementation
            instances.append(current)
            current = None
    except Exception:
        for instance in reversed(([current] if current is not None else []) + instances):
            try: await instance.aclose()
            except Exception: pass
        raise
    return ExtensionInstanceRegistry(instances, registrations)
