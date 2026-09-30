"""Startup-only provider extension discovery and configured instance activation."""
from __future__ import annotations

import asyncio
import threading
from dataclasses import dataclass
from enum import Enum
from importlib import metadata
from inspect import isawaitable
from types import MappingProxyType
from typing import Awaitable, Callable, Iterable, Mapping

from llm_proxy.configuration.models import ProviderConfig
from llm_proxy.domain.errors import ProviderProtocolError
from llm_proxy.provider_extensions import CompletionGateway, ProviderCapabilities, ProviderExtension, ProviderExtensionMetadata, ProviderHealth, ProviderHealthGateway, ProviderInstanceConfig, ProviderListedModel, ProviderListingError, ProviderListingFailureCategory, ProviderModelListingGateway, ProviderSemanticDeclaration, SdkCompatibility, WireGateway, provider_semantic_declaration, validate_sdk_compatibility


ENTRY_POINT_GROUP = "llm_proxy.providers"


@dataclass(frozen=True, slots=True)
class ProviderDiagnostic:
    source: str
    message: str
    extension_id: str | None = None


class ProviderRegistryError(RuntimeError):
    pass


try:
    from enum import StrEnum
except ImportError:  # pragma: no cover - compatibility for Python 3.10
    class StrEnum(str, Enum):
        pass


class ProviderExtensionOrigin(StrEnum):
    BUILT_IN = "built_in"
    ENTRY_POINT = "entry_point"
    UNKNOWN = "unknown"


@dataclass(frozen=True, slots=True)
class ProviderRuntimeSupport:
    origin: ProviderExtensionOrigin
    declaration: ProviderSemanticDeclaration | None = None


class ProviderExtensionRegistry:
    def __init__(self, extensions: Iterable[tuple], diagnostics: Iterable[ProviderDiagnostic] = ()) -> None:
        entries: dict[str, ProviderExtension] = {}
        origins: dict[str, ProviderExtensionOrigin] = {}
        declarations: dict[str, ProviderSemanticDeclaration | None] = {}
        failures = list(diagnostics)
        identities: set[str] = set()
        normalized_extensions = []
        for item in extensions:
            if len(item) == 2:
                source, extension = item
                origin = ProviderExtensionOrigin.UNKNOWN
            elif len(item) == 3:
                source, extension, origin = item
                if not isinstance(origin, ProviderExtensionOrigin):
                    raise TypeError("provider extension origin must be ProviderExtensionOrigin")
            else:
                raise ValueError("provider extension entries must contain source, extension, and optional origin")
            normalized_extensions.append((source, extension, origin))
        for source, extension, origin in sorted(normalized_extensions, key=lambda value: value[0]):
            validated = _extension_values(extension)
            if validated is None:
                failures.append(ProviderDiagnostic(source, "invalid provider extension contract"))
                continue
            metadata, compatibility_value, capabilities, factory = validated
            extension_id = metadata.extension_id
            if extension_id in identities:
                raise ProviderRegistryError(f"duplicate provider extension ID '{extension_id}'")
            identities.add(extension_id)
            compatibility = validate_sdk_compatibility(compatibility_value)
            if not compatibility.compatible:
                failures.append(ProviderDiagnostic(source, compatibility.diagnostic, extension_id))
                continue
            try:
                factory_create = getattr(factory, "create", None)
            except Exception:
                factory_create = None
            if not callable(factory_create):
                failures.append(ProviderDiagnostic(source, "invalid provider extension contract", extension_id))
                continue
            try:
                declaration = provider_semantic_declaration(extension)
            except (TypeError, ValueError) as error:
                failures.append(ProviderDiagnostic(source, f"invalid provider semantic declaration: {type(error).__name__}", extension_id))
                continue
            if extension_id in entries:
                raise ProviderRegistryError(f"duplicate provider extension ID '{extension_id}'")
            entries[extension_id] = extension
            origins[extension_id] = origin
            declarations[extension_id] = declaration
        self._entries = MappingProxyType(dict(sorted(entries.items())))
        self._origins = MappingProxyType(origins)
        self._declarations = MappingProxyType(declarations)
        self._diagnostics = tuple(failures)

    @property
    def entries(self) -> Mapping[str, ProviderExtension]:
        return self._entries

    @property
    def diagnostics(self) -> tuple[ProviderDiagnostic, ...]:
        return self._diagnostics

    def get(self, extension_id: str) -> ProviderExtension | None:
        return self._entries.get(extension_id)

    def support(self, extension_id: str) -> ProviderRuntimeSupport:
        return ProviderRuntimeSupport(self._origins.get(extension_id, ProviderExtensionOrigin.UNKNOWN), self._declarations.get(extension_id))

    def origin(self, extension_id: str) -> ProviderExtensionOrigin:
        return self._origins.get(extension_id, ProviderExtensionOrigin.UNKNOWN)

    def declaration(self, extension_id: str) -> ProviderSemanticDeclaration | None:
        return self._declarations.get(extension_id)


def discover_extensions(
    builtins: Iterable[tuple[str, object]],
    entry_points: Callable[[], Iterable[object]] | None = None,
) -> ProviderExtensionRegistry:
    enumerate_points = entry_points or _provider_entry_points
    discovered = [(source, extension, ProviderExtensionOrigin.BUILT_IN) for source, extension in builtins]
    diagnostics: list[ProviderDiagnostic] = []
    for point in sorted(enumerate_points(), key=lambda item: (getattr(item, "name", ""), getattr(item, "value", ""))):
        source = f"entry point {getattr(point, 'name', '<unknown>')}"
        try:
            discovered.append((source, point.load(), ProviderExtensionOrigin.ENTRY_POINT))
        except Exception as error:  # installed extensions are isolated until configured
            diagnostics.append(ProviderDiagnostic(source, f"load failed: {type(error).__name__}"))
    return ProviderExtensionRegistry(discovered, diagnostics)


def _provider_entry_points() -> Iterable[object]:
    points = metadata.entry_points()
    return points.select(group=ENTRY_POINT_GROUP) if hasattr(points, "select") else points.get(ENTRY_POINT_GROUP, ())


def _extension_values(value: object):
    try:
        metadata = value.metadata
        compatibility = value.compatibility
        capabilities = value.capabilities
        factory = value.factory
    except Exception:
        return None
    if not isinstance(metadata, ProviderExtensionMetadata) or not isinstance(compatibility, SdkCompatibility) or not isinstance(capabilities, ProviderCapabilities):
        return None
    return metadata, compatibility, capabilities, factory


class ProviderGatewayRegistry:
    def __init__(self, gateways: Mapping[str, CompletionGateway], providers: Mapping[str, ProviderConfig] | None = None, capabilities: Mapping[str, ProviderCapabilities] | None = None, health_checks: Mapping[str, Callable[[], Awaitable[ProviderHealth]]] | None = None, provider_identity: Mapping | None = None, support: Mapping[str, ProviderRuntimeSupport] | None = None) -> None:
        self._gateways = MappingProxyType(dict(gateways))
        self._providers = MappingProxyType(dict(providers)) if providers is not None else None
        self._provider_identity = provider_identity
        self._capabilities = MappingProxyType(dict(capabilities or {}))
        self._health_checks = MappingProxyType(dict(health_checks or {}))
        self._support = MappingProxyType(dict(support or {}))

    def get(self, instance_name: str) -> CompletionGateway:
        try:
            return self._gateways[instance_name]
        except KeyError as error:
            raise ProviderProtocolError("configured provider instance is unavailable", provider=instance_name) from error

    def capabilities(self, instance_name: str) -> ProviderCapabilities:
        return self._capabilities.get(instance_name, ProviderCapabilities())

    def support(self, instance_name: str) -> ProviderRuntimeSupport:
        return self._support.get(instance_name, ProviderRuntimeSupport(ProviderExtensionOrigin.UNKNOWN))

    def get_wire(self, instance_name: str) -> WireGateway | None:
        gateway = self._gateways.get(instance_name)
        return gateway if isinstance(gateway, WireGateway) else None

    async def health(self, instance_name: str) -> ProviderHealth | None:
        check = self._health_checks.get(instance_name)
        if check is None:
            return None
        result = await check()
        if not isinstance(result, ProviderHealth):
            raise ProviderRegistryError("provider health returned invalid result")
        return result

    @property
    def instance_names(self) -> tuple[str, ...]:
        return tuple(self._gateways)

    def assert_matches(self, provider_identity: Mapping) -> None:
        if self._provider_identity is not None and provider_identity != self._provider_identity:
            raise ProviderRegistryError("provider instance configuration changed; restart is required")

    async def list_models(self, instance_name: str, timeout_seconds: int) -> tuple[ProviderListedModel, ...]:
        capability = self.capabilities(instance_name)
        gateway = self.get(instance_name)
        if not capability.model_listing or not isinstance(gateway, ProviderModelListingGateway):
            raise ProviderListingError(instance_name, ProviderListingFailureCategory.INVALID_RESPONSE, "provider does not support model listing")
        try:
            async with asyncio.timeout(timeout_seconds):
                result = await gateway.list_models()
        except TimeoutError as error:
            raise ProviderListingError(instance_name, ProviderListingFailureCategory.TIMEOUT, "provider model listing timed out") from error
        except ProviderListingError:
            raise
        except Exception as error:
            raise ProviderListingError(instance_name, ProviderListingFailureCategory.UNAVAILABLE, "provider model listing failed") from error
        if not isinstance(result, tuple) or not all(isinstance(item, ProviderListedModel) for item in result):
            raise ProviderListingError(instance_name, ProviderListingFailureCategory.INVALID_RESPONSE, "provider model listing returned invalid records")
        names = [item.upstream_model for item in result]
        if len(names) != len(set(names)):
            raise ProviderListingError(instance_name, ProviderListingFailureCategory.INVALID_RESPONSE, "provider model listing returned duplicate records")
        return result

    async def aclose(self) -> None:
        for gateway in self._gateways.values():
            await _close_gateway(gateway)


async def _close_gateway(gateway: object) -> None:
    close = getattr(gateway, "aclose", None)
    if not callable(close):
        close = getattr(gateway, "close", None)
    if not callable(close):
        return
    result = close()
    if isawaitable(result):
        await result


def _rollback_gateways(gateways: Iterable[object]) -> None:
    async def cleanup() -> None:
        for gateway in gateways:
            try:
                await _close_gateway(gateway)
            except Exception:
                pass

    try:
        asyncio.get_running_loop()
    except RuntimeError:
        asyncio.run(cleanup())
        return
    def run() -> None:
        try:
            asyncio.run(cleanup())
        except BaseException:
            pass
    thread = threading.Thread(target=run)
    thread.start()
    thread.join()


def activate_provider_instances(registry: ProviderExtensionRegistry, providers: Mapping[str, ProviderConfig], identity_policy=None) -> ProviderGatewayRegistry:
    created: dict[str, CompletionGateway] = {}
    capabilities: dict[str, ProviderCapabilities] = {}
    health_checks: dict[str, Callable[[], Awaitable[ProviderHealth]]] = {}
    support: dict[str, ProviderRuntimeSupport] = {}
    try:
        for name, provider in providers.items():
            if not provider.enabled:
                continue
            extension = registry.get(provider.extension_id)
            if extension is None:
                diagnostic = next((item.message for item in registry.diagnostics if item.extension_id == provider.extension_id), None)
                suffix = f": {diagnostic}" if diagnostic else ""
                raise ProviderRegistryError(f"provider instance '{name}' requires unavailable extension '{provider.extension_id}'{suffix}")
            runtime_support = registry.support(provider.extension_id)
            try:
                declaration = provider_semantic_declaration(extension)
            except (TypeError, ValueError) as error:
                raise ProviderRegistryError(f"provider instance '{name}' has invalid semantic declaration: {error}") from error
            runtime_support = ProviderRuntimeSupport(runtime_support.origin, declaration)
            gateway = extension.factory.create(ProviderInstanceConfig(name, provider.extension_id, provider.config))
            if extension.capabilities.model_listing and not isinstance(gateway, ProviderModelListingGateway):
                raise ProviderRegistryError(f"provider instance '{name}' advertises model listing but gateway does not implement it")
            if extension.capabilities.health:
                health = getattr(gateway, "health", None)
                if callable(health):
                    health_checks[name] = health
                else:
                    extension_health = getattr(extension, "health", None)
                    if not callable(extension_health):
                        raise ProviderRegistryError(f"provider instance '{name}' advertises health but has no health operation")
                    instance_config = ProviderInstanceConfig(name, provider.extension_id, provider.config)
                    health_checks[name] = lambda health=extension_health, config=instance_config: health(config)
            from llm_proxy.management.command_registry import validate_command_executor
            validate_command_executor(gateway, extension.capabilities.management_commands)
            created[name] = gateway
            capabilities[name] = extension.capabilities
            support[name] = runtime_support
    except Exception as error:
        _rollback_gateways(created.values())
        if isinstance(error, ProviderRegistryError):
            raise
        raise ProviderRegistryError(f"provider activation failed: {type(error).__name__}: {error}") from error
    if identity_policy is None:
        from llm_proxy.application.runtime import ActivationPolicy
        identity_policy = ActivationPolicy().provider_identity
    return ProviderGatewayRegistry(created, providers, capabilities, health_checks, identity_policy(providers), support)
