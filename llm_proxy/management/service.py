from __future__ import annotations

from typing import Any

from llm_proxy import __version__
from llm_proxy.configuration.revision import configuration_mapping
from llm_proxy.configuration.store import ConfigurationStore
from llm_proxy.application.provider_registry import ProviderExtensionRegistry, ProviderGatewayRegistry
from llm_proxy.application.runtime import ActivationPolicy
from llm_proxy.provider_extensions import ProviderHealth

from .configuration_service import ConfigurationManagementService


class ManagementQueryService:
    def __init__(self, store: ConfigurationStore, extensions: ProviderExtensionRegistry, gateways: ProviderGatewayRegistry, configuration: ConfigurationManagementService, command_registry=None, activation_policy: ActivationPolicy | None = None) -> None:
        self._store, self._extensions, self._gateways, self._configuration = store, extensions, gateways, configuration
        self._command_registry = command_registry
        self._activation_policy = activation_policy or ActivationPolicy()

    def status(self) -> dict[str, Any]:
        config = self._store.snapshot.config
        return {
            "gateway_version": __version__,
            "active_revision": self._configuration.active_revision,
            "persisted_revision": self._configuration.persisted_revision,
            "restart_required": self._configuration.active_revision != self._configuration.persisted_revision,
            "interfaces": sorted(name.value for name, value in config.interfaces.items() if value.enabled),
            "management_bind": "loopback" if config.server.host in {"127.0.0.1", "::1", "localhost"} else "remote",
            "last_successful_activation": self._store.snapshot.loaded_at,
            "last_rejected_reload": self._store.last_rejected_reload,
        }

    def configuration(self) -> dict[str, Any]:
        document = configuration_mapping(self._store.snapshot.config)
        for provider in document["providers"].values():
            provider["config"] = {"configured": bool(provider["config"])}
        for extension in document.get("extensions", {}).values():
            extension["config"] = {"configured": bool(extension["config"])}
        for interface in document["interfaces"].values():
            if interface.get("authentication"):
                interface["authentication"]["token"] = None
        return {"revision": self._configuration.active_revision, "configuration": document}

    def extensions(self) -> dict[str, Any]:
        entries = []
        for extension_id, extension in self._extensions.entries.items():
            entries.append({"extension_id": extension_id, "display_name": extension.metadata.display_name, "package_version": extension.metadata.package_version, "capabilities": extension.capabilities.__dict__ if hasattr(extension.capabilities, "__dict__") else {name: getattr(extension.capabilities, name) for name in extension.capabilities.__dataclass_fields__}})
        return {"extensions": entries, "diagnostics": [{"source": item.source, "message": item.message, "extension_id": item.extension_id} for item in self._extensions.diagnostics[:20]]}

    def providers(self) -> dict[str, Any]:
        return {"providers": self._provider_inventory()}

    async def configuration_administration(self) -> dict[str, Any]:
        config = self._store.snapshot.config
        lifecycle = self._activation_policy.administration_lifecycle(config)
        providers = []
        for provider in self._provider_inventory():
            provider["health"] = await self._provider_health(provider)
            providers.append(provider)
        return {
            "revision": self._configuration.active_revision,
            "server": {"host": config.server.host, "port": config.server.port, "timeout_seconds": config.server.timeout_seconds, "config_reload_seconds": config.server.config_reload_seconds, "log_level": config.server.log_level, **lifecycle["server"]},
            "management": {"enabled": config.management.enabled, "allow_remote": config.management.allow_remote, "command_timeout_seconds": config.management.command_timeout_seconds, **lifecycle["management"]},
            "interfaces": [{"name": name.value, "enabled": value.enabled, "expose_thinking": value.expose_thinking, "authentication_configured": value.authentication is not None and value.authentication.token is not None, **lifecycle["interfaces"]} for name, value in sorted(config.interfaces.items(), key=lambda item: item[0].value)],
            "providers": providers,
            "extensions": self.extensions()["extensions"],
            "diagnostics": self.extensions()["diagnostics"],
        }

    def _provider_inventory(self) -> list[dict[str, Any]]:
        values = []
        for name, provider in sorted(self._store.snapshot.config.providers.items()):
            extension = self._extensions.get(provider.extension_id)
            capabilities = {field: getattr(extension.capabilities, field) for field in extension.capabilities.__dataclass_fields__} if extension else {}
            provider_lifecycle = self._activation_policy.administration_lifecycle(self._store.snapshot.config)["providers"]
            values.append({"name": name, "extension_id": provider.extension_id, "enabled": provider.enabled, "active": name in self._gateways.instance_names, "configured": bool(provider.config), **provider_lifecycle, "command_count": self._command_registry.count(name) if self._command_registry else 0, "capabilities": capabilities})
        return values

    async def _provider_health(self, provider: dict[str, Any]) -> dict[str, Any]:
        if not provider["capabilities"].get("health"):
            return {"available": False, "status": "unavailable", "diagnostic": "health capability is unavailable"}
        try:
            result = await self._gateways.health(provider["name"])
            if result is None:
                raise TypeError()
            return {"available": True, "status": result.status.value, "diagnostic": result.diagnostic}
        except Exception:
            return {"available": False, "status": "unavailable", "diagnostic": "health check unavailable"}

    def models(self) -> dict[str, Any]:
        return {"models": [{"name": name, "upstream_model": item.upstream_model, "provider": item.provider, "interfaces": sorted(value.value for value in item.interfaces), "aliases": {key.value: list(value) for key, value in item.aliases.by_interface.items()}} for name, item in sorted(self._store.snapshot.config.models.items())]}
