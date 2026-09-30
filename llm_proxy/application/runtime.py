from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from llm_proxy.configuration.loader import ConfigurationError
from llm_proxy.configuration.models import GatewayConfig
from llm_proxy.extensions.kernel import ExtensionInstanceRegistry


@dataclass(frozen=True, slots=True)
class RuntimeDependencies:
    provider_gateways: object | None = None
    extension_instances: ExtensionInstanceRegistry | None = None

    def validate_capabilities(self, config: GatewayConfig) -> None:
        if self.extension_instances is None:
            return
        for policy in config.policies:
            for binding in policy.actions.capabilities:
                try:
                    self.extension_instances.validate_arguments(
                        binding.instance, binding.family, binding.version,
                        binding.capability, binding.arguments,
                    )
                except Exception as error:
                    raise ConfigurationError(
                        f"policies.{policy.name}.actions.capabilities.{binding.id}",
                        "capability arguments rejected",
                    ) from error


@dataclass(frozen=True, slots=True)
class ActivationDecision:
    restart_required: bool
    changed_groups: tuple[str, ...] = ()
    message: str = "provider, extension, or management configuration changed; restart is required"


class ActivationPolicy:
    _SERVER_RESTART = ("host", "port", "config_reload_seconds", "log_level", "transport_debug", "file_logging")
    _SERVER_RELOAD = ("payload_trace_mode", "payload_trace_include_content", "timeout_seconds")
    _INTERFACE_RESTART = ("enabled",)
    _INTERFACE_RELOAD = ("authentication", "expose_thinking")

    @staticmethod
    def _provider_identity(config: GatewayConfig) -> dict[str, tuple[Any, ...]]:
        providers = config.providers if hasattr(config, "providers") else config
        return {
            name: (item.name, item.extension_id, item.enabled, dict(item.config))
            for name, item in providers.items()
        }

    @staticmethod
    def _extension_identity(config: GatewayConfig) -> dict[str, tuple[Any, ...]]:
        return {
            name: (item.name, item.extension_id, item.enabled, dict(item.config))
            for name, item in config.extensions.items()
        }

    def provider_identity(self, config: GatewayConfig) -> dict[str, tuple[Any, ...]]:
        return self._provider_identity(config)

    def classify(self, active: GatewayConfig, candidate: GatewayConfig) -> ActivationDecision:
        changed: list[str] = []
        if self._provider_identity(active) != self._provider_identity(candidate):
            changed.append("providers")
        if self._extension_identity(active) != self._extension_identity(candidate):
            changed.append("extensions")
        if active.management != candidate.management:
            changed.append("management")
        if any(getattr(active.server, field) != getattr(candidate.server, field) for field in self._SERVER_RESTART):
            changed.append("server")
        interface_changed = False
        for name in set(active.interfaces) | set(candidate.interfaces):
            old, new = active.interfaces.get(name), candidate.interfaces.get(name)
            if old is None or new is None:
                interface_changed = True
                break
            if any(getattr(old, field) != getattr(new, field) for field in self._INTERFACE_RESTART):
                interface_changed = True
                break
        if interface_changed:
            changed.append("interfaces")
        return ActivationDecision(bool(changed), tuple(changed))

    def administration_lifecycle(self, config: GatewayConfig) -> dict[str, Any]:
        return {
            "server": {
                "lifecycle": "mixed",
                "restart_required_fields": list(self._SERVER_RESTART),
                "reload_safe_fields": list(self._SERVER_RELOAD),
            },
            "interfaces": {
                "lifecycle": "mixed",
                "restart_required_fields": list(self._INTERFACE_RESTART),
                "reload_safe_fields": list(self._INTERFACE_RELOAD),
            },
            "management": {"lifecycle": "restart_required"},
            "providers": {"lifecycle": "mixed", "reload_safe_fields": ["listing_enabled"]},
        }
