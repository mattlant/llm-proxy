from __future__ import annotations

from llm_proxy.configuration.loader import ConfigurationError

from .models import GatewayConfig, InterfaceName, ModelAccessMode


class ConfigurationValidator:
    def validate(self, config: GatewayConfig) -> None:
        enabled_interfaces = {
            interface for interface, value in config.interfaces.items() if value.enabled
        }
        if not enabled_interfaces:
            raise ConfigurationError("interfaces", "at least one interface must be enabled")

        enabled_providers = {
            name for name, provider in config.providers.items() if provider.enabled
        }
        if not enabled_providers:
            raise ConfigurationError("providers", "at least one provider must be enabled")
        if not config.models and config.model_access.mode is not ModelAccessMode.PROVIDER_PASSTHROUGH:
            raise ConfigurationError("models", "at least one model profile is required")

        self._validate_model_names(config)
        self._validate_models(config, enabled_interfaces, enabled_providers)
        self._validate_aliases(config, enabled_interfaces)
        self._validate_policies(config, enabled_interfaces, enabled_providers)
        seen = set()
        for index, policy in enumerate(config.policies):
            for binding in policy.actions.capabilities:
                if binding.id in seen: raise ConfigurationError(f"policies[{index}].actions.capabilities", f"duplicate capability binding '{binding.id}'")
                seen.add(binding.id)
                if binding.instance not in config.extensions or not config.extensions[binding.instance].enabled: raise ConfigurationError(f"policies[{index}].actions.capabilities", "references unknown or disabled extension instance")
                if (binding.family, binding.version) != ("side_effect", 1): raise ConfigurationError(f"policies[{index}].actions.capabilities", "unsupported capability family")

    @staticmethod
    def _validate_model_names(config: GatewayConfig) -> None:
        seen: dict[str, str] = {}
        for name, profile in config.models.items():
            if "::" in name:
                raise ConfigurationError(f"models.{name}", "profile name must not contain '::'")
            folded = name.casefold()
            if folded in seen:
                raise ConfigurationError(
                    f"models.{name}",
                    f"profile name collides with '{seen[folded]}' under case-insensitive lookup",
                )
            seen[folded] = profile.name
        provider_names: dict[str, str] = {}
        for name in config.providers:
            if "::" in name:
                raise ConfigurationError(f"providers.{name}", "provider name must not contain '::'")
            folded = name.casefold()
            if folded in provider_names:
                raise ConfigurationError(f"providers.{name}", f"provider name collides with '{provider_names[folded]}' under case-insensitive lookup")
            provider_names[folded] = name

    @staticmethod
    def _validate_models(
        config: GatewayConfig,
        enabled_interfaces: set[InterfaceName],
        enabled_providers: set[str],
    ) -> None:
        for name, profile in config.models.items():
            model_path = f"models.{name}"
            if profile.provider not in config.providers:
                raise ConfigurationError(
                    f"{model_path}.provider",
                    f"references unknown provider '{profile.provider}'",
                )
            if profile.provider not in enabled_providers:
                raise ConfigurationError(
                    f"{model_path}.provider",
                    f"references disabled provider '{profile.provider}'",
                )
            for interface in profile.interfaces:
                if interface not in config.interfaces:
                    raise ConfigurationError(
                        f"{model_path}.interfaces",
                        f"references unknown interface '{interface.value}'",
                    )
                if interface not in enabled_interfaces:
                    raise ConfigurationError(
                        f"{model_path}.interfaces",
                        f"references disabled interface '{interface.value}'",
                    )
            for interface in profile.aliases.by_interface:
                if interface not in profile.interfaces:
                    raise ConfigurationError(
                        f"{model_path}.aliases.{interface.value}",
                        "alias interface is not exposed by this profile",
                    )

    @staticmethod
    def _validate_aliases(config: GatewayConfig, enabled_interfaces: set[InterfaceName]) -> None:
        aliases_by_interface: dict[InterfaceName, dict[str, str]] = {}
        profile_names_by_interface: dict[InterfaceName, dict[str, str]] = {
            interface: {}
            for interface in enabled_interfaces
        }
        for name, profile in config.models.items():
            for interface in profile.interfaces:
                profile_names_by_interface.setdefault(interface, {})[name.casefold()] = name

        for name, profile in config.models.items():
            for interface, aliases in profile.aliases.by_interface.items():
                authorities = aliases_by_interface.setdefault(interface, {})
                for alias in aliases:
                    if "::" in alias:
                        raise ConfigurationError(
                            f"models.{name}.aliases.{interface.value}",
                            "alias must not contain '::'",
                        )
                    folded = alias.casefold()
                    if folded in authorities:
                        raise ConfigurationError(
                            f"models.{name}.aliases.{interface.value}",
                            f"alias '{alias}' collides with profile '{authorities[folded]}' under case-insensitive lookup",
                        )
                    if folded in profile_names_by_interface.get(interface, {}):
                        target = profile_names_by_interface[interface][folded]
                        raise ConfigurationError(
                            f"models.{name}.aliases.{interface.value}",
                            f"alias '{alias}' shadows visible profile '{target}'",
                        )
                    authorities[folded] = name

    @staticmethod
    def _validate_policies(
        config: GatewayConfig,
        enabled_interfaces: set[InterfaceName],
        enabled_providers: set[str],
    ) -> None:
        profile_names = {name.casefold() for name in config.models}
        aliases = {
            alias.casefold()
            for profile in config.models.values()
            for values in profile.aliases.by_interface.values()
            for alias in values
        }
        for index, policy in enumerate(config.policies):
            path = f"policies[{index}]"
            for interface in policy.match.interfaces:
                if interface not in config.interfaces:
                    raise ConfigurationError(f"{path}.match.interfaces", f"references unknown interface '{interface.value}'")
                if interface not in enabled_interfaces:
                    raise ConfigurationError(f"{path}.match.interfaces", f"references disabled interface '{interface.value}'")
            if policy.actions.model is not None and policy.actions.model.casefold() not in profile_names | aliases:
                raise ConfigurationError(f"{path}.actions.model", f"references unknown model '{policy.actions.model}'")
            if policy.actions.provider is not None:
                if policy.actions.provider not in config.providers:
                    raise ConfigurationError(f"{path}.actions.provider", f"references unknown provider '{policy.actions.provider}'")
                if policy.actions.provider not in enabled_providers:
                    raise ConfigurationError(f"{path}.actions.provider", f"references disabled provider '{policy.actions.provider}'")
