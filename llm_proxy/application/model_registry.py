from __future__ import annotations

from dataclasses import dataclass

from llm_proxy.configuration.models import GatewayConfig, InterfaceName, ModelProfile

from .errors import InterfaceDisabledError, ModelNotExposedError, ModelNotFoundError


@dataclass(frozen=True, slots=True)
class ModelReference:
    requested_name: str
    canonical_name: str
    profile: ModelProfile


class ModelRegistry:
    def __init__(self, config: GatewayConfig):
        self._config = config
        self._profiles_by_name = {
            name.casefold(): profile
            for name, profile in config.models.items()
        }

    def resolve(self, interface: InterfaceName, requested_model: str) -> ModelReference:
        self._require_enabled_interface(interface, requested_model)
        folded_requested = requested_model.casefold()
        profile = self._profiles_by_name.get(folded_requested)
        if profile is not None:
            if interface not in profile.interfaces:
                raise ModelNotExposedError(interface.value, requested_model)
            return ModelReference(requested_model, profile.name, profile)

        for candidate in self._config.models.values():
            if interface not in candidate.interfaces:
                continue
            aliases = candidate.aliases.by_interface.get(interface, ())
            if any(alias.casefold() == folded_requested for alias in aliases):
                return ModelReference(requested_model, candidate.name, candidate)

        if any(folded_requested == name.casefold() for name in self._config.models):
            raise ModelNotExposedError(interface.value, requested_model)
        raise ModelNotFoundError(interface.value, requested_model)

    def list_models(self, interface: InterfaceName) -> tuple[ModelProfile, ...]:
        self._require_enabled_interface(interface, "<listing>")
        return tuple(
            profile
            for profile in self._config.models.values()
            if interface in profile.interfaces
        )

    def aliases_for(self, interface: InterfaceName, profile_name: str) -> tuple[str, ...]:
        reference = self.resolve(interface, profile_name)
        return reference.profile.aliases.by_interface.get(interface, ())

    def _require_enabled_interface(self, interface: InterfaceName, requested_model: str) -> None:
        config = self._config.interfaces.get(interface)
        if config is None or not config.enabled:
            raise InterfaceDisabledError(interface.value, requested_model)
