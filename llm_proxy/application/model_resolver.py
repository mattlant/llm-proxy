from __future__ import annotations

from dataclasses import dataclass, replace

from llm_proxy.configuration.models import (
    GatewayConfig,
    InterfaceName,
    ModelCompatibility,
    ModelAccessMode,
    StructuredOutputMode,
)
from llm_proxy.domain.requests import SamplingParameters

from .model_registry import ModelRegistry
from .errors import ModelNotFoundError
from .provider_registry import ProviderGatewayRegistry


@dataclass(frozen=True, slots=True)
class QualifiedModelIdentity:
    provider_instance_name: str
    upstream_model: str


def parse_qualified_model_identity(requested_model: str) -> QualifiedModelIdentity:
    if not isinstance(requested_model, str) or "::" not in requested_model:
        raise ModelNotFoundError("unknown", requested_model if isinstance(requested_model, str) else "")
    provider, upstream = requested_model.split("::", 1)
    if not provider or not upstream:
        raise ModelNotFoundError("unknown", requested_model)
    return QualifiedModelIdentity(provider, upstream)


@dataclass(frozen=True, slots=True)
class ResolvedModelExecution:
    requested_model: str
    canonical_model: str
    upstream_model: str
    interface: InterfaceName
    provider_instance_name: str
    parameters: SamplingParameters
    compatibility: ModelCompatibility


class ModelResolver:
    def __init__(self, registry: ModelRegistry, config: GatewayConfig, provider_gateways: ProviderGatewayRegistry | None = None):
        self._registry = registry
        self._config = config
        self._provider_gateways = provider_gateways

    def resolve(self, interface: InterfaceName, requested_model: str) -> ResolvedModelExecution:
        try:
            reference = self._registry.resolve(interface, requested_model)
        except ModelNotFoundError:
            if self._config.model_access.mode is not ModelAccessMode.PROVIDER_PASSTHROUGH:
                raise
            identity = parse_qualified_model_identity(requested_model)
            provider = next((name for name in self._config.providers if name.casefold() == identity.provider_instance_name.casefold()), None)
            if provider is None or not self._config.providers[provider].enabled or self._provider_gateways is None or provider not in self._provider_gateways.instance_names:
                raise
            capabilities = self._provider_gateways.capabilities(provider)
            if not capabilities.completion:
                raise
            return ResolvedModelExecution(requested_model, f"{provider}::{identity.upstream_model}", identity.upstream_model, interface, provider, SamplingParameters(), ModelCompatibility(native_tools=capabilities.native_tools, structured_output=StructuredOutputMode.UNSUPPORTED))
        profile = reference.profile
        return ResolvedModelExecution(requested_model, reference.canonical_name, profile.upstream_model, interface, profile.provider, replace(profile.parameters), profile.compatibility)
