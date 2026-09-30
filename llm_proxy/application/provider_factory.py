from __future__ import annotations

from .provider_registry import ProviderGatewayRegistry
from llm_proxy.provider_extensions import CompletionGateway


class ProviderClientFactory:
    """Compatibility name for a registry-backed provider lookup.

    New application construction injects ``ProviderGatewayRegistry`` directly.
    """
    def __init__(self, registry: ProviderGatewayRegistry) -> None:
        self._registry = registry

    def create(self, provider_instance_name: str) -> CompletionGateway:
        return self._registry.get(provider_instance_name)
