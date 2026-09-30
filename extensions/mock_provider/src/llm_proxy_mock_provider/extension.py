"""The standalone deterministic mock provider entry point."""
from __future__ import annotations

from .provider import DeterministicMockFactory
from llm_proxy.provider_extensions import ProviderCapabilities, ProviderExtensionMetadata, SdkCompatibility


class DeterministicMockExtension:
    metadata = ProviderExtensionMetadata("deterministic-mock", "Deterministic mock provider", "llm-proxy-mock-provider", "0.1.0")
    compatibility = SdkCompatibility("0.3.0", "0.3.999")
    capabilities = ProviderCapabilities(streaming=True, native_tools=True, parallel_tools=True, usage=True, cancellation=True, management_commands=True, model_listing=True)
    factory = DeterministicMockFactory()


extension = DeterministicMockExtension()
