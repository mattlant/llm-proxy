from __future__ import annotations

from urllib.parse import urlparse

from llm_proxy import __version__
from llm_proxy.infrastructure.http_client import AsyncHttpTransport
from llm_proxy.provider_extensions import (
    ProviderCapabilities, ProviderExtensionMetadata, ProviderFactory, ProviderInstanceConfig, SdkCompatibility, WireProtocolCapability,
)

from .client import OllamaProviderClient


class OllamaFactory(ProviderFactory):
    def __init__(self, transport_factory=None, timeout: float | None = None) -> None:
        self._transport_factory = transport_factory
        self._timeout = timeout

    def create(self, config: ProviderInstanceConfig) -> OllamaProviderClient:
        base_url = config.config.get("base_url")
        outbound_interface = config.config.get("outbound_interface", "openai")
        if not isinstance(base_url, str) or not base_url.strip():
            raise ValueError("Ollama config base_url must be non-blank")
        parsed = urlparse(base_url)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise ValueError("Ollama config base_url must use an HTTP or HTTPS URL with a host")
        if outbound_interface != "openai":
            raise ValueError("Ollama config outbound_interface must be 'openai'")
        transport: AsyncHttpTransport | None = self._transport_factory() if self._transport_factory else None
        return OllamaProviderClient(base_url.rstrip("/"), config.instance_name, transport=transport, timeout=self._timeout)


class OllamaExtension:
    metadata = ProviderExtensionMetadata("ollama", "Ollama OpenAI-compatible", "llm-proxy", __version__)
    compatibility = SdkCompatibility("0.1.0", "0.5.999")
    capabilities = ProviderCapabilities(streaming=True, native_tools=True, parallel_tools=True, usage=True, cancellation=True, model_listing=True, wire_protocols=(WireProtocolCapability("openai.chat-completions", "1", "/v1/chat/completions", requires_stream_usage=True),))
    factory = OllamaFactory()
