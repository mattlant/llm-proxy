from __future__ import annotations

import httpx

from llm_proxy.app import create_app
from llm_proxy.application.provider_registry import ProviderGatewayRegistry
from llm_proxy.application.runtime import RuntimeDependencies
from llm_proxy.configuration.loader import ConfigurationLoader
from llm_proxy.configuration.store import ConfigurationStore
from llm_proxy.domain.content import TextContent
from llm_proxy.domain.events import ResponseCompleted, ResponseStarted, TextDelta
from llm_proxy.domain.messages import Message, MessageRole
from llm_proxy.domain.responses import CompletionResponse, FinishReason, Usage
from llm_proxy.extensions import CapabilityRegistration, ExtensionMetadata, ExtensionSdkCompatibility, SIDE_EFFECT_FAMILY, SIDE_EFFECT_VERSION
from llm_proxy.provider_extensions import ProviderCapabilities, WireProtocolCapability
from llm_proxy.provider_extensions.wire import WireResponse, WireStream


class _Capability:
    def __init__(self, events): self._events = events
    def validate_arguments(self, arguments): pass
    async def invoke(self, context, arguments): self._events.append("capability")


class _Instance:
    def __init__(self, events):
        self.registrations = (CapabilityRegistration(SIDE_EFFECT_FAMILY, SIDE_EFFECT_VERSION, "run", _Capability(events)),)
        self._events = events
    async def aclose(self): self._events.append("extension-close")


class _Extension:
    metadata = ExtensionMetadata("test.pipeline", "Test pipeline", "test", "1")
    compatibility = ExtensionSdkCompatibility("1.0.0", "1.0.0")
    events = []
    class factory:
        @staticmethod
        def create(config): return _Instance(_Extension.events)


class _EntryPoint:
    name = "pipeline"
    def load(self): return _Extension()


class _Gateway:
    def __init__(self, events): self._events = events
    async def complete(self, execution):
        self._events.append("provider")
        return CompletionResponse("response", execution.upstream_model, Message(MessageRole.ASSISTANT, (TextContent("ok"),)), FinishReason.END_TURN, None, Usage(1, 1))
    def stream(self, execution):
        self._events.append("provider")
        return self._events_stream(execution)
    async def _events_stream(self, execution):
        yield ResponseStarted("response", execution.upstream_model)
        yield TextDelta("text", "ok")
        yield ResponseCompleted(FinishReason.END_TURN, None, Usage(1, 1))

    async def complete_wire(self, execution):
        self._events.append("wire-complete")
        return WireResponse(200, (), b'{"wire":true}')

    async def stream_wire(self, execution):
        self._events.append("wire-stream")

        async def chunks():
            yield b'data: {"wire":true}\n\n'
            yield b'data: [DONE]\n\n'

        async def close():
            return None

        return WireStream(200, (), chunks(), close)


class _Factory:
    def __init__(self, gateway): self._gateway = gateway
    def get(self, instance): return self._gateway


async def test_lifespan_extension_executes_before_complete_and_stream_provider_dispatch(tmp_path) -> None:
    path = tmp_path / "config.yaml"
    path.write_text("""\
server: {host: 127.0.0.1, port: 11435, timeout_seconds: 0, config_reload_seconds: 1, log_level: INFO}
interfaces: {openai: {enabled: true}}
providers: {test: {extension: ollama, enabled: true, config: {base_url: http://unused.test, outbound_interface: openai}}}
extensions: {side: {extension: test.pipeline, enabled: true, config: {}}}
models: {test: {upstream_model: test, provider: test, interfaces: [openai]}}
policies: [{name: effect, enabled: true, match: {}, actions: {capabilities: [{id: side, instance: side, family: side_effect, version: 1, capability: run}]}}]
""", encoding="utf-8")
    events = _Extension.events = []
    app = create_app(ConfigurationStore(path, ConfigurationLoader(), 1), _Factory(_Gateway(events)), extension_entry_points=(_EntryPoint(),))

    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://proxy.test") as client:
            complete = await client.post("/v1/chat/completions", json={"model": "test", "messages": [{"role": "user", "content": "hello"}]})
            stream = await client.post("/v1/chat/completions", json={"model": "test", "stream": True, "messages": [{"role": "user", "content": "hello"}]})

    assert complete.status_code == stream.status_code == 200
    assert events[:4] == ["capability", "provider", "capability", "provider"]
    assert events[-1] == "extension-close"


async def test_lifespan_extension_executes_before_wire_complete_and_stream_provider_dispatch(tmp_path) -> None:
    path = tmp_path / "config.yaml"
    path.write_text("""\
server: {host: 127.0.0.1, port: 11435, timeout_seconds: 0, config_reload_seconds: 1, log_level: INFO}
interfaces: {openai: {enabled: true}}
providers: {test: {extension: ollama, enabled: true, config: {base_url: http://unused.test, outbound_interface: openai}}}
extensions: {side: {extension: test.pipeline, enabled: true, config: {}}}
models: {test: {upstream_model: test, provider: test, interfaces: [openai]}}
policies: [{name: effect, enabled: true, match: {}, actions: {capabilities: [{id: side, instance: side, family: side_effect, version: 1, capability: run}]}}]
""", encoding="utf-8")
    events = _Extension.events = []
    store = ConfigurationStore(path, ConfigurationLoader(), 1)
    wire = WireProtocolCapability("openai.chat-completions", "1", "/v1/chat/completions")
    gateways = ProviderGatewayRegistry({"test": _Gateway(events)}, providers=store.snapshot.config.providers, capabilities={"test": ProviderCapabilities(wire_protocols=(wire,))})
    store = ConfigurationStore(path, ConfigurationLoader(), 1, runtime_dependencies=RuntimeDependencies(provider_gateways=gateways))
    app = create_app(store, gateways, extension_entry_points=(_EntryPoint(),))

    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://proxy.test") as client:
            complete = await client.post("/v1/chat/completions", json={"model": "test", "messages": [{"role": "user", "content": "hello"}]})
            stream = await client.post("/v1/chat/completions", json={"model": "test", "stream": True, "messages": [{"role": "user", "content": "hello"}]})

    assert complete.status_code == stream.status_code == 200
    assert complete.json() == {"wire": True}
    assert stream.text.endswith("data: [DONE]\n\n")
    assert events[:4] == ["capability", "wire-complete", "capability", "wire-stream"]
    assert events[-1] == "extension-close"
