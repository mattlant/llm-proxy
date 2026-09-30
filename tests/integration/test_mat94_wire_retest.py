from contextlib import asynccontextmanager
import logging

import httpx

from llm_proxy.app import create_app
from llm_proxy.application.provider_registry import ProviderGatewayRegistry
from llm_proxy.application.runtime import RuntimeDependencies
from llm_proxy.configuration.loader import ConfigurationLoader
from llm_proxy.configuration.store import ConfigurationStore
from llm_proxy.infrastructure.http_client import HttpResponse
from llm_proxy.providers.ollama.client import OllamaProviderClient
from llm_proxy.providers.ollama.extension import OllamaExtension


class RawResponse:
    status_code = 200
    headers = {}
    raw_headers = ((b"x-upstream", b"one"),)

    async def aiter_bytes(self):
        yield b'data: {"id":"r","model":"m","choices":[]}'
        yield b'\n\ndata: [DONE]\n\n'


class Transport:
    def __init__(self):
        self.body = None
        self.response = RawResponse()
        self.closed = 0

    async def post_raw(self, url, headers, body, timeout, **kwargs):
        self.body = body
        return HttpResponse(200, {}, b'{"vendor":null}', ((b"x-upstream", b"one"),))

    async def stream_raw(self, url, headers, body, timeout, **kwargs):
        self.body = body

        @asynccontextmanager
        async def context():
            try:
                yield self.response
            finally:
                self.closed += 1

        return context()


def app_with_transport(config_file, transport):
    store = ConfigurationStore(config_file, ConfigurationLoader(), 1)
    client = OllamaProviderClient("http://provider.test", "test-provider", transport=transport)
    gateways = ProviderGatewayRegistry(
        {"test-provider": client},
        providers=store.snapshot.config.providers,
        capabilities={"test-provider": OllamaExtension.capabilities},
    )
    store = ConfigurationStore(config_file, ConfigurationLoader(), 1, runtime_dependencies=RuntimeDependencies(provider_gateways=gateways))
    return create_app(store, gateways)


async def test_openai_endpoint_selects_wire_and_preserves_payload(config_file):
    transport = Transport()
    app = app_with_transport(config_file, transport)
    body = b'{"model":"test-model","messages":[{"role":"user","content":[{"type":"image_url","image_url":{"url":"x"}}]}],"vendor":null}'
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://proxy.test") as client:
        response = await client.post("/v1/chat/completions", content=body, headers={"content-type": "application/json"})
    assert response.status_code == 200
    assert response.content == b'{"vendor":null}'
    assert b'"vendor":null' in transport.body


async def test_wire_request_emits_preserve_lifecycle_and_bounded_parameters(config_file, caplog):
    transport = Transport()
    app = app_with_transport(config_file, transport)
    caplog.set_level(logging.DEBUG, logger="llm-proxy")
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://proxy.test") as client:
        response = await client.post("/v1/chat/completions", content=b'{"model":"test-model","messages":[{"role":"user","content":"private"}],"temperature":0.2,"api_key":"secret"}', headers={"content-type": "application/json"})
    assert response.status_code == 200
    messages = [record.getMessage() for record in caplog.records if record.name == "llm-proxy" and record.getMessage().startswith("completion_")]
    assert [message.split()[0] for message in messages] == ["completion_started", "completion_decision", "completion_effective_parameters", "completion_finished"]
    assert "mode=preserve" in messages[1]
    assert "temperature=0.2" in messages[2]
    assert "private" not in messages[2]
    assert "secret" not in messages[2]


async def test_openai_endpoint_relays_wire_stream_and_closes(config_file):
    transport = Transport()
    app = app_with_transport(config_file, transport)
    body = b'{"model":"test-model","stream":true,"messages":[]}'
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://proxy.test") as client:
        response = await client.post("/v1/chat/completions", content=body, headers={"content-type": "application/json"})
    assert response.status_code == 200
    assert response.content == b'data: {"id":"r","model":"m","choices":[]}\n\ndata: [DONE]\n\n'
    assert transport.closed == 1
