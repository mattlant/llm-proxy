from contextlib import asynccontextmanager

import pytest

from llm_proxy.domain.requests import SamplingParameters
from llm_proxy.provider_extensions import WireExecution, WirePatch, WirePatchOperation, WireProtocolCapability, WireRequest, WireRequestProjection
from llm_proxy.providers.ollama.client import OllamaProviderClient


class RawResponse:
    status_code = 200
    headers = {}
    raw_headers = ((b"x-upstream", b"one"),)
    def __init__(self): self.closed = 0
    async def aiter_bytes(self):
        yield b'data: {"id":"r","model":"m","choices":[{"delta":{},"finish_reason":null}]}\n\n'
        yield b'data: [DONE]\n\n'
    async def aclose(self): self.closed += 1


class Transport:
    def __init__(self): self.body = None; self.response = RawResponse(); self.closed = 0
    async def post_raw(self, url, headers, body, timeout, **kwargs):
        self.body = body
        from llm_proxy.infrastructure.http_client import HttpResponse
        return HttpResponse(200, {}, b'{"vendor":null}', ((b"x-upstream", b"one"),))
    async def stream_raw(self, url, headers, body, timeout, **kwargs):
        self.body = body
        transport = self
        @asynccontextmanager
        async def context():
            try: yield transport.response
            finally: transport.closed += 1
        return context()


def execution(stream=False, patch=WirePatch()):
    capability = WireProtocolCapability("openai.chat-completions", "1", "/v1/chat/completions")
    request = WireRequest(capability, b'{"model":"alias","messages":[],"vendor":null}', (), {"model":"alias", "messages":[], "vendor":None}, WireRequestProjection("alias", stream, SamplingParameters()))
    return WireExecution(request, "ollama", "upstream", __import__("llm_proxy.provider_extensions", fromlist=["ProviderExecutionPolicy"]).ProviderExecutionPolicy(), patch)


@pytest.mark.asyncio
async def test_wire_client_relays_non_stream_body_and_extra_patch() -> None:
    transport = Transport()
    client = OllamaProviderClient("http://provider.test", "ollama", transport=transport)
    response = await client.complete_wire(execution(patch=WirePatch((WirePatchOperation("/seed", 7),))))
    assert response.body == b'{"vendor":null}'
    assert b'"seed":7' in transport.body


@pytest.mark.asyncio
async def test_wire_client_relays_fragmented_stream_and_closes_once() -> None:
    transport = Transport()
    client = OllamaProviderClient("http://provider.test", "ollama", transport=transport)
    stream = await client.stream_wire(execution(stream=True))
    assert [chunk async for chunk in stream] == [b'data: {"id":"r","model":"m","choices":[{"delta":{},"finish_reason":null}]}\n\n', b'data: [DONE]\n\n']
    assert transport.closed == 1
