from __future__ import annotations

import json

import httpx
import pytest

from llm_proxy.domain.content import TextContent
from llm_proxy.domain.errors import ContextLimitExceededError, DeveloperRoleCompatibilityError, ProviderProtocolError, ProviderUnavailableError, UnknownModelError
from llm_proxy.domain.messages import DeveloperRoleMode, Message, MessageRole
from llm_proxy.domain.requests import CompletionRequest, SamplingParameters
from llm_proxy.infrastructure.http_client import HttpResponse, HttpxAsyncHttpTransport
from llm_proxy.provider_extensions import CompletionExecution, ProviderExecutionPolicy, ProviderInstanceConfig
from llm_proxy.providers.ollama.extension import OllamaFactory


def execution() -> CompletionExecution:
    request = CompletionRequest(
        model="qwable-alias",
        messages=(Message(MessageRole.USER, (TextContent("Read README."),)),),
        parameters=SamplingParameters(),
    )
    return CompletionExecution(request, "hf.co/lordx64/Qwable-v2-GGUF:Q4_K_M", "rtx-3090", SamplingParameters(temperature=0.7), request.model)


def make_client(transport: object):
    return OllamaFactory(transport_factory=lambda: transport).create(
        ProviderInstanceConfig("rtx-3090", "ollama", {"base_url": "http://provider.test/", "outbound_interface": "openai"})
    )


def success_payload() -> dict[str, object]:
    return {
        "id": "chatcmpl-123",
        "model": "hf.co/lordx64/Qwable-v2-GGUF:Q4_K_M",
        "choices": [{"index": 0, "message": {"role": "assistant", "content": "File contents"}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 4},
    }


@pytest.mark.asyncio
async def test_executes_canonical_request_through_fake_ollama_endpoint() -> None:
    observed: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        observed["url"] = str(request.url)
        observed["headers"] = {key.lower(): value for key, value in request.headers.items()}
        observed["body"] = json.loads(request.read())
        return httpx.Response(200, json=success_payload())

    client = make_client(HttpxAsyncHttpTransport(lambda: httpx.AsyncClient(transport=httpx.MockTransport(handler))))

    response = await client.complete(execution())

    assert observed["url"] == "http://provider.test/v1/chat/completions"
    assert observed["body"]["model"] == "hf.co/lordx64/Qwable-v2-GGUF:Q4_K_M"
    assert observed["body"]["temperature"] == 0.7
    assert observed["body"]["messages"] == [{"role": "user", "content": "Read README."}]
    assert observed["headers"]["accept"] == "application/json"
    assert response.id == "chatcmpl-123"
    assert response.message.content == (TextContent("File contents"),)
    assert response.usage.input_tokens == 10
    assert response.usage.output_tokens == 4


def test_rejects_unsupported_outbound_interface_during_activation() -> None:
    with pytest.raises(ValueError, match="outbound_interface"):
        OllamaFactory().create(ProviderInstanceConfig("rtx-3090", "ollama", {"base_url": "http://provider.test/", "outbound_interface": "ollama"}))


class StaticTransport:
    def __init__(self, response: HttpResponse) -> None:
        self.response = response
        self.calls = 0

    async def post_json(self, url: str, headers: object, body: object, timeout: float | None, *, trace_provider: str | None = None) -> HttpResponse:
        self.calls += 1
        return self.response


@pytest.mark.asyncio
async def test_rejects_developer_role_before_transport() -> None:
    transport = StaticTransport(HttpResponse(200, {}, b"{}"))
    request = CompletionRequest("qwable", (Message(MessageRole.DEVELOPER, (TextContent("instructions"),)),))
    rejected = CompletionExecution(
        request, "upstream", "rtx-3090", SamplingParameters(), request.model,
        policy=ProviderExecutionPolicy(developer_role_mode=DeveloperRoleMode.REJECT),
    )
    client = make_client(transport)

    with pytest.raises(DeveloperRoleCompatibilityError):
        await client.complete(rejected)
    assert transport.calls == 0


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("status", "body", "error"),
    [
        (400, b'{"error":"bad request"}', ProviderProtocolError),
        (404, b'{"error":"model not found"}', UnknownModelError),
        (404, b'{"error":"endpoint not found"}', ProviderProtocolError),
        (413, b'{"error":"context too large"}', ContextLimitExceededError),
        (408, b'{"error":"timeout"}', ProviderUnavailableError),
        (500, b'{"error":"server error"}', ProviderUnavailableError),
    ],
)
async def test_maps_upstream_http_statuses(status: int, body: bytes, error: type[Exception]) -> None:
    client = make_client(StaticTransport(HttpResponse(status, {}, body)))

    with pytest.raises(error):
        await client.complete(execution())


@pytest.mark.asyncio
@pytest.mark.parametrize("body", [b"not-json", b"[]"])
async def test_rejects_malformed_success_body(body: bytes) -> None:
    client = make_client(StaticTransport(HttpResponse(200, {}, body)))

    with pytest.raises(ProviderProtocolError):
        await client.complete(execution())
