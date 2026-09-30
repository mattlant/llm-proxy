from __future__ import annotations

import httpx
import pytest

from llm_proxy.domain.errors import ProviderUnavailableError
from llm_proxy.infrastructure.http_client import HttpxAsyncHttpTransport, provider_url


def test_provider_url_normalizes_base_and_path_boundaries() -> None:
    assert provider_url("http://provider.test/", "/v1/chat/completions") == "http://provider.test/v1/chat/completions"
    assert provider_url("http://provider.test", "v1/chat/completions") == "http://provider.test/v1/chat/completions"


@pytest.mark.asyncio
async def test_post_json_sends_exact_url_headers_body_and_timeout() -> None:
    observed: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        observed["url"] = str(request.url)
        observed["headers"] = {key.lower(): value for key, value in request.headers.items()}
        observed["body"] = request.read()
        return httpx.Response(200, json={"ok": True})

    transport = HttpxAsyncHttpTransport(
        lambda: httpx.AsyncClient(transport=httpx.MockTransport(handler))
    )
    response = await transport.post_json(
        provider_url("http://provider.test/", "/v1/chat/completions"),
        {"X-Request-ID": "request-only"},
        {"model": "qwable", "stream": False},
        12.5,
    )

    assert response.status_code == 200
    assert response.body == b'{"ok":true}'
    assert observed["url"] == "http://provider.test/v1/chat/completions"
    headers = observed["headers"]
    assert isinstance(headers, dict)
    assert headers["accept"] == "application/json"
    assert headers["content-type"] == "application/json"
    assert headers["x-request-id"] == "request-only"
    assert observed["body"] == b'{"model":"qwable","stream":false}'


@pytest.mark.asyncio
async def test_stream_json_uses_sse_headers_and_closes_client() -> None:
    closed = 0

    class TrackingClient(httpx.AsyncClient):
        async def aclose(self) -> None:
            nonlocal closed
            closed += 1
            await super().aclose()

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.headers["accept"] == "text/event-stream"
        assert request.headers["content-type"] == "application/json"
        return httpx.Response(200, content=b"data: {}\n\n")

    transport = HttpxAsyncHttpTransport(
        lambda: TrackingClient(transport=httpx.MockTransport(handler))
    )
    context = await transport.stream_json(
        "http://provider.test/v1/chat/completions",
        {},
        {"model": "qwable", "stream": True},
        5.0,
    )
    async with context as response:
        chunks = [chunk async for chunk in response.aiter_bytes()]

    assert chunks == [b"data: {}\n\n"]
    assert closed == 1


@pytest.mark.asyncio
async def test_network_failure_maps_to_provider_unavailable() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused", request=request)

    transport = HttpxAsyncHttpTransport(
        lambda: httpx.AsyncClient(transport=httpx.MockTransport(handler))
    )

    with pytest.raises(ProviderUnavailableError, match="provider request failed"):
        await transport.post_json("http://provider.test/v1/chat/completions", {}, {}, None)


@pytest.mark.asyncio
async def test_get_raw_sends_no_body_and_closes_client() -> None:
    closed = 0

    class TrackingClient(httpx.AsyncClient):
        async def aclose(self) -> None:
            nonlocal closed
            closed += 1
            await super().aclose()

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "GET"
        assert request.url.path == "/api/tags"
        assert request.read() == b""
        return httpx.Response(200, content=b'{"models": []}')

    response = await HttpxAsyncHttpTransport(lambda: TrackingClient(transport=httpx.MockTransport(handler))).get_raw(
        "http://provider.test/api/tags", ((b"accept", b"application/json"),), 3
    )

    assert response.body == b'{"models": []}'
    assert closed == 1
