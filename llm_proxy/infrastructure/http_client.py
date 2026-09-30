from __future__ import annotations

from collections.abc import AsyncIterator, Callable, Mapping
from contextlib import AbstractAsyncContextManager
from dataclasses import dataclass
from typing import Any, Protocol

import httpx

from llm_proxy.domain.errors import ProviderUnavailableError
from llm_proxy.observability.payload_trace import current_recorder


JsonValue = Any


def provider_url(base_url: str, path: str) -> str:
    if not isinstance(base_url, str) or not base_url.strip():
        raise ValueError("provider base URL must be non-blank")
    if not isinstance(path, str):
        raise TypeError("provider path must be a string")
    return f"{base_url.rstrip('/')}/{path.lstrip('/')}"


@dataclass(frozen=True, slots=True)
class HttpResponse:
    status_code: int
    headers: Mapping[str, str]
    body: bytes
    raw_headers: tuple[tuple[bytes, bytes], ...] = ()


class HttpStreamResponse(Protocol):
    status_code: int
    headers: Mapping[str, str]
    raw_headers: tuple[tuple[bytes, bytes], ...]

    def aiter_bytes(self) -> AsyncIterator[bytes]:
        ...

    async def aclose(self) -> None:
        ...


class AsyncHttpTransport(Protocol):
    async def get_raw(self, url: str, headers: tuple[tuple[bytes, bytes], ...], timeout: float | None, *, trace_provider: str | None = None) -> HttpResponse: ...
    async def post_raw(self, url: str, headers: tuple[tuple[bytes, bytes], ...], body: bytes, timeout: float | None, *, trace_provider: str | None = None) -> HttpResponse: ...
    async def stream_raw(self, url: str, headers: tuple[tuple[bytes, bytes], ...], body: bytes, timeout: float | None, *, trace_provider: str | None = None) -> AbstractAsyncContextManager[HttpStreamResponse]: ...
    async def post_json(
        self,
        url: str,
        headers: Mapping[str, str],
        body: Mapping[str, JsonValue],
        timeout: float | None,
        *,
        trace_provider: str | None = None,
    ) -> HttpResponse:
        ...

    async def stream_json(
        self,
        url: str,
        headers: Mapping[str, str],
        body: Mapping[str, JsonValue],
        timeout: float | None,
        *,
        trace_provider: str | None = None,
    ) -> AbstractAsyncContextManager[HttpStreamResponse]:
        ...


class HttpxAsyncHttpTransport:
    async def get_raw(self, url: str, headers: tuple[tuple[bytes, bytes], ...], timeout: float | None, *, trace_provider: str | None = None) -> HttpResponse:
        client = self._client_factory()
        try:
            request = client.build_request("GET", url, headers=headers, timeout=timeout)
            recorder = current_recorder()
            if recorder is not None:
                recorder.record_raw_http("provider_request_raw", b"", provider=trace_provider, method=request.method, url=str(request.url), path=request.url.path, query=request.url.query, headers=request.headers.raw)
            response = await client.send(request)
            return HttpResponse(response.status_code, dict(response.headers), await response.aread(), tuple(response.headers.raw))
        except httpx.RequestError as exc:
            raise ProviderUnavailableError("provider request failed", details={"url": url, "error": str(exc)}) from exc
        finally:
            await client.aclose()
    def __init__(self, client_factory: Callable[[], httpx.AsyncClient] | None = None) -> None:
        self._client_factory = client_factory or httpx.AsyncClient

    async def post_json(
        self,
        url: str,
        headers: Mapping[str, str],
        body: Mapping[str, JsonValue],
        timeout: float | None,
        *,
        trace_provider: str | None = None,
    ) -> HttpResponse:
        request_headers = self._headers(headers, "application/json")
        client = self._client_factory()
        try:
            request = client.build_request("POST", url, headers=request_headers, json=dict(body), timeout=timeout)
            recorder = current_recorder()
            if recorder is not None:
                recorder.record_raw_http("provider_request_raw", request.content, provider=trace_provider, method=request.method, url=str(request.url), path=request.url.path, query=request.url.query, headers=request.headers.raw)
            response = await client.send(request)
            content = await response.aread()
            return HttpResponse(response.status_code, dict(response.headers), content, tuple(response.headers.raw))
        except httpx.RequestError as exc:
            raise ProviderUnavailableError(
                "provider request failed",
                details={"url": url, "error": str(exc)},
            ) from exc
        finally:
            await client.aclose()

    async def post_raw(self, url: str, headers: tuple[tuple[bytes, bytes], ...], body: bytes, timeout: float | None, *, trace_provider: str | None = None) -> HttpResponse:
        client = self._client_factory()
        try:
            request = client.build_request("POST", url, headers=headers, content=body, timeout=timeout)
            recorder = current_recorder()
            if recorder is not None:
                recorder.record_raw_http("provider_request_raw", request.content, provider=trace_provider, method=request.method, url=str(request.url), path=request.url.path, query=request.url.query, headers=request.headers.raw)
            response = await client.send(request)
            return HttpResponse(response.status_code, dict(response.headers), await response.aread(), tuple(response.headers.raw))
        except httpx.RequestError as exc:
            raise ProviderUnavailableError("provider request failed", details={"url": url, "error": str(exc)}) from exc
        finally:
            await client.aclose()

    async def stream_json(
        self,
        url: str,
        headers: Mapping[str, str],
        body: Mapping[str, JsonValue],
        timeout: float | None,
        *,
        trace_provider: str | None = None,
    ) -> AbstractAsyncContextManager[HttpStreamResponse]:
        return _HttpxStreamContext(
            self._client_factory,
            url,
            self._headers(headers, "text/event-stream"),
            body,
            timeout,
            trace_provider,
        )

    async def stream_raw(self, url: str, headers: tuple[tuple[bytes, bytes], ...], body: bytes, timeout: float | None, *, trace_provider: str | None = None) -> AbstractAsyncContextManager[HttpStreamResponse]:
        return _HttpxStreamContext(self._client_factory, url, headers, body, timeout, trace_provider, raw=True)

    @staticmethod
    def _headers(headers: Mapping[str, str], accept: str) -> dict[str, str]:
        result = {key: value for key, value in headers.items()}
        result.setdefault("Content-Type", "application/json")
        result.setdefault("Accept", accept)
        return result


class _HttpxStreamResponse:
    def __init__(self, response: httpx.Response) -> None:
        self._response = response
        self.status_code = response.status_code
        self.headers = response.headers
        self.raw_headers = tuple(response.headers.raw)

    def aiter_bytes(self) -> AsyncIterator[bytes]:
        return self._response.aiter_bytes()

    async def aclose(self) -> None:
        await self._response.aclose()


class _HttpxStreamContext:
    def __init__(
        self,
        client_factory: Callable[[], httpx.AsyncClient],
        url: str,
        headers: Mapping[str, str] | tuple[tuple[bytes, bytes], ...],
        body: Mapping[str, JsonValue] | bytes,
        timeout: float | None,
        trace_provider: str | None,
        raw: bool = False,
    ) -> None:
        self._client_factory = client_factory
        self._url = url
        self._headers = headers
        self._body = body
        self._timeout = timeout
        self._trace_provider = trace_provider
        self._raw = raw
        self._client: httpx.AsyncClient | None = None
        self._response: httpx.Response | None = None

    async def __aenter__(self) -> HttpStreamResponse:
        self._client = self._client_factory()
        try:
            request = self._client.build_request("POST", self._url, headers=self._headers, content=self._body if self._raw else None, json=None if self._raw else dict(self._body), timeout=self._timeout)
            recorder = current_recorder()
            if recorder is not None:
                recorder.record_raw_http("provider_request_raw", request.content, provider=self._trace_provider, stream=True, method=request.method, url=str(request.url), path=request.url.path, query=request.url.query, headers=request.headers.raw)
            self._response = await self._client.send(request, stream=True)
            return _HttpxStreamResponse(self._response)
        except httpx.RequestError as exc:
            await self._client.aclose()
            self._client = None
            raise ProviderUnavailableError(
                "provider stream request failed",
                details={"url": self._url, "error": str(exc)},
            ) from exc

    async def __aexit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        if self._response is not None:
            await self._response.aclose()
            self._response = None
        if self._client is not None:
            await self._client.aclose()
            self._client = None
