from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import Mapping
from typing import Any

import httpx

from llm_proxy.provider_extensions import CompletionExecution, CompletionGateway, ProviderListedModel, ProviderListingError, ProviderListingFailureCategory, WireExecution, WireResponse, WireStream
from llm_proxy.domain.errors import (
    CompletionError,
    ContextLimitExceededError,
    ProviderProtocolError,
    ProviderUnavailableError,
    UnknownModelError,
)
from llm_proxy.domain.events import ResponseFailed, ResponseStarted
from llm_proxy.domain.responses import CompletionResponse
from llm_proxy.infrastructure.http_client import AsyncHttpTransport, HttpResponse, HttpxAsyncHttpTransport, provider_url
from llm_proxy.infrastructure.wire_headers import filter_request_headers, filter_response_headers
from llm_proxy.observability.payload_trace import current_recorder
from llm_proxy.observability.operational_logging import ExecutionMode, OperationalLogger, current_operational_request

from .request_mapper import OllamaOpenAIRequestMapper
from .response_mapper import OllamaOpenAIResponseMapper
from .stream_chunk_parser import OpenAIStreamParser
from .stream_parser import SseDecoder
from .wire_observer import OllamaWireStreamObserver

logger = logging.getLogger(__name__)


def _raw_headers(headers: Mapping[str, str]) -> tuple[tuple[bytes, bytes], ...]:
    raw = getattr(headers, "raw", None)
    if raw is not None:
        return tuple(raw)
    return tuple((name.encode("latin-1"), value.encode("latin-1")) for name, value in headers.items())


class OllamaProviderClient(CompletionGateway):
    def __init__(
        self,
        base_url: str,
        provider_name: str,
        transport: AsyncHttpTransport | None = None,
        request_mapper: OllamaOpenAIRequestMapper | None = None,
        response_mapper: OllamaOpenAIResponseMapper | None = None,
        timeout: float | None = None,
    ) -> None:
        self._transport = transport or HttpxAsyncHttpTransport()
        self._request_mapper = request_mapper or OllamaOpenAIRequestMapper()
        self._response_mapper = response_mapper or OllamaOpenAIResponseMapper()
        self._timeout = timeout
        self._base_url = base_url
        self._provider_name = provider_name

    @property
    def provider_name(self) -> str:
        return self._provider_name

    async def list_models(self) -> tuple[ProviderListedModel, ...]:
        try:
            response = await self._transport.get_raw(provider_url(self._base_url, "/api/tags"), (), None, trace_provider=self._provider_name)
            if not 200 <= response.status_code < 300:
                raise ProviderListingError(self._provider_name, ProviderListingFailureCategory.UNAVAILABLE, "Ollama tags request failed")
            payload = self._decode_json(response.body)
            records = payload.get("models") if isinstance(payload, dict) else None
            if not isinstance(records, list):
                raise ValueError("models must be a list")
            names: list[str] = []
            for record in records:
                if not isinstance(record, dict):
                    raise ValueError("model record must be an object")
                name = record.get("name") or record.get("model")
                if not isinstance(name, str) or not name.strip() or name in names:
                    raise ValueError("model name is invalid")
                names.append(name)
            return tuple(ProviderListedModel(name) for name in names)
        except ProviderListingError:
            raise
        except ProviderUnavailableError as error:
            raise ProviderListingError(self._provider_name, ProviderListingFailureCategory.UNAVAILABLE, "Ollama tags request failed") from error
        except Exception as error:
            raise ProviderListingError(self._provider_name, ProviderListingFailureCategory.INVALID_RESPONSE, "Ollama tags response is invalid") from error

    async def complete(self, execution: CompletionExecution) -> CompletionResponse:
        self._validate_provider(execution)
        body = self._request_mapper.map_request(execution)
        OperationalLogger().log_effective_parameters(current_operational_request(), ExecutionMode.CANONICAL, body)
        recorder = current_recorder()
        if recorder is not None:
            recorder.record("provider_request", body, provider=self._provider_name, stream=False)
        response = await self._transport.post_json(
            provider_url(self._base_url, "/v1/chat/completions"),
            {"Content-Type": "application/json", "Accept": "application/json"},
            body,
            self._timeout,
            trace_provider=self._provider_name,
        )
        if recorder is not None:
            recorder.record_raw_http("provider_response_raw", response.body, provider=self._provider_name, status_code=response.status_code, headers=response.raw_headers or _raw_headers(response.headers))
        self._raise_for_status(response, self._provider_name)
        payload = self._decode_json(response.body)
        if recorder is not None:
            recorder.record("provider_response", payload, provider=self._provider_name, stream=False)
        return self._response_mapper.map_response(
            payload,
            expose_thinking=execution.policy.expose_thinking,
        )

    def stream(self, execution: CompletionExecution):
        return self._stream(execution)

    async def complete_wire(self, execution: WireExecution) -> WireResponse:
        self._validate_wire_provider(execution)
        body = execution.patch.apply(execution.request)
        self._log_wire_parameters(body)
        response = await self._transport.post_raw(provider_url(self._base_url, execution.request.capability.endpoint), self._wire_headers(execution.request.headers, b"application/json"), body, self._timeout, trace_provider=self._provider_name)
        recorder = current_recorder()
        if recorder is not None:
            recorder.record_raw_http("provider_response_raw", response.body, provider=self._provider_name, status_code=response.status_code, headers=response.raw_headers)
        return WireResponse(response.status_code, filter_response_headers(response.raw_headers or _raw_headers(response.headers)), response.body)

    async def stream_wire(self, execution: WireExecution) -> WireStream:
        self._validate_wire_provider(execution)
        body = execution.patch.apply(execution.request)
        self._log_wire_parameters(body)
        context = await self._transport.stream_raw(provider_url(self._base_url, execution.request.capability.endpoint), self._wire_headers(execution.request.headers, b"text/event-stream"), body, self._timeout, trace_provider=self._provider_name)
        response = await context.__aenter__()
        headers = filter_response_headers(getattr(response, "raw_headers", ()) or _raw_headers(response.headers))
        observer = OllamaWireStreamObserver(self._provider_name, expose_thinking=execution.policy.expose_thinking)
        closed = False
        async def close_context() -> None:
            nonlocal closed
            if not closed:
                closed = True
                await context.__aexit__(None, None, None)
        async def chunks():
            try:
                async for chunk in response.aiter_bytes():
                    recorder = current_recorder()
                    if recorder is not None:
                        recorder.record_raw_http("provider_stream_bytes_raw", chunk, provider=self._provider_name, stream=True)
                    try:
                        observer.observe(chunk)
                    except Exception as error:
                        logger.debug("wire stream observer failed: %s", type(error).__name__)
                    yield chunk
            finally:
                try:
                    observer.finish()
                except Exception as error:
                    logger.debug("wire stream observer finish failed: %s", type(error).__name__)
                await close_context()
        async def close():
            await close_context()
        return WireStream(response.status_code, headers, chunks(), close)

    def _validate_wire_provider(self, execution: WireExecution) -> None:
        if execution.provider_instance_name != self._provider_name:
            raise ProviderProtocolError("configured provider does not match resolved execution", provider=self._provider_name)

    @staticmethod
    def _log_wire_parameters(body: bytes) -> None:
        try:
            payload = json.loads(body)
            if isinstance(payload, dict):
                OperationalLogger().log_effective_parameters(current_operational_request(), ExecutionMode.PRESERVE, payload)
        except Exception:
            context = current_operational_request()
            if context is not None:
                logging.getLogger("llm-proxy").debug("completion_effective_parameters request_id=%s mode=preserve effective_parameters=unavailable", context.request_id)

    @staticmethod
    def _wire_headers(headers: tuple[tuple[bytes, bytes], ...], accept: bytes) -> tuple[tuple[bytes, bytes], ...]:
        return filter_request_headers(headers, ((b"content-type", b"application/json"), (b"accept", accept)))

    async def _stream(self, execution: CompletionExecution):
        self._validate_provider(execution)
        body = self._request_mapper.map_request(execution)
        OperationalLogger().log_effective_parameters(current_operational_request(), ExecutionMode.CANONICAL, body)
        recorder = current_recorder()
        if recorder is not None:
            recorder.record("provider_request", body, provider=self._provider_name, stream=True)
        context = await self._transport.stream_json(
            provider_url(self._base_url, "/v1/chat/completions"),
            {"Content-Type": "application/json", "Accept": "text/event-stream"},
            body,
            self._timeout,
            trace_provider=self._provider_name,
        )
        started = False
        parser = OpenAIStreamParser(expose_thinking=execution.policy.expose_thinking)
        decoder = SseDecoder()
        try:
            async with context as response:
                if recorder is not None:
                    recorder.record_raw_http("provider_stream_start_raw", provider=self._provider_name, stream=True, status_code=response.status_code, headers=_raw_headers(response.headers))
                if not 200 <= response.status_code < 300:
                    chunks: list[bytes] = []
                    async for chunk in response.aiter_bytes():
                        if recorder is not None:
                            recorder.record_raw_http("provider_stream_bytes_raw", chunk, provider=self._provider_name, stream=True)
                        chunks.append(chunk)
                    body_bytes = b"".join(chunks)
                    self._raise_for_status(
                        HttpResponse(response.status_code, response.headers, body_bytes),
                        self._provider_name,
                    )
                async for chunk in response.aiter_bytes():
                    if recorder is not None:
                        recorder.record_raw_http("provider_stream_bytes_raw", chunk, provider=self._provider_name, stream=True)
                    for record in decoder.feed(chunk):
                        if recorder is not None:
                            try:
                                payload = json.loads(record.data) if record.data != "[DONE]" else record.data
                            except json.JSONDecodeError:
                                payload = record.data
                            recorder.record("provider_stream_record", payload, provider=self._provider_name, stream=True)
                        events = parser.feed(record)
                        for event in events:
                            started = started or isinstance(event, ResponseStarted)
                            yield event
                for record in decoder.finish():
                    if recorder is not None:
                        try:
                            payload = json.loads(record.data) if record.data != "[DONE]" else record.data
                        except json.JSONDecodeError:
                            payload = record.data
                        recorder.record("provider_stream_record", payload, provider=self._provider_name, stream=True)
                    events = parser.feed(record)
                    for event in events:
                        started = started or isinstance(event, ResponseStarted)
                        yield event
                for event in parser.finish():
                    started = started or isinstance(event, ResponseStarted)
                    yield event
        except asyncio.CancelledError:
            raise
        except httpx.RequestError as exc:
            error = ProviderUnavailableError("provider stream read failed", provider=self._provider_name, details={"error": str(exc)})
            if started:
                yield ResponseFailed(error)
                return
            raise error from exc
        except UnicodeDecodeError as exc:
            error = ProviderProtocolError("upstream stream contains invalid UTF-8", provider=self._provider_name)
            if started:
                yield ResponseFailed(error)
                return
            raise error from exc
        except CompletionError as error:
            if started:
                yield ResponseFailed(error)
                return
            raise

    def _validate_provider(self, execution: CompletionExecution) -> None:
        if execution.provider_instance_name != self._provider_name:
            raise ProviderProtocolError("configured provider does not match resolved execution", provider=self._provider_name)

    @staticmethod
    def _decode_json(body: bytes) -> Mapping[str, Any]:
        try:
            payload = json.loads(body.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ProviderProtocolError("upstream success body is not valid JSON") from exc
        if not isinstance(payload, Mapping):
            raise ProviderProtocolError("upstream success body must be a JSON object")
        return payload

    @classmethod
    def _raise_for_status(cls, response: HttpResponse, provider: str) -> None:
        if 200 <= response.status_code < 300:
            return
        body = response.body.decode("utf-8", errors="replace")
        if response.status_code == 400:
            raise ProviderProtocolError("upstream rejected the request", provider=provider, details={"status_code": response.status_code, "body": body})
        if response.status_code == 404:
            if cls._looks_like_missing_model(body):
                raise UnknownModelError("upstream model was not found", provider=provider, details={"status_code": response.status_code, "body": body})
            raise ProviderProtocolError("upstream endpoint or resource was not found", provider=provider, details={"status_code": response.status_code, "body": body})
        if response.status_code == 413:
            raise ContextLimitExceededError("upstream context limit exceeded", provider=provider, details={"status_code": response.status_code, "body": body})
        if response.status_code in {408, 429} or 500 <= response.status_code <= 599:
            raise ProviderUnavailableError("upstream provider is unavailable", provider=provider, details={"status_code": response.status_code, "body": body})
        raise ProviderProtocolError("unexpected upstream HTTP status", provider=provider, details={"status_code": response.status_code, "body": body})

    @staticmethod
    def _looks_like_missing_model(body: str) -> bool:
        try:
            payload = json.loads(body)
        except json.JSONDecodeError:
            return False
        if not isinstance(payload, Mapping):
            return False
        error = payload.get("error")
        return isinstance(error, str) and "model" in error.lower() and "not found" in error.lower()
