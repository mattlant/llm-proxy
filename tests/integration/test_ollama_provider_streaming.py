from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any

import httpx
import pytest

from llm_proxy.domain.errors import DeveloperRoleCompatibilityError, ProviderProtocolError, ProviderUnavailableError
from llm_proxy.domain.content import TextContent
from llm_proxy.domain.events import ResponseCompleted, ResponseFailed, ResponseStarted, TextDelta
from llm_proxy.domain.messages import DeveloperRoleMode, Message, MessageRole
from llm_proxy.domain.requests import CompletionRequest, SamplingParameters
from llm_proxy.infrastructure.http_client import HttpStreamResponse
from llm_proxy.provider_extensions import CompletionExecution, ProviderExecutionPolicy, ProviderInstanceConfig
from llm_proxy.providers.ollama.extension import OllamaFactory


def execution() -> CompletionExecution:
    request = CompletionRequest(
        model="qwable",
        messages=(Message(MessageRole.USER, (TextContent("Hello"),)),),
    )
    return CompletionExecution(request, "hf.co/qwable:Q4_K_M", "rtx-3090", SamplingParameters(), request.model)


def make_client(transport: object):
    return OllamaFactory(transport_factory=lambda: transport).create(
        ProviderInstanceConfig("rtx-3090", "ollama", {"base_url": "http://provider.test/"})
    )


@dataclass
class FakeStreamResponse(HttpStreamResponse):
    status_code: int
    chunks: list[bytes]
    error: Exception | None = None
    consumed: int = 0

    @property
    def headers(self) -> dict[str, str]:
        return {}

    async def aiter_bytes(self):
        for chunk in self.chunks:
            self.consumed += 1
            yield chunk
        if self.error is not None:
            raise self.error

    async def aclose(self) -> None:
        pass


class FakeContext:
    def __init__(self, response: FakeStreamResponse) -> None:
        self.response = response
        self.entered = 0
        self.exited = 0

    async def __aenter__(self) -> FakeStreamResponse:
        self.entered += 1
        return self.response

    async def __aexit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        self.exited += 1


class FakeTransport:
    def __init__(self, context: FakeContext) -> None:
        self.context = context
        self.url: str | None = None
        self.body: dict[str, Any] | None = None

    async def stream_json(self, url: str, headers: Any, body: dict[str, Any], timeout: float | None, *, trace_provider: str | None = None):
        self.url = url
        self.body = body
        return self.context


def valid_start() -> bytes:
    return b'data: {"id":"stream-1","model":"qwable","choices":[{"delta":{"role":"assistant","content":"hello"},"finish_reason":null}]}\n\n'


def valid_end() -> bytes:
    return b'data: {"choices":[{"delta":{},"finish_reason":"stop"}]}\n\ndata: [DONE]\n\n'


def trailing_usage_and_done() -> bytes:
    return b'data: {"choices":[{"delta":{},"finish_reason":"stop"}]}\n\ndata: {"choices":[],"usage":{"prompt_tokens":7,"completion_tokens":2}}\n\ndata: [DONE]\n\n'


def finish_without_done() -> bytes:
    return b'data: {"choices":[{"delta":{},"finish_reason":"stop"}]}\n\n'


@pytest.mark.asyncio
async def test_normal_stream_maps_events_and_closes_context_once() -> None:
    response = FakeStreamResponse(200, [valid_start(), valid_end()])
    context = FakeContext(response)
    transport = FakeTransport(context)
    client = make_client(transport)

    events = [event async for event in client.stream(execution())]

    assert transport.url == "http://provider.test/v1/chat/completions"
    assert isinstance(events[0], ResponseStarted)
    assert TextDelta("text_0", "hello") in events
    assert context.entered == 1
    assert context.exited == 1
    assert response.consumed == 2


@pytest.mark.asyncio
async def test_stream_consumes_trailing_usage_and_done_before_closing_context() -> None:
    response = FakeStreamResponse(200, [valid_start(), trailing_usage_and_done()])
    context = FakeContext(response)
    client = make_client(FakeTransport(context))

    events = [event async for event in client.stream(execution())]

    completion = next(event for event in events if isinstance(event, ResponseCompleted))
    assert completion.usage.input_tokens == 7
    assert completion.usage.output_tokens == 2
    assert response.consumed == 2
    assert context.exited == 1


@pytest.mark.asyncio
async def test_stream_allows_clean_eof_after_finish_reason() -> None:
    response = FakeStreamResponse(200, [valid_start(), finish_without_done()])
    context = FakeContext(response)

    events = [event async for event in make_client(FakeTransport(context)).stream(execution())]

    assert any(isinstance(event, ResponseCompleted) for event in events)
    assert response.consumed == 2
    assert context.exited == 1


@pytest.mark.asyncio
async def test_rejects_developer_role_before_stream_transport() -> None:
    context = FakeContext(FakeStreamResponse(200, []))
    transport = FakeTransport(context)
    request = CompletionRequest("qwable", (Message(MessageRole.DEVELOPER, (TextContent("instructions"),)),), stream=True)
    rejected = CompletionExecution(
        request, "upstream", "rtx-3090", SamplingParameters(), request.model,
        policy=ProviderExecutionPolicy(developer_role_mode=DeveloperRoleMode.REJECT),
    )

    with pytest.raises(DeveloperRoleCompatibilityError):
        _ = [event async for event in make_client(transport).stream(rejected)]

    assert transport.url is None
    assert context.entered == 0


@pytest.mark.asyncio
async def test_http_error_before_events_raises_and_closes_context() -> None:
    response = FakeStreamResponse(500, [b'{"error":"server"}'])
    context = FakeContext(response)
    client = make_client(FakeTransport(context))

    with pytest.raises(ProviderUnavailableError):
        _ = [event async for event in client.stream(execution())]

    assert context.entered == 1
    assert context.exited == 1


@pytest.mark.asyncio
async def test_malformed_sse_after_start_emits_response_failed() -> None:
    response = FakeStreamResponse(200, [valid_start(), b"data: not-json\n\n"])
    context = FakeContext(response)
    client = make_client(FakeTransport(context))

    events = [event async for event in client.stream(execution())]

    assert isinstance(events[0], ResponseStarted)
    assert isinstance(events[-1], ResponseFailed)
    assert isinstance(events[-1].error, ProviderProtocolError)
    assert context.exited == 1


@pytest.mark.asyncio
async def test_upstream_read_error_after_start_emits_response_failed() -> None:
    response = FakeStreamResponse(200, [valid_start()], httpx.ReadError("read failed"))
    context = FakeContext(response)
    client = make_client(FakeTransport(context))

    events = [event async for event in client.stream(execution())]

    assert isinstance(events[-1], ResponseFailed)
    assert isinstance(events[-1].error, ProviderUnavailableError)
    assert context.exited == 1


@pytest.mark.asyncio
async def test_consumer_stopping_early_closes_context() -> None:
    response = FakeStreamResponse(200, [valid_start(), valid_end()])
    context = FakeContext(response)
    client = make_client(FakeTransport(context))
    iterator = client.stream(execution())

    async for event in iterator:
        if isinstance(event, TextDelta):
            break
    await iterator.aclose()

    assert context.exited == 1
    assert response.consumed == 1


@pytest.mark.asyncio
async def test_cancellation_re_raises_and_closes_without_consuming_more_chunks() -> None:
    cancelled = asyncio.Event()

    class BlockingResponse(FakeStreamResponse):
        async def aiter_bytes(self):
            self.consumed += 1
            yield valid_start()
            await cancelled.wait()

    response = BlockingResponse(200, [])
    context = FakeContext(response)
    client = make_client(FakeTransport(context))

    async def consume() -> None:
        async for _ in client.stream(execution()):
            pass

    task = asyncio.create_task(consume())
    await asyncio.sleep(0)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    cancelled.set()

    assert context.exited == 1
    assert response.consumed == 1
