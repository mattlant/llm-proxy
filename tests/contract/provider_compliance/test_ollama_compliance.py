"""Provider-owned compliance bridge for the built-in Ollama extension."""
from __future__ import annotations

import asyncio
from dataclasses import dataclass

import pytest

from llm_proxy_provider_compliance import ComplianceAdapter, ErrorCase
from llm_proxy_provider_compliance import suite
from llm_proxy.domain.requests import CompletionRequest, SamplingParameters
from llm_proxy.infrastructure.http_client import HttpResponse
from llm_proxy.provider_extensions import (
    CompletionExecution, CompletionResponse, FinishReason, Message, MessageRole,
    ProviderInstanceConfig, ProviderUnavailableError, ResponseCompleted, ResponseStarted,
    TextCompleted, TextContent, TextDelta, TextStarted, Usage,
)
from llm_proxy.providers.ollama.extension import OllamaExtension, OllamaFactory


@dataclass
class _StreamResponse:
    chunks: tuple[bytes, ...]
    status_code: int = 200
    headers: dict[str, str] = None  # type: ignore[assignment]
    def __post_init__(self): self.headers = {}
    async def aiter_bytes(self):
        for chunk in self.chunks: yield chunk
    async def aclose(self): pass


class _Context:
    def __init__(self, response): self.response = response
    async def __aenter__(self): return self.response
    async def __aexit__(self, *args): pass


class _Transport:
    def __init__(self, failing: bool = False): self.failing = failing
    async def post_json(self, *args, **kwargs):
        if self.failing: return HttpResponse(500, {}, b'{"error":"unavailable"}')
        return HttpResponse(200, {}, b'{"id":"ollama-response","model":"model","choices":[{"message":{"role":"assistant","content":"compliant"},"finish_reason":"stop"}],"usage":{"prompt_tokens":1,"completion_tokens":2}}')
    async def stream_json(self, *args, **kwargs):
        payload = b'data: {"id":"ollama-response","model":"model","choices":[{"delta":{"role":"assistant","content":"compliant"},"finish_reason":null}]}\n\ndata: {"choices":[{"delta":{},"finish_reason":"stop"}]}\n\ndata: [DONE]\n\n'
        return _Context(_StreamResponse((payload,)))


def _config(name: str) -> ProviderInstanceConfig:
    return ProviderInstanceConfig(name, "ollama", {"base_url": "http://127.0.0.1:1", "outbound_interface": "openai"})


def _execution(name: str) -> CompletionExecution:
    return CompletionExecution(CompletionRequest("model", (Message(MessageRole.USER, (TextContent("hello"),)),)), "model", name, SamplingParameters(), "model")


def _gateway(config):
    return OllamaFactory(transport_factory=lambda: _Transport(config.instance_name.startswith("compliance-error"))).create(config)


async def _isolation(first, second): assert first is not second
async def _cleanup(gateway): return None
async def _cancellation(gateway, execution):
    task = asyncio.create_task(anext(gateway.stream(execution))); task.cancel()
    with pytest.raises(asyncio.CancelledError): await task


@pytest.fixture
def provider_compliance_adapter():
    expected = CompletionResponse("ollama-response", "model", Message(MessageRole.ASSISTANT, (TextContent("compliant"),)), FinishReason.END_TURN, None, Usage(1, 2))
    stream = (ResponseStarted("ollama-response", "model"), TextStarted("text_0"), TextDelta("text_0", "compliant"), TextCompleted("text_0"), ResponseCompleted(FinishReason.END_TURN, None, Usage(None, None)))
    return ComplianceAdapter(OllamaExtension(), _config, ProviderInstanceConfig("bad", "ollama", {"base_url": "bad"}), _execution, expected, _execution, stream, create_gateway=_gateway, error_cases=(ErrorCase(_execution, ProviderUnavailableError),), cancellation_probe=_cancellation, cleanup_probe=_cleanup, isolation_probe=_isolation)


def test_metadata(provider_compliance_adapter): suite.test_metadata_and_compatibility(provider_compliance_adapter)
def test_configuration(provider_compliance_adapter): suite.test_capabilities_and_configuration(provider_compliance_adapter)
async def test_factory(provider_compliance_adapter): await suite.test_factory_construction_and_isolation(provider_compliance_adapter)
async def test_completion(provider_compliance_adapter): await suite.test_completion(provider_compliance_adapter)
async def test_stream(provider_compliance_adapter): await suite.test_streaming(provider_compliance_adapter)
async def test_lifecycle(provider_compliance_adapter): await suite.test_health_and_cleanup(provider_compliance_adapter); await suite.test_cancellation_propagates(provider_compliance_adapter); await suite.test_typed_errors(provider_compliance_adapter)
