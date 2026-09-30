from __future__ import annotations

import base64
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx
import pytest

from llm_proxy.application.model_registry import ModelRegistry
from llm_proxy.application.model_resolver import ModelResolver
from llm_proxy.application.policy_context import RequestPolicyContext
from llm_proxy.application.policy_engine import PolicyEngine
from llm_proxy.configuration.loader import ConfigurationLoader
from llm_proxy.configuration.models import InterfaceName
from llm_proxy.configuration.store import ConfigurationSnapshot
from llm_proxy.configuration.validator import ConfigurationValidator
from llm_proxy.domain.content import TextContent, ToolCallContent, ToolResultContent
from llm_proxy.domain.events import ResponseCompleted, ResponseStarted, TextDelta, ToolCallCompleted, ToolCallStarted
from llm_proxy.domain.messages import Message, MessageRole
from llm_proxy.domain.requests import AutomaticToolChoice, CompletionRequest, SamplingParameters
from llm_proxy.domain.responses import FinishReason
from llm_proxy.domain.tools import ToolDefinition
from llm_proxy.infrastructure.http_client import HttpStreamResponse, HttpxAsyncHttpTransport
from llm_proxy.observability.payload_trace import PayloadTraceRecorder, TraceMode
from llm_proxy.provider_extensions import CompletionExecution, ProviderExecutionPolicy, ProviderInstanceConfig
from llm_proxy.providers.ollama.extension import OllamaFactory
from tests.unit.configuration.test_validator import realistic_config


def traced_bytes(value: dict) -> bytes:
    if value["encoding"] == "utf-8":
        return value["data"].encode("utf-8")
    assert value["encoding"] == "base64"
    return base64.b64decode(value["data"])


SCHEMA = {
    "type": "object",
    "properties": {"path": {"type": "string"}},
    "required": ["path"],
    "additionalProperties": False,
    "$defs": {"Path": {"type": "string"}},
    "$ref": "#/$defs/Path",
}


def services() -> ConfigurationSnapshot:
    config_data = realistic_config()
    config_data["providers"] = {"test-ollama": next(iter(config_data["providers"].values()))}
    for model in config_data["models"].values():
        model["provider"] = "test-ollama"
    config = ConfigurationLoader().load_mapping(config_data)
    ConfigurationValidator().validate(config)
    registry = ModelRegistry(config)
    resolver = ModelResolver(registry, config)
    return ConfigurationSnapshot(
        config=config,
        registry=registry,
        resolver=resolver,
        policy_engine=PolicyEngine(config, registry),
        loaded_at=1.0,
        source_path=Path("test-config.yaml"),
    )


def completion_execution(snapshot: ConfigurationSnapshot) -> CompletionExecution:
    requested_model = "local-opus"
    base = snapshot.resolver.resolve(InterfaceName.ANTHROPIC, requested_model)
    request = CompletionRequest(
        model=requested_model,
        messages=(
            Message(MessageRole.SYSTEM, (TextContent("You are a file assistant."),)),
            Message(MessageRole.USER, (TextContent("Read README.md."),)),
            Message(
                MessageRole.ASSISTANT,
                (TextContent("I will use tools."), ToolCallContent("call_1", "Read", {"path": "README.md"})),
            ),
            Message(MessageRole.TOOL, (ToolResultContent("call_1", (TextContent("contents"),)),)),
        ),
        tools=(ToolDefinition("Read", "Read a file", SCHEMA),),
        tool_choice=AutomaticToolChoice(),
        parameters=SamplingParameters(),
        stream=True,
    )
    context = RequestPolicyContext(
        InterfaceName.ANTHROPIC,
        requested_model,
        base.canonical_model,
        request.messages,
        ("Read",),
        {},
    )
    final = snapshot.policy_engine.apply(context, base)
    return CompletionExecution(
        request=request,
        upstream_model=final.upstream_model,
        provider_instance_name=final.provider_instance_name,
        parameters=final.parameters,
        response_model=request.model,
        policy=ProviderExecutionPolicy(
            structured_output_mode=final.compatibility.structured_output.value,
            expose_thinking=final.compatibility.expose_thinking,
            native_tools=final.compatibility.native_tools,
        ),
    )


def make_client(transport: object):
    return OllamaFactory(transport_factory=lambda: transport).create(
        ProviderInstanceConfig("test-ollama", "ollama", {"base_url": "http://provider.test:11434"})
    )


def non_stream_response() -> dict[str, Any]:
    return {
        "id": "chatcmpl-proof",
        "model": "hf.co/lordx64/Qwable-v2-GGUF:Q4_K_M",
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": "The file is available.",
                "tool_calls": [{"id": "call_2", "type": "function", "function": {"name": "Read", "arguments": '{"path":"README.md"}'}}],
            },
            "finish_reason": "tool_calls",
        }],
        "usage": {"prompt_tokens": 42, "completion_tokens": 9},
    }


@pytest.mark.asyncio
async def test_resolved_anthropic_alias_executes_non_streaming_through_ollama() -> None:
    snapshot = services()
    execution = completion_execution(snapshot)
    observed: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        observed["url"] = str(request.url)
        observed["body"] = json.loads(request.read())
        return httpx.Response(200, json=non_stream_response())

    client = make_client(HttpxAsyncHttpTransport(lambda: httpx.AsyncClient(transport=httpx.MockTransport(handler))))
    response = await client.complete(execution)

    assert observed["url"] == "http://provider.test:11434/v1/chat/completions"
    assert observed["body"]["model"] == "hf.co/lordx64/Qwable-v2-GGUF:Q4_K_M"
    assert observed["body"]["temperature"] == 0.7
    assert observed["body"]["tools"][0]["function"]["parameters"] == SCHEMA
    assert "local-opus" not in json.dumps(observed["body"])
    assert response.finish_reason is FinishReason.TOOL_USE
    assert response.message.content[0] == TextContent("The file is available.")
    assert isinstance(response.message.content[1], ToolCallContent)
    assert response.usage.input_tokens == 42
    assert response.usage.output_tokens == 9


@pytest.mark.asyncio
async def test_provider_trace_captures_mapped_request_and_raw_response(caplog) -> None:
    caplog.set_level(logging.DEBUG, logger="llm-proxy.payload")
    execution = completion_execution(services())
    recorder = PayloadTraceRecorder()
    context = recorder.begin(TraceMode.BOTH)
    tokens = recorder.bind(context)
    response_body = b'{ "id":"chatcmpl-proof", "model":"hf.co/lordx64/Qwable-v2-GGUF:Q4_K_M", "choices":[{"index":0,"message":{"role":"assistant","content":"The file is available."},"finish_reason":"stop"}] }'
    response_headers = [(b"X-Dupe", b"one"), (b"x-dupe", b"two")]
    observed: dict[str, bytes] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        observed["request"] = request.read()
        return httpx.Response(200, headers=response_headers, content=response_body)

    try:
        client = make_client(HttpxAsyncHttpTransport(lambda: httpx.AsyncClient(transport=httpx.MockTransport(handler))))
        await client.complete(execution)
    finally:
        recorder.reset(tokens)
    records = [json.loads(record.message) for record in caplog.records if record.name == "llm-proxy.payload"]
    assert [record["stage"] for record in records] == ["provider_request", "provider_request_raw", "provider_response_raw", "provider_response"]
    assert {record["transaction_id"] for record in records} == {context.transaction_id}
    raw_request = next(record for record in records if record["stage"] == "provider_request_raw")
    raw_response = next(record for record in records if record["stage"] == "provider_response_raw")
    assert traced_bytes(raw_request["body"]) == observed["request"]
    assert traced_bytes(raw_response["body"]) == response_body
    assert [(pair["name"]["data"].encode(), pair["value"]["data"].encode()) for pair in raw_response["http"]["headers"][:2]] == response_headers


@dataclass
class ProofStreamResponse(HttpStreamResponse):
    chunks: list[bytes]
    status_code: int = 200

    @property
    def headers(self) -> dict[str, str]:
        return {}

    async def aiter_bytes(self):
        for chunk in self.chunks:
            yield chunk

    async def aclose(self) -> None:
        pass


class ProofStreamContext:
    def __init__(self, response: ProofStreamResponse) -> None:
        self.response = response

    async def __aenter__(self) -> ProofStreamResponse:
        return self.response

    async def __aexit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        pass


class ProofStreamTransport:
    def __init__(self, records: list[dict[str, Any]]) -> None:
        payload = b"".join(f"data: {json.dumps(record)}\n\n".encode() for record in records)
        self.context = ProofStreamContext(ProofStreamResponse([payload]))

    async def stream_json(self, url: str, headers: Any, body: Any, timeout: float | None, *, trace_provider: str | None = None):
        return self.context


@pytest.mark.asyncio
async def test_provider_stream_trace_reconstructs_exact_fragments(caplog) -> None:
    caplog.set_level(logging.DEBUG, logger="llm-proxy.payload")
    payload = b'data: {"id":"stream-proof","model":"hf.co/lordx64/Qwable-v2-GGUF:Q4_K_M","choices":[{"delta":{"role":"assistant"},"finish_reason":null}]}\n\ndata: {"choices":[{"delta":{},"finish_reason":"stop"}]}\n\ndata: [DONE]\n\n'
    fragments = [payload[:19], payload[19:97], payload[97:]]
    transport = ProofStreamTransport([])
    transport.context = ProofStreamContext(ProofStreamResponse(fragments))
    recorder = PayloadTraceRecorder()
    context = recorder.begin(TraceMode.RAW)
    tokens = recorder.bind(context)
    try:
        events = [event async for event in make_client(transport).stream(completion_execution(services()))]
    finally:
        recorder.reset(tokens)
    assert isinstance(events[0], ResponseStarted)
    records = [json.loads(record.message) for record in caplog.records if record.name == "llm-proxy.payload"]
    assert b"".join(traced_bytes(record["body"]) for record in records if record["stage"] == "provider_stream_bytes_raw") == payload


@pytest.mark.asyncio
async def test_resolved_anthropic_alias_executes_parallel_streaming_tool_transcript() -> None:
    snapshot = services()
    execution = completion_execution(snapshot)
    records = [
        {"id": "stream-proof", "model": "hf.co/lordx64/Qwable-v2-GGUF:Q4_K_M", "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}]},
        {"choices": [{"index": 0, "delta": {"content": "I will use tools.", "tool_calls": [{"index": 0, "id": "call_", "function": {"name": "Re"}}, {"index": 1, "id": "call_", "function": {"name": "W"}}]}, "finish_reason": None}]},
        {"choices": [{"index": 0, "delta": {"tool_calls": [{"index": 0, "id": "1", "function": {"name": "ad"}}, {"index": 1, "id": "2", "function": {"name": "rite"}}]}, "finish_reason": None}]},
        {"choices": [{"index": 0, "delta": {"tool_calls": [{"index": 0, "function": {"arguments": "{\"path\":"}}, {"index": 1, "function": {"arguments": "{\"path\":"}}]}, "finish_reason": None}]},
        {"choices": [{"index": 0, "delta": {"tool_calls": [{"index": 0, "function": {"arguments": "\"README.md\"}"}}, {"index": 1, "function": {"arguments": "\"output.txt\"}"}}]}, "finish_reason": None}]},
        {"choices": [], "usage": {"prompt_tokens": 50, "completion_tokens": 12}},
        {"choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}]},
    ]
    client = make_client(ProofStreamTransport(records))

    events = [event async for event in client.stream(execution)]

    assert isinstance(events[0], ResponseStarted)
    assert any(isinstance(event, TextDelta) and event.text == "I will use tools." for event in events)
    starts = [event for event in events if isinstance(event, ToolCallStarted)]
    completions = [event for event in events if isinstance(event, ToolCallCompleted)]
    assert [(event.tool_call_id, event.name) for event in starts] == [("call_1", "Read"), ("call_2", "Write")]
    assert [(event.tool_call_id, dict(event.arguments)) for event in completions] == [
        ("call_1", {"path": "README.md"}),
        ("call_2", {"path": "output.txt"}),
    ]
    assert isinstance(events[-1], ResponseCompleted)
    assert events[-1].finish_reason is FinishReason.TOOL_USE
    assert events[-1].usage.input_tokens == 50
    assert events[-1].usage.output_tokens == 12
