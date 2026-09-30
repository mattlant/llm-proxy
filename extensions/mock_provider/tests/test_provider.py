from __future__ import annotations

import asyncio

import pytest

from llm_proxy_mock_provider.extension import extension
from llm_proxy_mock_provider.provider import parse_scenario
from llm_proxy.provider_extensions import (
    CompletionExecution, ManagementCommandContext, Message, MessageRole, ProviderInstanceConfig,
    SourceInterface, TextContent,
)
from llm_proxy.domain.requests import CompletionRequest, SamplingParameters


def _execution(*, stream: bool = False, source: SourceInterface = SourceInterface.OPENAI) -> CompletionExecution:
    request = CompletionRequest("model", (Message(MessageRole.USER, (TextContent("hello"),)),), stream=stream)
    return CompletionExecution(request, "upstream", "mock", SamplingParameters(), "model", source_interface=source)


def _gateway():
    return extension.factory.create(ProviderInstanceConfig("mock", "deterministic-mock", {"scenario": {"version": 1, "rules": [{"id": "openai", "priority": 1, "match": {"source_interface": "openai", "text": "hello"}, "responses": [{"content": [{"type": "text", "text": "hi"}, {"type": "tool_call", "name": "weather", "arguments": {"city": "Toronto"}}], "usage": {"input_tokens": 1, "output_tokens": 2}}]}]}}))


async def test_response_stream_and_instance_local_commands() -> None:
    gateway = _gateway()
    response = await gateway.complete(_execution())
    assert response.id == "mock-response-1"
    assert response.message.content[0] == TextContent("hi")
    events = [event async for event in gateway.stream(_execution(stream=True))]
    assert [type(event).__name__ for event in events] == ["ResponseStarted", "TextStarted", "TextDelta", "TextCompleted", "ToolCallStarted", "ToolCallArgumentsDelta", "ToolCallCompleted", "ResponseCompleted"]
    result = await gateway.execute_management_command("inspect_captured_requests", {}, ManagementCommandContext("mock", "one"))
    assert [item["call_index"] for item in result["requests"]] == [1, 2]
    assert result["requests"][0]["metadata"] == {}


def test_scenario_is_strict_and_non_executable() -> None:
    with pytest.raises(ValueError, match="must contain only"):
        parse_scenario({"version": 1, "rules": [], "$ref": "file:///nope"})
    with pytest.raises(ValueError, match="requires responses"):
        parse_scenario({"version": 1, "rules": [{"id": "bad"}]})


async def test_response_model_and_message_history_matching_are_canonical() -> None:
    scenario = {
        "version": 1,
        "rules": [{
            "id": "target", "match": {
                "response_model": "model", "messages": [{"role": "user", "content": [{"type": "text", "text": "hello"}]}],
            }, "responses": [{"text": "matched"}],
        }],
    }
    gateway = extension.factory.create(ProviderInstanceConfig("mock", "deterministic-mock", {"scenario": scenario}))
    assert (await gateway.complete(_execution())).message.content == (TextContent("matched"),)


async def test_capture_redaction_truncation_and_concurrent_reservations() -> None:
    scenario = {"version": 1, "rules": [{"id": "reply", "responses": [{"text": "ok", "latency_ms": 1}]}]}
    gateway = extension.factory.create(ProviderInstanceConfig("mock", "deterministic-mock", {"scenario": scenario, "capture_limit": 2, "capture_max_bytes": 128, "redacted_metadata_keys": ["secret"]}))
    long_request = CompletionRequest("model", (Message(MessageRole.USER, (TextContent("x" * 512),)),), metadata={"secret": "no", "safe": "yes"})
    execution = CompletionExecution(long_request, "upstream", "mock", SamplingParameters(), "model")
    await asyncio.gather(*(gateway.complete(execution) for _ in range(3)))
    captured = (await gateway.execute_management_command("inspect_captured_requests", {}, ManagementCommandContext("mock", "capture")))["requests"]
    assert [item["call_index"] for item in captured] == [2, 3]
    assert all(item["truncated"] and item["text"].endswith("...[truncated]") for item in captured)
    assert all(item["metadata"] == {"safe": "yes"} for item in captured)


@pytest.mark.parametrize("scenario", [
    {"version": 1, "rules": [{"id": "rule", "match": {"stream": "yes"}, "responses": [{"text": "x"}]}]},
    {"version": 1, "rules": [{"id": "rule", "responses": [{"content": [{"type": "tool_call", "name": "x", "arguments": []}]}]}]},
    {"version": 1, "rules": [{"id": "rule", "responses": [{"text": "x", "usage": {"input_tokens": -1}}]}]},
    {"version": 1, "rules": [{"id": "rule", "responses": [{"text": "x"}]}], "expectations": [{"id": "missing", "rule_id": "nope", "count": 1}]},
])
def test_scenario_rejects_invalid_semantics_before_publication(scenario) -> None:
    with pytest.raises(ValueError):
        parse_scenario(scenario)
