"""Provider-owned bridge to the reusable provider compliance suite."""
from __future__ import annotations

import asyncio

import pytest

from llm_proxy_mock_provider.extension import extension
from llm_proxy.provider_extensions import (
    CompletionExecution, CompletionResponse, FinishReason,
    ManagementCommandContext, Message, MessageRole,
    ProviderInstanceConfig, ProviderUnavailableError, ResponseCompleted,
    ResponseStarted, TextCompleted, TextContent, TextDelta, TextStarted, Usage,
)

from llm_proxy_provider_compliance import ComplianceAdapter, ErrorCase, ManagementCase
from llm_proxy_provider_compliance import suite


def _scenario(*, error: bool = False, latency_ms: int = 0, models: tuple[str, ...] = ()) -> dict:
    return {"version": 1, "models": list(models), "rules": [{"id": "reply", "responses": [{"id": "mock-response-1", "text": "compliant", "usage": {"input_tokens": 1, "output_tokens": 2}, "error": error, "latency_ms": latency_ms}]}]}


def _config(name: str) -> ProviderInstanceConfig:
    return ProviderInstanceConfig(name, "deterministic-mock", {"scenario": _scenario(error=name.startswith("compliance-error"), latency_ms=50 if name == "compliance-cancellation" else 0, models=(f"{name}-model",))})


def _execution(name: str) -> CompletionExecution:
    from llm_proxy.domain.requests import CompletionRequest, SamplingParameters
    return CompletionExecution(CompletionRequest("model", (Message(MessageRole.USER, (TextContent("hello"),)),)), "upstream", name, SamplingParameters(), "model")


async def _isolation(first, second) -> None:
    await first.complete(_execution("compliance-one"))
    assert (await second.execute_management_command("inspect_captured_requests", {}, ManagementCommandContext("compliance-two", "isolation")))["requests"] == []


async def _cleanup(gateway) -> None:
    assert isinstance(await gateway.execute_management_command("inspect_captured_requests", {}, ManagementCommandContext("compliance-cleanup", "cleanup")), dict)


async def _cancellation(gateway, execution) -> None:
    task = asyncio.create_task(anext(gateway.stream(execution)))
    await asyncio.sleep(0)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task


async def _management_isolation(first, second) -> None:
    await first.execute_management_command("enqueue_response", {"response": {"text": "first"}}, ManagementCommandContext("compliance-management", "one"))
    assert (await second.execute_management_command("inspect_captured_requests", {}, ManagementCommandContext("compliance-management-other", "two")))["requests"] == []


@pytest.fixture
def provider_compliance_adapter() -> ComplianceAdapter:
    expected = CompletionResponse("mock-response-1", "model", Message(MessageRole.ASSISTANT, (TextContent("compliant"),)), FinishReason.END_TURN, None, Usage(1, 2))
    stream = (ResponseStarted("mock-response-1", "model"), TextStarted("mock-block-1-1"), TextDelta("mock-block-1-1", "compliant"), TextCompleted("mock-block-1-1"), ResponseCompleted(FinishReason.END_TURN, None, Usage(1, 2)))
    return ComplianceAdapter(extension, _config, ProviderInstanceConfig("bad", "deterministic-mock", {"capture_limit": 0}), _execution, expected, _execution, stream, management_cases=(ManagementCase("inspect_captured_requests", {}, {"requests": []}),), error_cases=(ErrorCase(_execution, ProviderUnavailableError),), cancellation_probe=_cancellation, cleanup_probe=_cleanup, isolation_probe=_isolation, management_isolation_probe=_management_isolation, package_module="llm_proxy_mock_provider", package_distribution="llm-proxy-mock-provider")


def test_metadata(provider_compliance_adapter): suite.test_metadata_and_compatibility(provider_compliance_adapter)
def test_configuration(provider_compliance_adapter): suite.test_capabilities_and_configuration(provider_compliance_adapter)
async def test_factory(provider_compliance_adapter): await suite.test_factory_construction_and_isolation(provider_compliance_adapter)
async def test_completion(provider_compliance_adapter): await suite.test_completion(provider_compliance_adapter)
async def test_model_listing(provider_compliance_adapter): await suite.test_model_listing(provider_compliance_adapter)


async def test_model_listing_isolated_per_instance(provider_compliance_adapter):
    first = provider_compliance_adapter.gateway("first")
    second = provider_compliance_adapter.gateway("second")

    assert [model.upstream_model for model in await first.list_models()] == ["first-model"]
    assert [model.upstream_model for model in await second.list_models()] == ["second-model"]
async def test_stream(provider_compliance_adapter): await suite.test_streaming(provider_compliance_adapter)
async def test_lifecycle(provider_compliance_adapter): await suite.test_health_and_cleanup(provider_compliance_adapter); await suite.test_cancellation_propagates(provider_compliance_adapter); await suite.test_typed_errors(provider_compliance_adapter)
async def test_management(provider_compliance_adapter): await suite.test_management_commands(provider_compliance_adapter)
