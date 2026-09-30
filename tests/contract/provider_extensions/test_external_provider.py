from __future__ import annotations

import ast
import asyncio
import importlib
import sys
from pathlib import Path

import pytest

from llm_proxy.domain.content import TextContent
from llm_proxy.domain.events import ResponseCompleted, ResponseStarted, TextCompleted, TextDelta, TextStarted
from llm_proxy.domain.messages import Message, MessageRole
from llm_proxy.domain.requests import CompletionRequest, SamplingParameters
from llm_proxy.domain.responses import FinishReason, Usage
from llm_proxy.provider_extensions import CompletionExecution, ProviderInstanceConfig, ProviderUnavailableError, validate_sdk_compatibility


@pytest.fixture
def extension_module(monkeypatch: pytest.MonkeyPatch):
    fixture_root = Path(__file__).parents[2] / "fixtures"
    monkeypatch.syspath_prepend(str(fixture_root))
    sys.modules.pop("external_provider.deterministic_provider", None)
    return importlib.import_module("external_provider.deterministic_provider")


def execution() -> CompletionExecution:
    request = CompletionRequest("public-model", (Message(MessageRole.USER, (TextContent("hello"),)),))
    parameters = SamplingParameters()
    return CompletionExecution(request, "upstream-model", "fixture-instance", parameters, "public-model")


def test_external_provider_imports_no_gateway_private_module_or_framework() -> None:
    source = (Path(__file__).parents[2] / "fixtures/external_provider/deterministic_provider.py").read_text()
    imported = {node.module for node in ast.walk(ast.parse(source)) if isinstance(node, ast.ImportFrom) and node.module is not None}
    assert "llm_proxy.provider_extensions" in imported
    assert not any(module.startswith("llm_proxy.application") for module in imported)
    assert not any(module.startswith("llm_proxy.providers") for module in imported)
    assert not any(module.startswith("llm_proxy.infrastructure") for module in imported)
    assert "fastapi" not in imported


async def test_external_provider_constructs_and_emits_canonical_values(extension_module) -> None:
    extension = extension_module.DeterministicExtension()
    gateway = extension.factory.create(ProviderInstanceConfig("fixture-instance", "example.deterministic", {"answer": "fixed"}))
    response = await gateway.complete(execution())
    events = [event async for event in gateway.stream(execution())]
    assert response.model == "public-model"
    assert response.message.content == (TextContent("external response"),)
    assert events == [ResponseStarted("external-response-1", "public-model"), TextStarted("text-1"), TextDelta("text-1", "external stream"), TextCompleted("text-1"), ResponseCompleted(FinishReason.END_TURN, None, Usage(1, 2))]


def test_external_provider_propagates_construction_failure(extension_module) -> None:
    with pytest.raises(ValueError, match="construction failure"):
        extension_module.DeterministicExtension().factory.create(ProviderInstanceConfig("fixture-instance", "example.deterministic", {"fail_construction": True}))


async def test_external_provider_uses_typed_errors_and_propagates_cancellation(extension_module) -> None:
    extension = extension_module.DeterministicExtension()
    failing = extension.factory.create(ProviderInstanceConfig("fixture-instance", "example.deterministic", {"fail_completion": True}))
    with pytest.raises(ProviderUnavailableError, match="deterministic provider unavailable"):
        await failing.complete(execution())

    streaming = extension.factory.create(ProviderInstanceConfig("fixture-instance", "example.deterministic", {"await_cancellation": True}))
    pending = asyncio.create_task(anext(streaming.stream(execution())))
    await asyncio.sleep(0)
    pending.cancel()
    with pytest.raises(asyncio.CancelledError):
        await pending


def test_external_provider_declares_a_compatible_sdk_range(extension_module) -> None:
    assert validate_sdk_compatibility(extension_module.DeterministicExtension.compatibility).compatible
