"""Test-only broken adapters proving focused compliance diagnostics."""
from __future__ import annotations

import ast

import pytest

from llm_proxy_provider_compliance import ComplianceAdapter, ErrorCase, ManagementCase, ProviderComplianceViolation, observe_stream
from llm_proxy_provider_compliance import suite
from llm_proxy.provider_extensions import (
    CompletionExecution, ProviderCapabilities, ProviderExtensionMetadata,
    ProviderInstanceConfig, ResponseStarted, SdkCompatibility,
)


class _Extension:
    metadata = ProviderExtensionMetadata("broken", "Broken", "broken", "1")
    compatibility = SdkCompatibility("999.0.0")
    capabilities = ProviderCapabilities()
    factory = object()


async def _cleanup(_gateway):
    raise AssertionError("resource leaked")


async def _leaking_isolation(_first, _second):
    raise AssertionError("cross-instance state leaked")


async def _swallowed_cancellation(_gateway, _execution):
    raise AssertionError("cancellation swallowed")


def _adapter(**overrides):
    values = dict(
        extension=_Extension(), valid_config=lambda name: ProviderInstanceConfig(name, "broken", {}),
        invalid_config=None, completion_execution=lambda name: None, expected_response=None,
        cleanup_probe=_cleanup, isolation_probe=_leaking_isolation,
        error_cases=(ErrorCase(lambda name: None, ValueError),),
        create_gateway=lambda config: object(),
    )
    values.update(overrides)
    return ComplianceAdapter(**values)  # type: ignore[arg-type]


def test_incompatible_sdk_declaration_is_rejected() -> None:
    with pytest.raises(ProviderComplianceViolation, match="compatibility"):
        suite.test_metadata_and_compatibility(_adapter())


async def test_malformed_stream_order_is_rejected() -> None:
    async def events():
        yield ResponseStarted("id", "model")
        yield ResponseStarted("other", "model")
    with pytest.raises(ProviderComplianceViolation, match="duplicate ResponseStarted"):
        await observe_stream("broken-stream", events())


async def test_cancellation_swallowing_is_rejected() -> None:
    adapter = _adapter(extension=type("Cancel", (), {"metadata": _Extension.metadata, "compatibility": SdkCompatibility("0.1.0"), "capabilities": ProviderCapabilities(streaming=True, cancellation=True), "factory": object()})(), stream_execution=lambda name: None, expected_stream=(), cancellation_probe=_swallowed_cancellation)
    with pytest.raises(AssertionError, match="cancellation swallowed"):
        await suite.test_cancellation_propagates(adapter)


async def test_cross_instance_leak_is_rejected() -> None:
    class Factory:
        def create(self, config): return object()
    adapter = _adapter(extension=type("Isolated", (), {"metadata": _Extension.metadata, "compatibility": SdkCompatibility("0.1.0"), "capabilities": ProviderCapabilities(), "factory": Factory()})())
    with pytest.raises(AssertionError, match="cross-instance state leaked"):
        await suite.test_factory_construction_and_isolation(adapter)


async def test_cleanup_leak_is_rejected() -> None:
    adapter = _adapter()
    with pytest.raises(AssertionError, match="resource leaked"):
        await suite.test_health_and_cleanup(adapter)


def test_private_gateway_import_is_detected() -> None:
    modules = {node.module for node in ast.walk(ast.parse("from llm_proxy.application import registry")) if isinstance(node, ast.ImportFrom)}
    assert any(module.startswith("llm_proxy.application") for module in modules)


async def test_invalid_management_descriptor_is_rejected() -> None:
    class Gateway:
        management_commands = (object(),)
    extension = type("Managed", (), {"metadata": _Extension.metadata, "compatibility": SdkCompatibility("0.1.0"), "capabilities": ProviderCapabilities(management_commands=True), "factory": object()})()
    adapter = _adapter(extension=extension, create_gateway=lambda config: Gateway(), management_cases=(ManagementCase("bad", {}),), management_isolation_probe=lambda first, second: None)
    with pytest.raises(ProviderComplianceViolation, match="descriptor is not"):
        await suite.test_management_commands(adapter)
