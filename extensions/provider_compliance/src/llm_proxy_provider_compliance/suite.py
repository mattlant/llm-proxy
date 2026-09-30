"""Collectable pytest compliance suite.

Provider tests expose ``provider_compliance_adapter`` and import this module's
tests (``from llm_proxy_provider_compliance.suite import *``).
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
from importlib import import_module
from importlib.metadata import PackageNotFoundError, version

import pytest

from llm_proxy.provider_extensions import (
    PROVIDER_SDK_VERSION, CompletionResponse, ManagementCommandContext,
    ManagementCommandDescriptor, ProviderCapabilities, ProviderExtensionMetadata, ProviderHealth,
    ProviderInstanceConfig, SdkCompatibility, validate_sdk_compatibility,
    ProviderListedModel, ProviderModelListingGateway,
)

from .adapter import ComplianceAdapter, ProviderComplianceViolation
from .stream import observe_stream

pytestmark = pytest.mark.provider_compliance


def _fail(adapter: ComplianceAdapter, area: str, detail: str) -> None:
    raise ProviderComplianceViolation(f"{adapter.provider_id}: {area}: {detail}")


def _json_compatible(value: object) -> bool:
    if value is None or isinstance(value, (bool, int, float, str)):
        return True
    if isinstance(value, Mapping):
        return all(isinstance(key, str) and _json_compatible(item) for key, item in value.items())
    if isinstance(value, (list, tuple)):
        return all(_json_compatible(item) for item in value)
    return False


def test_metadata_and_compatibility(provider_compliance_adapter: ComplianceAdapter) -> None:
    adapter = provider_compliance_adapter
    adapter.validate_shape()
    extension = adapter.extension
    if not isinstance(extension.metadata, ProviderExtensionMetadata): _fail(adapter, "metadata", "metadata is not ProviderExtensionMetadata")
    if not isinstance(extension.compatibility, SdkCompatibility): _fail(adapter, "compatibility", "compatibility is not SdkCompatibility")
    if not validate_sdk_compatibility(extension.compatibility).compatible: _fail(adapter, "compatibility", validate_sdk_compatibility(extension.compatibility).diagnostic)
    if validate_sdk_compatibility(replace(extension.compatibility, minimum="999.0.0"), PROVIDER_SDK_VERSION).compatible: _fail(adapter, "compatibility", "below-minimum SDK accepted")
    if validate_sdk_compatibility(replace(extension.compatibility, maximum="0.0.1"), PROVIDER_SDK_VERSION).compatible: _fail(adapter, "compatibility", "above-maximum SDK accepted")


def test_installed_package_identity(provider_compliance_adapter: ComplianceAdapter) -> None:
    adapter = provider_compliance_adapter
    if adapter.package_module is None:
        pytest.skip("package identity is not applicable to a non-installed built-in extension")
    try:
        module = import_module(adapter.package_module)
        installed_version = version(adapter.package_distribution or "")
    except (ImportError, PackageNotFoundError) as error:
        _fail(adapter, "package", f"installed package identity unavailable: {error}")
    if not getattr(module, "__file__", None): _fail(adapter, "package", "module has no installed file")
    if installed_version != adapter.extension.metadata.package_version:
        _fail(adapter, "package", f"installed version {installed_version} differs from metadata {adapter.extension.metadata.package_version}")


def test_capabilities_and_configuration(provider_compliance_adapter: ComplianceAdapter) -> None:
    adapter = provider_compliance_adapter
    capabilities = adapter.extension.capabilities
    if not isinstance(capabilities, ProviderCapabilities): _fail(adapter, "capabilities", "not ProviderCapabilities")
    if not capabilities.completion: _fail(adapter, "capabilities", "completion is mandatory")
    if capabilities.parallel_tools and not capabilities.native_tools: _fail(adapter, "capabilities", "parallel_tools without native_tools")
    one, two = adapter.valid_config("compliance-one"), adapter.valid_config("compliance-two")
    if not isinstance(one, ProviderInstanceConfig) or not isinstance(two, ProviderInstanceConfig): _fail(adapter, "configuration", "valid config factory did not return ProviderInstanceConfig")
    if one.instance_name == two.instance_name: _fail(adapter, "configuration", "valid configs are not isolated")
    if one.extension_id != adapter.provider_id or two.extension_id != adapter.provider_id: _fail(adapter, "configuration", "config extension_id differs from extension metadata")
    try:
        one.config["mutation"] = "forbidden"  # type: ignore[index]
    except TypeError:
        pass
    else:
        _fail(adapter, "configuration", "config mapping is mutable")
    if adapter.invalid_config is not None:
        with pytest.raises((TypeError, ValueError)):
            (adapter.create_gateway or adapter.extension.factory.create)(adapter.invalid_config)


@pytest.mark.asyncio
async def test_factory_construction_and_isolation(provider_compliance_adapter: ComplianceAdapter) -> None:
    adapter = provider_compliance_adapter
    if adapter.isolation_probe is None: _fail(adapter, "configuration", "no two-instance isolation probe supplied")
    first, second = adapter.gateway("compliance-one"), adapter.gateway("compliance-two")
    if first is second: _fail(adapter, "configuration", "factory returned the same gateway for distinct instances")
    await adapter.isolation_probe(first, second)


@pytest.mark.asyncio
async def test_completion(provider_compliance_adapter: ComplianceAdapter) -> None:
    adapter = provider_compliance_adapter
    if not adapter.extension.capabilities.completion: pytest.skip("completion not declared")
    execution = adapter.completion_execution("compliance-completion")
    response = await adapter.gateway("compliance-completion").complete(execution)
    if not isinstance(response, CompletionResponse): _fail(adapter, "completion", "did not return CompletionResponse")
    if response != adapter.expected_response: _fail(adapter, "completion", f"normalized response differs: {response!r}")
    if execution != adapter.completion_execution("compliance-completion"):
        _fail(adapter, "completion", "provider mutated CompletionExecution")


@pytest.mark.asyncio
async def test_model_listing(provider_compliance_adapter: ComplianceAdapter) -> None:
    adapter = provider_compliance_adapter
    if not adapter.extension.capabilities.model_listing:
        pytest.skip("model listing not declared")
    gateway = adapter.gateway("compliance-listing")
    if not isinstance(gateway, ProviderModelListingGateway):
        _fail(adapter, "model_listing", "declared but gateway does not implement listing port")
    models = await gateway.list_models()
    if not isinstance(models, tuple) or not all(isinstance(model, ProviderListedModel) for model in models):
        _fail(adapter, "model_listing", "did not return normalized immutable records")
    if len({model.upstream_model for model in models}) != len(models):
        _fail(adapter, "model_listing", "returned duplicate model identifiers")


@pytest.mark.asyncio
async def test_streaming(provider_compliance_adapter: ComplianceAdapter) -> None:
    adapter = provider_compliance_adapter
    if not adapter.extension.capabilities.streaming: pytest.skip("streaming not declared")
    if adapter.stream_execution is None or adapter.expected_stream is None: _fail(adapter, "streaming", "declared but no canonical stream case supplied")
    execution = adapter.stream_execution("compliance-stream")
    events = await observe_stream(adapter.provider_id, adapter.gateway("compliance-stream").stream(execution))
    if events != adapter.expected_stream: _fail(adapter, "streaming", f"normalized event sequence differs: {events!r}")
    if execution != adapter.stream_execution("compliance-stream"):
        _fail(adapter, "streaming", "provider mutated CompletionExecution")


@pytest.mark.asyncio
async def test_management_commands(provider_compliance_adapter: ComplianceAdapter) -> None:
    adapter = provider_compliance_adapter
    if not adapter.extension.capabilities.management_commands:
        if adapter.management_cases: _fail(adapter, "management", "cases supplied while capability is absent")
        pytest.skip("management commands not declared")
    gateway = adapter.gateway("compliance-management")
    descriptors = getattr(gateway, "management_commands", ())
    if not all(isinstance(descriptor, ManagementCommandDescriptor) for descriptor in descriptors): _fail(adapter, "management", "descriptor is not a ManagementCommandDescriptor")
    names = [descriptor.name for descriptor in descriptors]
    if not descriptors or len(names) != len(set(names)): _fail(adapter, "management", "descriptors are absent or non-unique")
    for descriptor in descriptors:
        if not _json_compatible(descriptor.input_schema) or not _json_compatible(descriptor.output_schema): _fail(adapter, "management", f"{descriptor.name} schema is not JSON-compatible")
        try:
            descriptor.input_schema["mutation"] = True  # type: ignore[index]
        except TypeError:
            pass
        else:
            _fail(adapter, "management", f"{descriptor.name} input schema is mutable")
    for case in adapter.management_cases:
        result = await gateway.execute_management_command(case.name, case.payload, ManagementCommandContext("compliance-management", "compliance"))
        if not _json_compatible(result): _fail(adapter, "management", f"{case.name} returned non-JSON-compatible result")
        if case.expected_result is not None:
            if not isinstance(result, Mapping) or dict(result) != dict(case.expected_result): _fail(adapter, "management", f"{case.name} result differs")
    if adapter.management_isolation_probe is None: _fail(adapter, "management", "no exact-instance targeting probe supplied")
    await adapter.management_isolation_probe(gateway, adapter.gateway("compliance-management-other"))


@pytest.mark.asyncio
async def test_health_and_cleanup(provider_compliance_adapter: ComplianceAdapter) -> None:
    adapter = provider_compliance_adapter
    gateway = adapter.gateway("compliance-cleanup")
    if adapter.extension.capabilities.health:
        if adapter.health is None: _fail(adapter, "health", "declared but no health invocation supplied")
        if not isinstance(await adapter.health(gateway), ProviderHealth): _fail(adapter, "health", "did not return ProviderHealth")
    if adapter.cleanup_probe is None: _fail(adapter, "cleanup", "no deterministic cleanup probe supplied")
    await adapter.cleanup_probe(gateway)
    await adapter.cleanup_probe(gateway)


@pytest.mark.asyncio
async def test_cancellation_propagates(provider_compliance_adapter: ComplianceAdapter) -> None:
    adapter = provider_compliance_adapter
    if not adapter.extension.capabilities.cancellation or not adapter.extension.capabilities.streaming: pytest.skip("cancellation stream not declared")
    if adapter.stream_execution is None or adapter.cancellation_probe is None: _fail(adapter, "cancellation", "no deterministic cancellation probe supplied")
    await adapter.cancellation_probe(adapter.gateway("compliance-cancellation"), adapter.stream_execution("compliance-cancellation"))


@pytest.mark.asyncio
async def test_typed_errors(provider_compliance_adapter: ComplianceAdapter) -> None:
    adapter = provider_compliance_adapter
    if not adapter.error_cases: _fail(adapter, "errors", "no typed error case supplied")
    for index, case in enumerate(adapter.error_cases):
        with pytest.raises(case.error_type):
            gateway = adapter.gateway(f"compliance-error-{index}")
            execution = case.execution(f"compliance-error-{index}")
            if case.operation == "completion":
                await gateway.complete(execution)
            else:
                async for _ in gateway.stream(execution):
                    pass
