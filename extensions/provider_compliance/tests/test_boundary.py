from __future__ import annotations

import ast
from pathlib import Path

import pytest

from llm_proxy_provider_compliance import ComplianceAdapter, ErrorCase, ManagementCase, ProviderComplianceViolation
from llm_proxy.provider_extensions import ProviderCapabilities, ProviderExtensionMetadata, SdkCompatibility


def test_distribution_imports_only_public_gateway_sdk() -> None:
    source_root = Path(__file__).parents[1] / "src/llm_proxy_provider_compliance"
    imports = {
        node.module
        for path in source_root.glob("*.py")
        for node in ast.walk(ast.parse(path.read_text()))
        if isinstance(node, ast.ImportFrom) and node.module
    }
    gateway_imports = {module for module in imports if module.startswith("llm_proxy")}
    assert gateway_imports <= {"llm_proxy.provider_extensions"}


def _extension(capabilities: ProviderCapabilities):
    return type("SampleExtension", (), {
        "metadata": ProviderExtensionMetadata("sample", "Sample", "sample", "1.0.0"),
        "compatibility": SdkCompatibility("0.1.0"),
        "capabilities": capabilities,
        "factory": object(),
    })()


async def _isolate(first, second) -> None:
    assert first is not second


async def _cleanup(gateway) -> None:
    return None


_ERROR_CASES = (ErrorCase(lambda name: None, ValueError),)  # type: ignore[arg-type]
_MANAGEMENT_CASES = (ManagementCase("inspect", {}),)


def test_adapter_rejects_incomplete_declared_capabilities() -> None:
    adapter = ComplianceAdapter(_extension(ProviderCapabilities(streaming=True, cancellation=True)), lambda name: None, None, lambda name: None, None)  # type: ignore[arg-type]
    with pytest.raises(ProviderComplianceViolation, match=r"sample: streaming: declared"):
        adapter.validate_shape()


def test_adapter_rejects_missing_isolation_probe() -> None:
    adapter = ComplianceAdapter(_extension(ProviderCapabilities()), lambda name: None, None, lambda name: None, None, cleanup_probe=_cleanup, error_cases=_ERROR_CASES)  # type: ignore[arg-type]
    with pytest.raises(ProviderComplianceViolation, match=r"sample: configuration: no two-instance isolation"):
        adapter.validate_shape()


def test_adapter_rejects_non_completion_extension() -> None:
    adapter = ComplianceAdapter(_extension(ProviderCapabilities(completion=False)), lambda name: None, None, lambda name: None, None, isolation_probe=_isolate, cleanup_probe=_cleanup)  # type: ignore[arg-type]
    with pytest.raises(ProviderComplianceViolation, match=r"sample: capabilities: completion is mandatory"):
        adapter.validate_shape()


def test_adapter_rejects_partial_package_identity() -> None:
    adapter = ComplianceAdapter(_extension(ProviderCapabilities()), lambda name: None, None, lambda name: None, None, isolation_probe=_isolate, cleanup_probe=_cleanup, error_cases=_ERROR_CASES, package_module="sample")  # type: ignore[arg-type]
    with pytest.raises(ProviderComplianceViolation, match=r"sample: package: module and distribution"):
        adapter.validate_shape()


def test_adapter_rejects_missing_cleanup_and_error_probes() -> None:
    adapter = ComplianceAdapter(_extension(ProviderCapabilities()), lambda name: None, None, lambda name: None, None, isolation_probe=_isolate)  # type: ignore[arg-type]
    with pytest.raises(ProviderComplianceViolation, match=r"sample: cleanup: no deterministic cleanup"):
        adapter.validate_shape()

    adapter = ComplianceAdapter(_extension(ProviderCapabilities()), lambda name: None, None, lambda name: None, None, isolation_probe=_isolate, cleanup_probe=_cleanup)  # type: ignore[arg-type]
    with pytest.raises(ProviderComplianceViolation, match=r"sample: errors: no typed error case"):
        adapter.validate_shape()


def test_adapter_rejects_missing_management_targeting_probe() -> None:
    adapter = ComplianceAdapter(
        _extension(ProviderCapabilities(management_commands=True)), lambda name: None, None,
        lambda name: None, None, isolation_probe=_isolate, cleanup_probe=_cleanup,
        error_cases=_ERROR_CASES, management_cases=_MANAGEMENT_CASES,
    )  # type: ignore[arg-type]
    with pytest.raises(ProviderComplianceViolation, match=r"sample: management: no exact-instance targeting"):
        adapter.validate_shape()
