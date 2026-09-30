"""The stable provider-author integration seam.

This module deliberately imports only ``llm_proxy.provider_extensions``.
"""
from __future__ import annotations

from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from typing import Any, Literal, Type

from llm_proxy.provider_extensions import CompletionExecution, CompletionResponse, ProviderExtension, ProviderHealth, ProviderInstanceConfig


class ProviderComplianceViolation(AssertionError):
    """A diagnostic which always identifies the provider and contract area."""


@dataclass(frozen=True, slots=True)
class ManagementCase:
    name: str
    payload: Mapping[str, Any]
    expected_result: Mapping[str, Any] | None = None


@dataclass(frozen=True, slots=True)
class ErrorCase:
    """A deterministic public execution which must raise a typed error."""

    execution: Callable[[str], CompletionExecution]
    error_type: Type[Exception]
    operation: Literal["completion", "stream"] = "completion"


CancellationProbe = Callable[[Any, CompletionExecution], Awaitable[None]]
IsolationProbe = Callable[[Any, Any], Awaitable[None]]
ManagementIsolationProbe = Callable[[Any, Any], Awaitable[None]]


@dataclass(frozen=True, slots=True)
class ComplianceAdapter:
    """Provider-owned deterministic observations required by the generic suite.

    ``create_gateway`` is intentionally provider-owned: it permits a fake
    transport without exposing gateway runtime services to this package.
    """

    extension: ProviderExtension
    valid_config: Callable[[str], ProviderInstanceConfig]
    invalid_config: ProviderInstanceConfig | None
    completion_execution: Callable[[str], CompletionExecution]
    expected_response: CompletionResponse
    stream_execution: Callable[[str], CompletionExecution] | None = None
    expected_stream: tuple[Any, ...] | None = None
    create_gateway: Callable[[ProviderInstanceConfig], Any] | None = None
    management_cases: tuple[ManagementCase, ...] = ()
    management_isolation_probe: ManagementIsolationProbe | None = None
    error_cases: tuple[ErrorCase, ...] = ()
    health: Callable[[Any], Awaitable[ProviderHealth]] | None = None
    cleanup_probe: Callable[[Any], Awaitable[None]] | None = None
    cancellation_probe: CancellationProbe | None = None
    isolation_probe: IsolationProbe | None = None
    package_module: str | None = None
    package_distribution: str | None = None

    @property
    def provider_id(self) -> str:
        return self.extension.metadata.extension_id

    def gateway(self, instance_name: str) -> Any:
        config = self.valid_config(instance_name)
        return self.create_gateway(config) if self.create_gateway else self.extension.factory.create(config)

    def validate_shape(self) -> None:
        """Reject incomplete capability declarations before provider execution."""
        capabilities = self.extension.capabilities
        if not capabilities.completion:
            raise ProviderComplianceViolation(f"{self.provider_id}: capabilities: completion is mandatory for provider extensions")
        if capabilities.streaming and (self.stream_execution is None or self.expected_stream is None):
            raise ProviderComplianceViolation(f"{self.provider_id}: streaming: declared but no canonical stream case supplied")
        if capabilities.cancellation and capabilities.streaming and self.cancellation_probe is None:
            raise ProviderComplianceViolation(f"{self.provider_id}: cancellation: declared but no deterministic cancellation probe supplied")
        if self.cleanup_probe is None:
            raise ProviderComplianceViolation(f"{self.provider_id}: cleanup: no deterministic cleanup probe supplied")
        if not self.error_cases:
            raise ProviderComplianceViolation(f"{self.provider_id}: errors: no typed error case supplied")
        if capabilities.health and self.health is None:
            raise ProviderComplianceViolation(f"{self.provider_id}: health: declared but no health invocation supplied")
        if capabilities.management_commands and not self.management_cases:
            raise ProviderComplianceViolation(f"{self.provider_id}: management: declared but no management case supplied")
        if capabilities.management_commands and self.management_isolation_probe is None:
            raise ProviderComplianceViolation(f"{self.provider_id}: management: no exact-instance targeting probe supplied")
        if self.isolation_probe is None:
            raise ProviderComplianceViolation(f"{self.provider_id}: configuration: no two-instance isolation probe supplied")
        if (self.package_module is None) != (self.package_distribution is None):
            raise ProviderComplianceViolation(f"{self.provider_id}: package: module and distribution identity must be supplied together")
