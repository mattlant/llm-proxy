from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from typing import Mapping

from llm_proxy.configuration.models import CapabilityBinding, CapabilityFailureMode
from llm_proxy.extensions.kernel import ExtensionRegistryError
from llm_proxy.extensions.side_effect import SideEffectContext, SideEffectError, SideEffectExecutionMode, SideEffectFailureCategory

from .errors import CapabilityExecutionError, CapabilityTimeoutError
from llm_proxy.observability.operational_logging import OperationalLogger

@dataclass(frozen=True, slots=True)
class CapabilityInvocation:
    policy_name: str
    binding: CapabilityBinding

@dataclass(frozen=True, slots=True)
class CapabilityRequestContext:
    request_id: str
    trace_transaction_id: str | None
    source_interface: str
    execution_mode: SideEffectExecutionMode
    requested_model: str
    canonical_model: str
    provider_instance: str
    upstream_model: str
    stream: bool

class CapabilityExecutor:
    def __init__(self, instances) -> None: self._instances = instances
    async def execute(self, context: CapabilityRequestContext, invocations: tuple[CapabilityInvocation, ...]) -> None:
        for invocation in invocations:
            binding = invocation.binding
            started = time.monotonic_ns()
            try:
                capability = self._instances.get(binding.instance, binding.family, binding.version, binding.capability)
                side_context = SideEffectContext(context.request_id, context.trace_transaction_id, binding.id, context.source_interface, context.execution_mode, context.requested_model, context.canonical_model, context.provider_instance, context.upstream_model, context.stream)
                await asyncio.wait_for(capability.invoke(side_context, binding.arguments), binding.timeout_seconds)
            except asyncio.CancelledError:
                OperationalLogger().log_capability(context, binding, "cancelled", (time.monotonic_ns()-started)//1_000_000); raise
            except asyncio.TimeoutError:
                OperationalLogger().log_capability(context, binding, "timeout", (time.monotonic_ns()-started)//1_000_000)
                if binding.on_failure is CapabilityFailureMode.REQUIRED: raise CapabilityTimeoutError()
            except SideEffectError as error:
                OperationalLogger().log_capability(context, binding, error.category.value, (time.monotonic_ns()-started)//1_000_000)
                if binding.on_failure is CapabilityFailureMode.REQUIRED: raise CapabilityExecutionError(error.category)
            except Exception:
                OperationalLogger().log_capability(context, binding, "failed", (time.monotonic_ns()-started)//1_000_000)
                if binding.on_failure is CapabilityFailureMode.REQUIRED: raise CapabilityExecutionError(SideEffectFailureCategory.FAILED)
            else:
                OperationalLogger().log_capability(context, binding, "success", (time.monotonic_ns()-started)//1_000_000)
