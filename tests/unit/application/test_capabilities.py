from __future__ import annotations

import pytest

from llm_proxy.application.capabilities import CapabilityExecutor, CapabilityInvocation, CapabilityRequestContext
from llm_proxy.application.errors import CapabilityExecutionError, CapabilityTimeoutError
from llm_proxy.configuration.models import CapabilityBinding, CapabilityFailureMode
from llm_proxy.extensions.side_effect import SideEffectError, SideEffectExecutionMode, SideEffectFailureCategory

class Registry:
    def __init__(self, capability): self.capability = capability
    def get(self, *args): return self.capability

class Capability:
    def __init__(self, mode="ok"): self.mode, self.calls = mode, []
    async def invoke(self, context, arguments):
        self.calls.append(context.binding_id)
        if self.mode == "failed": raise SideEffectError(SideEffectFailureCategory.FAILED)

def context(): return CapabilityRequestContext("r", None, "openai", SideEffectExecutionMode.CANONICAL, "m", "m", "p", "u", False)
def invocation(identifier="one", mode=CapabilityFailureMode.REQUIRED): return CapabilityInvocation("p", CapabilityBinding(identifier, "instance", "side_effect", 1, "run", on_failure=mode))

async def test_executor_runs_records_in_order() -> None:
    cap = Capability(); await CapabilityExecutor(Registry(cap)).execute(context(), (invocation("one"), invocation("two")))
    assert cap.calls == ["one", "two"]

async def test_required_failure_stops_execution() -> None:
    with pytest.raises(CapabilityExecutionError): await CapabilityExecutor(Registry(Capability("failed"))).execute(context(), (invocation(),))
