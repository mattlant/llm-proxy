from __future__ import annotations

import asyncio

import pytest

from llm_proxy.application.provider_registry import ProviderGatewayRegistry
from llm_proxy.configuration.models import ProviderConfig
from llm_proxy.management.command_registry import ProviderCommandRegistry
from llm_proxy.management.command_service import CommandDispatchService, CommandQueryService
from llm_proxy.management.errors import CommandTimeout, InvalidCommandInput, InvalidCommandOutput
from llm_proxy.provider_extensions import ManagementCommandDescriptor, ManagementCommandMutability, ManagementCommandPermission


class Gateway:
    management_commands = (
        ManagementCommandDescriptor("get_state", "Return state", {"type": "object", "additionalProperties": False}, {"type": "object", "required": ["value"], "properties": {"value": {"type": "integer"}}}, ManagementCommandMutability.READ_ONLY, ManagementCommandPermission.INSPECT),
        ManagementCommandDescriptor("set_state", "Set state", {"type": "object", "required": ["value"], "properties": {"value": {"type": "integer"}}, "additionalProperties": False}, {"type": "object", "required": ["value"], "properties": {"value": {"type": "integer"}}}, ManagementCommandMutability.STATE_CHANGING, ManagementCommandPermission.OPERATE),
    )
    def __init__(self): self.value = 0
    async def execute_management_command(self, command_name, payload, context):
        if command_name == "get_state": return {"value": self.value}
        self.value = payload["value"]
        return {"value": self.value}


def registry(gateway=None):
    gateway = gateway or Gateway()
    providers = {"primary": ProviderConfig("primary", "fixture", True, {})}
    return ProviderCommandRegistry(ProviderGatewayRegistry({"primary": gateway}, providers), providers, {"primary": "fixture"}, {"primary": True}), gateway


async def test_dispatch_validates_and_is_instance_local():
    commands, gateway = registry()
    result = await CommandDispatchService(commands, 1).dispatch("primary", "set_state", {"value": 7})
    assert result["result"] == {"value": 7}
    with pytest.raises(InvalidCommandInput):
        await CommandDispatchService(commands, 1).dispatch("primary", "set_state", {})
    assert gateway.value == 7
    assert [item["name"] for item in CommandQueryService(commands).list("primary")["commands"]] == ["get_state", "set_state"]


async def test_timeout_failure_and_invalid_output_are_bounded():
    class BadGateway(Gateway):
        async def execute_management_command(self, command_name, payload, context):
            if command_name == "get_state":
                await asyncio.sleep(1)
            return {"not_value": 1}
    commands, _ = registry(BadGateway())
    with pytest.raises(CommandTimeout):
        await CommandDispatchService(commands, 0.001).dispatch("primary", "get_state", {})
    with pytest.raises(InvalidCommandOutput):
        await CommandDispatchService(commands, 1).dispatch("primary", "set_state", {"value": 1})


async def test_caller_cancellation_propagates_and_is_audited(caplog):
    started = asyncio.Event()
    cancelled = asyncio.Event()

    class WaitingGateway(Gateway):
        async def execute_management_command(self, command_name, payload, context):
            started.set()
            try:
                await asyncio.Future()
            except asyncio.CancelledError:
                cancelled.set()
                raise

    commands, _ = registry(WaitingGateway())
    task = asyncio.create_task(CommandDispatchService(commands, 1).dispatch("primary", "get_state", {}))
    await started.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert cancelled.is_set()
    assert any("outcome=command_cancelled" in record.message and "cancelled=true" in record.message for record in caplog.records)
