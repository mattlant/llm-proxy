from __future__ import annotations

import httpx
import pytest

from llm_proxy.app import create_app
from llm_proxy.application.provider_registry import ProviderGatewayRegistry
from llm_proxy.configuration.loader import ConfigurationLoader
from llm_proxy.configuration.store import ConfigurationStore
from llm_proxy.management.command_registry import ProviderCommandRegistry
from llm_proxy.management.command_service import CommandDispatchService, CommandQueryService
from llm_proxy.provider_extensions import ManagementCommandDescriptor, ManagementCommandMutability, ManagementCommandPermission
from tests.integration.test_admin_interface import _config


class CommandGateway:
    management_commands = (ManagementCommandDescriptor("get_state", "Return state", {"type": "object", "additionalProperties": False}, {"type": "object", "required": ["value"], "properties": {"value": {"type": "integer"}}}, ManagementCommandMutability.READ_ONLY, ManagementCommandPermission.INSPECT),)
    async def execute_management_command(self, command_name, payload, context):
        return {"value": 3}

    async def complete(self, execution):
        raise AssertionError("not used")


class Factory:
    def __init__(self): self.gateway = CommandGateway()
    def get(self, instance_name): return self.gateway


@pytest.mark.asyncio
async def test_admin_discovers_and_invokes_commands_through_generic_routes(tmp_path, monkeypatch, caplog):
    monkeypatch.setenv("LLM_PROXY_ADMIN_TOKEN", "synthetic-token")
    path = tmp_path / "config.yaml"
    path.write_text(_config(True), encoding="utf-8")
    store = ConfigurationStore(path, ConfigurationLoader(), 1)
    factory = Factory()
    app = create_app(store, factory)
    providers = store.snapshot.config.providers
    registry = ProviderCommandRegistry(ProviderGatewayRegistry({"test": factory.gateway}, providers), providers, {"test": "fixture"}, {"test": True})
    app.state.command_query_service = CommandQueryService(registry)
    app.state.command_dispatch_service = CommandDispatchService(registry, 1)
    headers = {"Authorization": "Bearer synthetic-token"}
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://proxy.test") as client:
        discovery = await client.get("/_admin/v1/providers/test/commands", headers=headers)
        invoked = await client.post("/_admin/v1/providers/test/commands/get_state", headers=headers, json={})
        invalid = await client.post("/_admin/v1/providers/test/commands/get_state", headers=headers, json={"secret": "not logged"})

    assert discovery.status_code == 200
    assert discovery.json()["commands"][0]["name"] == "get_state"
    assert invoked.status_code == 200
    assert invoked.json()["result"] == {"value": 3}
    assert invalid.status_code == 422
    audit = [record.message for record in caplog.records if record.message.startswith("admin_command")]
    assert any("outcome=success" in record for record in audit)
    assert any("outcome=invalid_command_input" in record for record in audit)
    assert all("not logged" not in record for record in audit)
