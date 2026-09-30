from __future__ import annotations

from importlib.metadata import EntryPoint

import pytest

from llm_proxy.application.provider_registry import activate_provider_instances, discover_extensions
from llm_proxy.configuration.models import ProviderConfig
from llm_proxy.management.command_registry import ProviderCommandRegistry
from llm_proxy.management.command_service import CommandDispatchService, CommandQueryService


async def test_external_provider_commands_are_instance_local_and_public_only(monkeypatch):
    monkeypatch.syspath_prepend(str(__import__("pathlib").Path(__file__).parents[1] / "fixtures"))
    point = EntryPoint("deterministic", "external_provider.deterministic_provider:DeterministicExtension", "llm_proxy.providers")
    extensions = discover_extensions((), entry_points=lambda: (point,))
    providers = {
        "first": ProviderConfig("first", "example.deterministic", True, {"initial_state": 1}),
        "second": ProviderConfig("second", "example.deterministic", True, {"initial_state": 2}),
    }
    gateways = activate_provider_instances(extensions, providers)
    commands = ProviderCommandRegistry(gateways, providers, {name: "example.deterministic" for name in providers}, {name: True for name in providers})
    service = CommandDispatchService(commands, 1)

    assert [item["name"] for item in CommandQueryService(commands).list("first")["commands"]] == ["get_state", "reset_state", "set_state"]
    assert (await service.dispatch("first", "set_state", {"value": 7}))["result"] == {"value": 7}
    assert (await service.dispatch("second", "get_state", {}))["result"] == {"value": 2}
    with pytest.raises(Exception):
        await service.dispatch("first", "set_state", {})
    assert (await service.dispatch("first", "get_state", {}))["result"] == {"value": 7}
