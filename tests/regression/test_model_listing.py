from __future__ import annotations

import asyncio

import httpx
import pytest
import yaml

from llm_proxy.app import create_app
from llm_proxy.application.errors import ModelNotFoundError
from llm_proxy.application.provider_registry import ProviderGatewayRegistry
from llm_proxy.application.runtime import RuntimeDependencies
from llm_proxy.configuration.loader import ConfigurationLoader
from llm_proxy.configuration.models import InterfaceName
from llm_proxy.configuration.store import ConfigurationStore
from llm_proxy.provider_extensions import ProviderCapabilities, ProviderListedModel


def listing_config() -> dict:
    return {
        "server": {"host": "127.0.0.1", "port": 11435, "timeout_seconds": 0, "config_reload_seconds": 1, "log_level": "INFO"},
        "interfaces": {"openai": {"enabled": True}, "ollama": {"enabled": True}},
        "providers": {
            "local": {"extension": "ollama", "config": {"base_url": "http://local.test", "outbound_interface": "openai"}, "enabled": True},
            "gpu": {"extension": "ollama", "config": {"base_url": "http://gpu.test", "outbound_interface": "openai"}, "enabled": True},
        },
        "models": {
            "local-model": {"upstream_model": "local-upstream", "provider": "local", "interfaces": ["openai", "ollama"]},
            "gpu-model": {"upstream_model": "gpu-upstream", "provider": "gpu", "interfaces": ["openai"]},
        },
        "policies": [],
    }


async def test_api_tags_lists_ollama_exposed_profiles(client, configure_rules):
    configure_rules(listing_config())

    response = await client.get("/api/tags")

    assert response.status_code == 200
    assert response.json() == {
        "models": [{"name": "local-model", "model": "local-upstream", "provider": "local"}]
    }


async def test_v1_models_lists_openai_exposed_profiles(client, configure_rules):
    configure_rules(listing_config())

    response = await client.get("/v1/models")

    assert response.status_code == 200
    assert response.json() == {
        "object": "list",
        "data": [
            {"id": "local-model", "object": "model", "owned_by": "local"},
            {"id": "gpu-model", "object": "model", "owned_by": "gpu"},
        ],
    }


class _ListingGateway:
    def __init__(self, models=(), *, wait: asyncio.Event | None = None, started: asyncio.Event | None = None):
        self.models, self.wait, self.started = tuple(models), wait, started
        self.cancelled = False

    async def list_models(self):
        if self.started:
            self.started.set()
        try:
            if self.wait:
                await self.wait.wait()
        except asyncio.CancelledError:
            self.cancelled = True
            raise
        return self.models


async def _configured_listing_client(tmp_path, raw, gateways):
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
    store = ConfigurationStore(path, ConfigurationLoader(), 60)
    registry = ProviderGatewayRegistry(
        gateways,
        store.snapshot.config.providers,
        {name: ProviderCapabilities(completion=True, model_listing=True) for name in gateways},
    )
    store = ConfigurationStore(path, ConfigurationLoader(), 60, runtime_dependencies=RuntimeDependencies(provider_gateways=registry))
    app = create_app(store, registry)
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://proxy.test"), app


def _passthrough_config():
    raw = listing_config()
    raw["model_access"] = {"mode": "provider_passthrough", "listing_timeout_seconds": 1}
    raw["providers"]["local"]["listing_enabled"] = True
    raw["providers"]["gpu"]["listing_enabled"] = True
    raw["models"]["profile-same-as-raw"] = {
        "upstream_model": "same",
        "provider": "gpu",
        "interfaces": ["openai", "ollama"],
    }
    return raw


@pytest.mark.asyncio
async def test_active_app_lists_profiles_and_two_provider_qualified_records(tmp_path):
    raw = _passthrough_config()
    raw["providers"]["cpu"] = {
        "extension": "ollama", "config": {"base_url": "http://cpu.test", "outbound_interface": "openai"},
        "enabled": True, "listing_enabled": True,
    }
    client, _ = await _configured_listing_client(tmp_path, raw, {
        "local": _ListingGateway((ProviderListedModel("z"),)),
        "gpu": _ListingGateway((ProviderListedModel("same"),)),
        "cpu": _ListingGateway((ProviderListedModel("same"),)),
    })
    async with client:
        tags = (await client.get("/api/tags")).json()["models"]
        models = (await client.get("/v1/models")).json()["data"]

    assert {item["name"] for item in tags} >= {"profile-same-as-raw", "gpu::same", "cpu::same"}
    assert [item["id"] for item in models if "::" in item["id"]] == ["cpu::same", "gpu::same", "local::z"]


@pytest.mark.asyncio
async def test_active_listing_is_concurrent_and_cleans_up_cancelled_calls(tmp_path):
    raw = _passthrough_config()
    release, first_started, second_started = asyncio.Event(), asyncio.Event(), asyncio.Event()
    first = _ListingGateway((ProviderListedModel("one"),), wait=release, started=first_started)
    second = _ListingGateway((ProviderListedModel("two"),), wait=release, started=second_started)
    client, app = await _configured_listing_client(tmp_path, raw, {"local": first, "gpu": second})
    async with client:
        request = asyncio.create_task(client.get("/v1/models"))
        await asyncio.wait_for(asyncio.gather(first_started.wait(), second_started.wait()), 1)
        release.set()
        response = await request
        assert response.status_code == 200

    cancelled_release, cancelled_started = asyncio.Event(), asyncio.Event()
    blocked = _ListingGateway(wait=cancelled_release, started=cancelled_started)
    app.state.model_listing_service._gateways = ProviderGatewayRegistry(
        {"local": blocked, "gpu": _ListingGateway()}, app.state.configuration_store.snapshot.config.providers,
        {"local": ProviderCapabilities(model_listing=True), "gpu": ProviderCapabilities(model_listing=True)},
    )
    listing = asyncio.create_task(app.state.model_listing_service.list_models(InterfaceName.OPENAI))
    await asyncio.wait_for(cancelled_started.wait(), 1)
    listing.cancel()
    with pytest.raises(asyncio.CancelledError):
        await listing
    assert blocked.cancelled


@pytest.mark.asyncio
async def test_active_native_listing_isolates_invalid_and_unavailable_providers(tmp_path):
    raw = _passthrough_config()

    class Unavailable(_ListingGateway):
        async def list_models(self):
            raise RuntimeError("upstream unavailable")

    class Invalid(_ListingGateway):
        async def list_models(self):
            return ("not-normalized",)

    client, _ = await _configured_listing_client(tmp_path, raw, {
        "local": _ListingGateway((ProviderListedModel("healthy"),)),
        "gpu": Invalid(),
    })
    async with client:
        assert (await client.get("/v1/models")).status_code == 200
        assert "local::healthy" in {item["id"] for item in (await client.get("/v1/models")).json()["data"]}

    client, _ = await _configured_listing_client(tmp_path, raw, {"local": Unavailable(), "gpu": Invalid()})
    async with client:
        assert {item["name"] for item in (await client.get("/api/tags")).json()["models"]} == {"local-model", "profile-same-as-raw"}


def test_configured_only_rejects_qualified_identity_while_profile_wins(tmp_path):
    raw = _passthrough_config()
    raw["model_access"]["mode"] = "configured_only"
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
    store = ConfigurationStore(path, ConfigurationLoader(), 60)
    assert store.snapshot.resolver.resolve(InterfaceName.OPENAI, "gpu-model").upstream_model == "gpu-upstream"
    with pytest.raises(ModelNotFoundError):
        store.snapshot.resolver.resolve(InterfaceName.OPENAI, "gpu::same")


def test_opted_out_and_non_listing_capable_providers_remain_direct_routes(tmp_path):
    raw = _passthrough_config()
    raw["providers"]["gpu"]["listing_enabled"] = False
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
    store = ConfigurationStore(path, ConfigurationLoader(), 60)
    gateways = ProviderGatewayRegistry(
        {"local": object(), "gpu": object()}, store.snapshot.config.providers,
        {"local": ProviderCapabilities(completion=True, model_listing=True), "gpu": ProviderCapabilities(completion=True)},
    )
    store = ConfigurationStore(path, ConfigurationLoader(), 60, runtime_dependencies=RuntimeDependencies(provider_gateways=gateways))
    execution = store.snapshot.resolver.resolve(InterfaceName.OPENAI, "GPU::opaque-model")
    assert execution.canonical_model == "gpu::opaque-model"
