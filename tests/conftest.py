from __future__ import annotations

import os
from pathlib import Path

import httpx
import pytest
import respx
import yaml

TEST_CONFIG_PATH = (Path(__file__).parent / "fixtures" / "test_config.yaml").resolve()
os.environ["LLM_PROXY_CONFIG"] = str(TEST_CONFIG_PATH)

from llm_proxy import app as app_module


@pytest.fixture
def config_file(tmp_path: Path) -> Path:
    path = tmp_path / "config.yaml"
    path.write_text(TEST_CONFIG_PATH.read_text(encoding="utf-8"), encoding="utf-8")
    return path


@pytest.fixture
def isolated_proxy(monkeypatch: pytest.MonkeyPatch, config_file: Path):
    store = app_module.ConfigurationStore(config_file, app_module.ConfigurationLoader(), 1.0)
    monkeypatch.setattr(app_module, "CONFIG_PATH", config_file)
    monkeypatch.setattr(app_module, "configuration_store", store)
    isolated_proxy = app_module.app
    isolated_proxy.state.configuration_store = store
    isolated_proxy.state.model_listing_service = app_module.ModelListingService(store, isolated_proxy.state.provider_gateways)
    return isolated_proxy


@pytest.fixture
def configure_rules(monkeypatch: pytest.MonkeyPatch, config_file: Path):
    def configure(config: dict) -> None:
        config_file.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
        store = app_module.ConfigurationStore(config_file, app_module.ConfigurationLoader(), 1.0)
        monkeypatch.setattr(
            app_module,
            "configuration_store",
            store,
        )
        app_module.app.state.configuration_store = store
        app_module.app.state.model_listing_service = app_module.ModelListingService(store, app_module.app.state.provider_gateways)

    return configure


@pytest.fixture
async def client(isolated_proxy):
    transport = httpx.ASGITransport(app=isolated_proxy)
    async with httpx.AsyncClient(transport=transport, base_url="http://proxy.test") as value:
        yield value


@pytest.fixture
def fake_upstream():
    with respx.mock(assert_all_called=False) as router:
        yield router
