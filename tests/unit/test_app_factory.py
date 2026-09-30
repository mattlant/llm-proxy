from __future__ import annotations

from dataclasses import replace
import os
import subprocess
import sys
from pathlib import Path

import httpx
import pytest

from llm_proxy.extensions import CapabilityRegistration, ExtensionMetadata, ExtensionSdkCompatibility, SIDE_EFFECT_FAMILY, SIDE_EFFECT_VERSION

from llm_proxy.app import create_app
import llm_proxy.app as app_module
from llm_proxy.configuration.loader import ConfigurationLoader
from llm_proxy.configuration.store import ConfigurationStore


def _import_app_subprocess(environment: dict[str, str], cwd: Path) -> subprocess.CompletedProcess[str]:
    code = "import llm_proxy.app as app; print(app.CONFIG_PATH); print(app.configuration_store.snapshot.source_path)"
    return subprocess.run(
        [sys.executable, "-c", code],
        cwd=cwd,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )


def test_startup_default_uses_current_working_directory(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_bytes((Path(__file__).parents[1] / "fixtures/test_config.yaml").read_bytes())
    environment = os.environ.copy()
    environment.pop("LLM_PROXY_CONFIG", None)
    environment["PYTHONPATH"] = str(Path(__file__).parents[2])

    result = _import_app_subprocess(environment, tmp_path)

    assert result.stdout.splitlines() == [str(config_path), str(config_path)]


def test_explicit_configuration_path_overrides_current_working_directory(tmp_path: Path, config_file: Path) -> None:
    environment = os.environ.copy()
    environment["LLM_PROXY_CONFIG"] = str(config_file)
    environment["PYTHONPATH"] = str(Path(__file__).parents[2])
    cwd = tmp_path / "different-cwd"
    cwd.mkdir()

    result = _import_app_subprocess(environment, cwd)

    assert result.stdout.splitlines() == [str(config_file), str(config_file)]


class _Capability:
    def validate_arguments(self, arguments): pass
    async def invoke(self, context, arguments): pass


class _Instance:
    registrations = (CapabilityRegistration(SIDE_EFFECT_FAMILY, SIDE_EFFECT_VERSION, "run", _Capability()),)
    async def aclose(self): pass


class _Extension:
    metadata = ExtensionMetadata("test.lifecycle", "Test lifecycle", "test", "1")
    compatibility = ExtensionSdkCompatibility("1.0.0", "1.0.0")
    class factory:
        @staticmethod
        def create(config): return _Instance()


class _EntryPoint:
    name = "test"
    def load(self): return _Extension()


async def test_application_factory_uses_injected_dependencies_without_network(tmp_path) -> None:
    path = tmp_path / "config.yaml"
    path.write_text("""\
server:
  host: 127.0.0.1
  port: 11435
  timeout_seconds: 0
  config_reload_seconds: 1
  log_level: INFO
interfaces:
  openai:
    enabled: false
  ollama:
    enabled: false
  anthropic:
    enabled: true
providers:
  unused:
    extension: ollama
    enabled: true
    config:
      base_url: http://unused.test
      outbound_interface: openai
models:
  hidden-model:
    upstream_model: hidden-upstream
    provider: unused
    interfaces: [anthropic]
policies: []
""", encoding="utf-8")
    store = ConfigurationStore(path, ConfigurationLoader(), 1)
    app = create_app(store)

    assert app.state.configuration_store is store
    assert app.state.provider_gateways is None
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://proxy.test") as client:
        assert (await client.post("/v1/chat/completions", json={})).status_code == 404
        assert (await client.post("/api/chat", json={})).status_code == 404


async def test_lifespan_routes_read_the_live_store_after_bootstrap_replacement(tmp_path) -> None:
    path = tmp_path / "config.yaml"
    path.write_text("""\
server: {host: 127.0.0.1, port: 11435, timeout_seconds: 0, config_reload_seconds: 1, log_level: INFO}
interfaces: {openai: {enabled: true}}
providers: {unused: {extension: ollama, enabled: true, config: {base_url: http://unused.test, outbound_interface: openai}}}
models: {unused: {upstream_model: unused, provider: unused, interfaces: [openai]}}
policies: []
""", encoding="utf-8")
    bootstrap = ConfigurationStore(path, ConfigurationLoader(), 1)
    app = create_app(bootstrap)

    async with app.router.lifespan_context(app):
        live = app.state.configuration_store
        live._snapshot = replace(live.snapshot, source_path=tmp_path / "live.yaml")
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://proxy.test") as client:
            response = await client.get("/_proxy/health")

    assert live is not bootstrap
    assert response.json()["config_file"].endswith("live.yaml")


async def test_lifespan_activates_configured_extension_instances(tmp_path) -> None:
    path = tmp_path / "config.yaml"
    path.write_text("""\
server: {host: 127.0.0.1, port: 11435, timeout_seconds: 0, config_reload_seconds: 1, log_level: INFO}
interfaces: {openai: {enabled: true}}
providers: {test: {extension: ollama, enabled: true, config: {base_url: http://unused.test, outbound_interface: openai}}}
extensions: {side: {extension: test.lifecycle, enabled: true, config: {}}}
models: {test: {upstream_model: test, provider: test, interfaces: [openai]}}
policies: []
""", encoding="utf-8")
    app = create_app(ConfigurationStore(path, ConfigurationLoader(), 1), extension_entry_points=(_EntryPoint(),))
    async with app.router.lifespan_context(app):
        assert app.state.extension_instances.has("side", "side_effect", 1, "run")
        assert app.state.capability_executor is not None


async def test_lifespan_activates_extensions_before_providers_and_closes_in_reverse(tmp_path, monkeypatch) -> None:
    path = tmp_path / "config.yaml"
    path.write_text("""\
server: {host: 127.0.0.1, port: 11435, timeout_seconds: 0, config_reload_seconds: 1, log_level: INFO}
interfaces: {openai: {enabled: true}}
providers: {test: {extension: ollama, enabled: true, config: {base_url: http://unused.test, outbound_interface: openai}}}
extensions: {side: {extension: test.lifecycle, enabled: true, config: {}}}
models: {test: {upstream_model: test, provider: test, interfaces: [openai]}}
policies: []
""", encoding="utf-8")
    events = []

    class Instance(_Instance):
        async def aclose(self): events.append("extension-close")

    class Extension(_Extension):
        class factory:
            @staticmethod
            def create(config):
                events.append("extension-activate")
                return Instance()

    class EntryPoint:
        name = "test"
        def load(self): return Extension()

    class Gateways:
        async def aclose(self): events.append("provider-close")

    def activate(registry, providers, identity_policy=None):
        events.append("provider-activate")
        return Gateways()

    monkeypatch.setattr(app_module, "activate_provider_instances", activate)
    app = create_app(ConfigurationStore(path, ConfigurationLoader(), 1), extension_entry_points=(EntryPoint(),))
    assert events == []
    async with app.router.lifespan_context(app):
        assert events == ["extension-activate", "provider-activate"]
    assert events == ["extension-activate", "provider-activate", "provider-close", "extension-close"]


def test_application_factory_fails_startup_for_unusable_file_logging_directory(tmp_path) -> None:
    target = tmp_path / "not-a-directory"
    target.write_text("x", encoding="utf-8")
    path = tmp_path / "config.yaml"
    path.write_text(f"""\
server:
  host: 127.0.0.1
  port: 11435
  timeout_seconds: 0
  config_reload_seconds: 1
  log_level: INFO
  file_logging:
    enabled: true
    directory: {target}
interfaces:
  openai:
    enabled: true
providers:
  test:
    extension: ollama
    enabled: true
    config:
      base_url: http://unused.test
      outbound_interface: openai
models:
  test:
    upstream_model: test
    provider: test
    interfaces: [openai]
policies: []
""", encoding="utf-8")
    store = ConfigurationStore(path, ConfigurationLoader(), 1)
    with pytest.raises(RuntimeError, match="file logging"):
        create_app(store, provider_gateways=object())


@pytest.mark.parametrize("enabled", [True, False])
def test_main_logs_management_ui_url_only_when_enabled(tmp_path, monkeypatch, caplog, enabled) -> None:
    path = tmp_path / "config.yaml"
    path.write_text(f"""\
server:
  host: 127.0.0.1
  port: 12345
  timeout_seconds: 0
  config_reload_seconds: 1
  log_level: INFO
management:
  enabled: {str(enabled).lower()}
interfaces:
  openai:
    enabled: true
providers:
  test:
    extension: ollama
    enabled: true
    config:
      base_url: http://upstream.test
      outbound_interface: openai
models:
  test:
    upstream_model: test
    provider: test
    interfaces: [openai]
policies: []
""", encoding="utf-8")
    store = ConfigurationStore(path, ConfigurationLoader(), 1)
    monkeypatch.setattr(app_module, "configuration_store", store)
    monkeypatch.setattr(app_module, "CONFIG_PATH", path)
    monkeypatch.setattr("uvicorn.run", lambda *args, **kwargs: None)
    caplog.set_level("INFO", logger="llm-proxy")

    app_module.main()

    messages = [record.message for record in caplog.records if record.name == "llm-proxy"]
    expected = "Management UI available at http://127.0.0.1:12345/management/"
    assert (expected in messages) is enabled
