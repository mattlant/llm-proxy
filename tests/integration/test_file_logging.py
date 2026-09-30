from __future__ import annotations

import logging

import pytest

from llm_proxy.app import create_app
from llm_proxy.configuration.loader import ConfigurationLoader
from llm_proxy.configuration.store import ConfigurationStore
from llm_proxy.observability import operational_logging


def _write_config(path, directory, *, enabled: bool, log_level: str = "DEBUG", transport_debug: bool = True) -> None:
    path.write_text(
        f"""\
server:
  host: 127.0.0.1
  port: 11435
  timeout_seconds: 0
  config_reload_seconds: 1
  log_level: {log_level}
  transport_debug: {str(transport_debug).lower()}
  file_logging:
    enabled: {str(enabled).lower()}
    directory: {directory}
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
""",
        encoding="utf-8",
    )


@pytest.fixture(autouse=True)
def cleanup_file_handler():
    yield
    operational_logging._remove_active_file_handler()


def test_create_app_routes_operational_and_transport_logs_to_launch_file(tmp_path, caplog):
    config_path = tmp_path / "config.yaml"
    log_dir = tmp_path / "diagnostics"
    _write_config(config_path, log_dir, enabled=True)
    store = ConfigurationStore(config_path, ConfigurationLoader(), 1)
    root = logging.getLogger()
    console_handlers = list(root.handlers)

    create_app(store, provider_gateways=object())
    first_file = next(log_dir.glob("*.log"))
    caplog.set_level(logging.DEBUG)
    logging.getLogger("llm-proxy").info("integration-operational")
    logging.getLogger("llm-proxy.payload").debug("integration-payload")
    logging.getLogger("httpx").debug("integration-httpx")
    logging.getLogger("httpcore").debug("integration-httpcore")
    operational_logging._active_file_handler.flush()

    content = first_file.read_text(encoding="utf-8")
    assert "integration-operational" in content
    assert "integration-payload" in content
    assert "integration-httpx" in content
    assert "integration-httpcore" in content
    assert all(handler in root.handlers for handler in console_handlers)
    assert any(record.message == "integration-operational" for record in caplog.records)

    create_app(store, provider_gateways=object())
    files = sorted(log_dir.glob("*.log"))
    assert len(files) == 2
    assert files[0] != files[1]
    assert "integration-operational" in first_file.read_text(encoding="utf-8")


def test_create_app_disabled_file_logging_keeps_console_only(tmp_path):
    config_path = tmp_path / "config.yaml"
    log_dir = tmp_path / "diagnostics"
    _write_config(config_path, log_dir, enabled=False)
    store = ConfigurationStore(config_path, ConfigurationLoader(), 1)

    create_app(store, provider_gateways=object())

    assert not log_dir.exists()
    assert operational_logging._active_file_handler is None


def test_create_app_file_destination_is_startup_only_across_reload(tmp_path):
    config_path = tmp_path / "config.yaml"
    first_dir = tmp_path / "first"
    second_dir = tmp_path / "second"
    _write_config(config_path, first_dir, enabled=True)
    store = ConfigurationStore(config_path, ConfigurationLoader(), 1)

    create_app(store, provider_gateways=object())
    first_files = list(first_dir.glob("*.log"))
    _write_config(config_path, second_dir, enabled=False)
    store.reload(force=True)

    assert len(first_files) == 1
    assert not second_dir.exists()
    assert operational_logging._active_file_handler is not None


def test_create_app_preserves_server_and_transport_level_filtering(tmp_path):
    config_path = tmp_path / "config.yaml"
    log_dir = tmp_path / "diagnostics"
    _write_config(config_path, log_dir, enabled=True, log_level="INFO", transport_debug=False)
    store = ConfigurationStore(config_path, ConfigurationLoader(), 1)

    create_app(store, provider_gateways=object())
    logging.getLogger("llm-proxy").debug("filtered-operational")
    logging.getLogger("httpx").debug("filtered-httpx")
    logging.getLogger("httpcore").debug("filtered-httpcore")
    logging.getLogger("llm-proxy").info("included-operational")
    operational_logging._active_file_handler.flush()

    content = next(log_dir.glob("*.log")).read_text(encoding="utf-8")
    assert "filtered-operational" not in content
    assert "filtered-httpx" not in content
    assert "filtered-httpcore" not in content
    assert "included-operational" in content
