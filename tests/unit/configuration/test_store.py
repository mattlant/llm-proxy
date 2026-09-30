from __future__ import annotations

import os
from pathlib import Path

import pytest
import yaml

from llm_proxy.configuration.loader import ConfigurationError, ConfigurationLoader
from llm_proxy.configuration.store import ConfigurationStore

from tests.unit.configuration.test_validator import realistic_config


class FakeClock:
    def __init__(self):
        self.value = 100.0

    def __call__(self) -> float:
        return self.value


def write_config(path: Path, raw: dict) -> None:
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")


def advance_file(path: Path, raw: dict, mtime: float) -> None:
    write_config(path, raw)
    os.utime(path, (mtime, mtime))


def test_valid_initial_load_creates_one_consistent_snapshot(tmp_path: Path, caplog):
    path = tmp_path / "config.yaml"
    write_config(path, realistic_config())
    clock = FakeClock()

    store = ConfigurationStore(path, ConfigurationLoader(), 1.0, clock)
    snapshot = store.snapshot

    assert snapshot.source_path == path
    assert snapshot.loaded_at == 100.0
    assert snapshot.registry.resolve("anthropic", "local-opus").canonical_name == "qwable"  # type: ignore[arg-type]
    assert snapshot.resolver._config is snapshot.config  # type: ignore[attr-defined]
    assert snapshot.policy_engine._config is snapshot.config  # type: ignore[attr-defined]
    assert not any("Configuration reloaded successfully" in record.getMessage() for record in caplog.records)


@pytest.mark.parametrize("content", [None, "server: ["])
def test_missing_or_invalid_initial_configuration_fails_fast(tmp_path: Path, content: str | None):
    path = tmp_path / "config.yaml"
    if content is not None:
        path.write_text(content, encoding="utf-8")

    with pytest.raises((ConfigurationError, OSError)):
        ConfigurationStore(path, ConfigurationLoader(), 1.0, FakeClock())


def test_valid_reload_replaces_entire_snapshot(tmp_path: Path, caplog):
    path = tmp_path / "config.yaml"
    raw = realistic_config()
    write_config(path, raw)
    clock = FakeClock()
    store = ConfigurationStore(path, ConfigurationLoader(), 1.0, clock)
    previous = store.snapshot
    caplog.clear()

    raw["models"]["qwable"]["parameters"]["temperature"] = 0.8
    advance_file(path, raw, 200.0)
    clock.value = 101.0

    assert store.reload() is True
    current = store.snapshot
    assert current is not previous
    assert current.config is not previous.config
    assert current.config.models["qwable"].parameters.temperature == 0.8
    assert current.registry is not previous.registry
    assert current.resolver._config is current.config  # type: ignore[attr-defined]
    assert [record.getMessage() for record in caplog.records if record.levelname == "INFO"] == [
        f"Configuration reloaded successfully: path={path}"
    ]


def test_bootstrap_source_mtime_is_preserved_and_intervening_write_is_detected(tmp_path: Path):
    path = tmp_path / "config.yaml"
    raw = realistic_config()
    write_config(path, raw)
    bootstrap = ConfigurationStore(path, ConfigurationLoader(), 1.0, FakeClock())
    source_mtime = bootstrap.snapshot.source_mtime
    raw["models"]["qwable"]["upstream_model"] = "changed-after-bootstrap"
    advance_file(path, raw, 200.0)

    live = ConfigurationStore.from_initial_config(
        path, ConfigurationLoader(), 1.0, bootstrap.snapshot.config, source_mtime,
    )

    assert live.snapshot.source_mtime == source_mtime
    assert live.reload(force=True)
    assert live.snapshot.config.models["qwable"].upstream_model == "changed-after-bootstrap"


def test_invalid_reload_retains_previous_snapshot_identity(tmp_path: Path, caplog):
    path = tmp_path / "config.yaml"
    write_config(path, realistic_config())
    clock = FakeClock()
    store = ConfigurationStore(path, ConfigurationLoader(), 1.0, clock)
    previous = store.snapshot
    caplog.clear()

    path.write_text("models: [", encoding="utf-8")
    os.utime(path, (200.0, 200.0))
    clock.value = 101.0

    assert store.reload() is False
    assert store.snapshot is previous
    assert not any("Configuration reloaded successfully" in record.getMessage() for record in caplog.records)


def test_invalid_matcher_reload_retains_previous_snapshot(tmp_path: Path):
    path = tmp_path / "config.yaml"
    raw = realistic_config()
    raw["policies"] = [{"name": "valid", "enabled": True, "match": {"model_contains_any": ["qw"]}, "actions": {"parameters": {"temperature": 0.2}}}]
    write_config(path, raw)
    store = ConfigurationStore(path, ConfigurationLoader(), 1.0, FakeClock())
    previous = store.snapshot
    raw["policies"][0]["match"]["model_contains_any"] = [1]
    advance_file(path, raw, 200.0)

    assert store.reload(force=True) is False
    assert store.snapshot is previous
    assert store.last_rejected_reload == "ConfigurationError"


def test_provider_affecting_mtime_reload_retains_previous_snapshot(tmp_path: Path):
    path = tmp_path / "config.yaml"
    raw = realistic_config()
    write_config(path, raw)
    store = ConfigurationStore(path, ConfigurationLoader(), 1, FakeClock())
    previous = store.snapshot
    raw["providers"]["rtx-3090"]["config"]["base_url"] = "http://other.test"
    advance_file(path, raw, 200.0)

    assert store.reload(force=True) is False
    assert store.snapshot is previous
    assert store.last_rejected_reload == "provider, extension, or management configuration changed; restart is required"


@pytest.mark.parametrize(("field", "value"), [("extension", "other.extension"), ("enabled", False), ("config", {"nested": {"changed": True}})])
def test_extension_activation_changes_require_restart(tmp_path: Path, field: str, value):
    path = tmp_path / "config.yaml"
    raw = realistic_config()
    raw["extensions"] = {"side": {"extension": "test.extension", "enabled": True, "config": {"nested": {"value": 1}}}}
    write_config(path, raw)
    store = ConfigurationStore(path, ConfigurationLoader(), 1, FakeClock())
    previous = store.snapshot
    raw["extensions"]["side"][field] = value
    advance_file(path, raw, 200.0)

    assert store.reload(force=True) is False
    assert store.snapshot is previous
    assert store.last_rejected_reload == "provider, extension, or management configuration changed; restart is required"


def test_unchanged_mtime_skips_reload_and_interval_is_respected(tmp_path: Path):
    path = tmp_path / "config.yaml"
    write_config(path, realistic_config())
    clock = FakeClock()
    store = ConfigurationStore(path, ConfigurationLoader(), 10.0, clock)
    previous = store.snapshot

    clock.value = 105.0
    assert store.reload() is False
    assert store.snapshot is previous

    clock.value = 110.0
    assert store.reload() is False
    assert store.snapshot is previous
