from pathlib import Path

import yaml
import pytest
from concurrent.futures import ThreadPoolExecutor

from llm_proxy.configuration.loader import ConfigurationLoader
from llm_proxy.configuration.store import ConfigurationStore
from llm_proxy.management.configuration_service import ConfigurationManagementService
from llm_proxy.configuration.atomic_writer import AtomicConfigurationWriter
from llm_proxy.management.errors import ActivationFailed, ConfigurationStateConflict, ConfigurationStateIndeterminate, PersistenceFailed, RevisionConflict
from tests.unit.configuration.test_validator import realistic_config


def test_validation_is_side_effect_free_and_classifies_provider_changes(tmp_path: Path):
    path = tmp_path / "config.yaml"
    raw = realistic_config()
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    store = ConfigurationStore(path, ConfigurationLoader(), 1)
    service = ConfigurationManagementService(store)
    before = path.read_bytes()
    raw["providers"]["rtx-3090"]["config"]["base_url"] = "http://other.test"
    result = service.validate(raw)
    assert result.valid and result.restart_required
    assert path.read_bytes() == before
    assert store.snapshot.config.providers["rtx-3090"].config["base_url"] != "http://other.test"


def test_listing_enabled_change_is_reload_safe(tmp_path: Path):
    path = tmp_path / "config.yaml"
    raw = realistic_config()
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    service = ConfigurationManagementService(ConfigurationStore(path, ConfigurationLoader(), 1))
    raw["providers"]["rtx-3090"]["listing_enabled"] = True
    result = service.validate(raw)
    assert result.valid and not result.restart_required


def test_listing_enabled_update_activates_snapshot_without_restart(tmp_path: Path):
    path = tmp_path / "config.yaml"; raw = realistic_config(); path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    store = ConfigurationStore(path, ConfigurationLoader(), 1); service = ConfigurationManagementService(store)
    raw["providers"]["rtx-3090"]["listing_enabled"] = True
    result = service.update(raw, service.persisted_revision)
    assert not result["restart_required"]
    assert store.snapshot.config.providers["rtx-3090"].listing_enabled


def test_safe_update_activates_and_conflict_writes_nothing(tmp_path: Path):
    path = tmp_path / "config.yaml"
    raw = realistic_config()
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    store = ConfigurationStore(path, ConfigurationLoader(), 1)
    service = ConfigurationManagementService(store)
    expected = service.persisted_revision
    raw["models"]["qwable"]["parameters"]["temperature"] = 0.8
    result = service.update(raw, expected)
    assert not result["restart_required"]
    assert result["active_revision"] == result["persisted_revision"]
    before = path.read_bytes()
    with pytest.raises(RevisionConflict):
        service.update(raw, expected)
    assert path.read_bytes() == before


def test_failed_write_and_activation_retain_active_snapshot(tmp_path: Path):
    path = tmp_path / "config.yaml"
    raw = realistic_config()
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    store = ConfigurationStore(path, ConfigurationLoader(), 1)
    previous = store.snapshot
    expected = ConfigurationManagementService(store).persisted_revision
    raw["models"]["qwable"]["parameters"]["temperature"] = 0.8
    before = path.read_bytes()

    class FailingWriter:
        def write(self, path, content):
            raise OSError("write failed")

    with pytest.raises(PersistenceFailed):
        ConfigurationManagementService(store, FailingWriter()).update(raw, expected)
    assert path.read_bytes() == before
    assert store.snapshot is previous

    store.activate_candidate = lambda candidate: (_ for _ in ()).throw(RuntimeError("activation failed"))  # type: ignore[method-assign]
    with pytest.raises(ActivationFailed):
        ConfigurationManagementService(store).update(raw, expected)
    assert store.snapshot is previous


def test_concurrent_updates_with_one_revision_allow_one_writer(tmp_path: Path):
    path = tmp_path / "config.yaml"
    raw = realistic_config()
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    store = ConfigurationStore(path, ConfigurationLoader(), 1)
    service = ConfigurationManagementService(store)
    expected = service.persisted_revision

    def update():
        candidate = realistic_config()
        candidate["models"]["qwable"]["parameters"]["temperature"] = 0.8
        try:
            service.update(candidate, expected)
            return "success"
        except RevisionConflict:
            return "conflict"

    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(lambda _: update(), range(2)))
    assert sorted(results) == ["conflict", "success"]


def _policy_service(tmp_path: Path):
    path = tmp_path / "config.yaml"; raw = realistic_config(); raw["policies"] = [{"name": "p", "enabled": True, "match": {"models": ["qwable"]}, "actions": {"parameters": {"temperature": 0.2}, "model": "qwable", "provider": "rtx-3090"}}]
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    store = ConfigurationStore(path, ConfigurationLoader(), 1)
    return path, store, ConfigurationManagementService(store)


def test_policy_validation_apply_and_stale_revision_leave_safe_state(tmp_path: Path):
    path, store, service = _policy_service(tmp_path)
    snapshot = service.policy_snapshot(); before = path.read_bytes(); invalid = service.validate_policies([{"name": "bad"}])
    assert not invalid.valid and path.read_bytes() == before
    policies = snapshot["policies"]; policies[0]["actions"]["parameters"]["temperature"] = 0.8
    result = service.apply_policies(policies, snapshot["revision"])
    assert set(result) == {"active_revision", "persisted_revision", "restart_required", "policies"}
    assert result["active_revision"] == result["persisted_revision"] == service.active_revision
    assert result["policies"][0]["actions"]["model"] == "qwable"
    assert result["policies"][0]["actions"]["provider"] == "rtx-3090"
    with pytest.raises(RevisionConflict): service.apply_policies(policies, snapshot["revision"])


def test_policy_snapshot_detects_active_persisted_mismatch(tmp_path: Path):
    path, _, service = _policy_service(tmp_path)
    raw = realistic_config(); raw["policies"] = []
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    with pytest.raises(ConfigurationStateConflict): service.policy_snapshot()


def test_policy_activation_failure_restores_confirmed_prior_state(tmp_path: Path):
    path, store, service = _policy_service(tmp_path)
    snapshot = service.policy_snapshot(); previous = store.snapshot; original_activate = store.activate_candidate
    calls = 0
    def fail_once(candidate):
        nonlocal calls
        calls += 1
        if calls == 1: raise RuntimeError("activation failed")
        original_activate(candidate)
    store.activate_candidate = fail_once  # type: ignore[method-assign]
    with pytest.raises(ActivationFailed): service.apply_policies(snapshot["policies"], snapshot["revision"])
    assert calls == 2 and store.snapshot is previous
    assert service.policy_snapshot()["revision"] == snapshot["revision"]


def test_policy_rollback_write_failure_reports_indeterminate_state(tmp_path: Path):
    path, store, service = _policy_service(tmp_path)
    snapshot = service.policy_snapshot(); previous = store.snapshot
    class FailRollbackWriter:
        def __init__(self): self.calls = 0; self.writer = AtomicConfigurationWriter()
        def write(self, target, content):
            self.calls += 1
            if self.calls == 2: raise OSError("rollback write failed")
            self.writer.write(target, content)
    writer = FailRollbackWriter(); service = ConfigurationManagementService(store, writer)
    store.activate_candidate = lambda _: (_ for _ in ()).throw(RuntimeError("activation failed"))  # type: ignore[method-assign]
    with pytest.raises(ConfigurationStateIndeterminate): service.apply_policies(snapshot["policies"], snapshot["revision"])
    assert writer.calls == 2 and store.snapshot is previous


def test_policy_rollback_activation_failure_reports_indeterminate_state(tmp_path: Path):
    _, store, service = _policy_service(tmp_path)
    snapshot = service.policy_snapshot(); previous = store.snapshot
    store.activate_candidate = lambda _: (_ for _ in ()).throw(RuntimeError("activation failed"))  # type: ignore[method-assign]
    with pytest.raises(ConfigurationStateIndeterminate): service.apply_policies(snapshot["policies"], snapshot["revision"])
    assert store.snapshot is previous


def test_policy_confirmation_failure_restores_prior_state(tmp_path: Path):
    path, store, _ = _policy_service(tmp_path)
    snapshot = store.snapshot
    class MismatchedConfirmationWriter:
        def __init__(self): self.calls = 0; self.writer = AtomicConfigurationWriter()
        def write(self, target, content):
            self.calls += 1
            if self.calls == 1:
                document = yaml.safe_load(content); document["policies"][0]["actions"]["parameters"]["temperature"] = 0.9
                content = yaml.safe_dump(document)
            self.writer.write(target, content)
    writer = MismatchedConfirmationWriter(); service = ConfigurationManagementService(store, writer)
    before_revision = service.policy_snapshot()["revision"]
    with pytest.raises(ActivationFailed): service.apply_policies(service.policy_snapshot()["policies"], before_revision)
    assert writer.calls == 2 and store.snapshot is snapshot and service.policy_snapshot()["revision"] == before_revision


def test_model_profiles_are_atomic_revision_safe_candidates(tmp_path: Path):
    path, store, service = _policy_service(tmp_path)
    snapshot = service.model_profiles_snapshot(); before = path.read_bytes()
    models = snapshot["models"]
    models["qwable"]["aliases"] = {"anthropic": ["new-alias"]}
    validation = service.validate_model_profiles(models)
    assert validation.valid and path.read_bytes() == before
    result = service.apply_model_profiles(models, snapshot["revision"])
    assert result["activation_class"] == "reload_safe"
    assert result["active_revision"] == result["persisted_revision"]
    assert result["models"]["qwable"]["aliases"] == {"anthropic": ["new-alias"]}
    with pytest.raises(RevisionConflict): service.apply_model_profiles(models, snapshot["revision"])


def test_model_profile_validation_rejects_invalid_provider_without_writing(tmp_path: Path):
    path, _, service = _policy_service(tmp_path)
    snapshot = service.model_profiles_snapshot(); before = path.read_bytes()
    snapshot["models"]["qwable"]["provider"] = "missing"
    result = service.validate_model_profiles(snapshot["models"])
    assert not result.valid and path.read_bytes() == before
