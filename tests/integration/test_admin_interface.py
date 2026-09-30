from pathlib import Path

import httpx
import pytest

from llm_proxy.app import create_app
from llm_proxy.configuration.loader import ConfigurationLoader
from llm_proxy.configuration.store import ConfigurationStore
from llm_proxy.management.errors import ActivationFailed, ConfigurationStateIndeterminate


def _config(enabled: bool, host: str = "127.0.0.1", allow_remote: bool = False) -> str:
    return f'''server:
  host: {host}
  port: 11435
  timeout_seconds: 0
  config_reload_seconds: 1
  log_level: INFO
management:
  enabled: {str(enabled).lower()}
  allow_remote: {str(allow_remote).lower()}
interfaces:
  openai: {{enabled: true}}
providers:
  test: {{extension: ollama, enabled: true, config: {{base_url: http://upstream.test, outbound_interface: openai}}}}
models:
  test: {{upstream_model: test, provider: test, interfaces: [openai], parameters: {{}}}}
policies: []
'''


@pytest.mark.asyncio
async def test_admin_routes_are_disabled_or_require_bearer_token(tmp_path: Path, monkeypatch, caplog):
    disabled_path = tmp_path / "disabled.yaml"
    disabled_path.write_text(_config(False), encoding="utf-8")
    disabled = create_app(ConfigurationStore(disabled_path, ConfigurationLoader(), 1))
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=disabled), base_url="http://proxy.test") as client:
        assert (await client.get("/_admin/v1/status")).status_code == 404

    monkeypatch.setenv("LLM_PROXY_ADMIN_TOKEN", "synthetic-token")
    enabled_path = tmp_path / "enabled.yaml"
    enabled_path.write_text(_config(True), encoding="utf-8")
    enabled = create_app(ConfigurationStore(enabled_path, ConfigurationLoader(), 1))
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=enabled), base_url="http://proxy.test") as client:
        assert (await client.get("/_admin/v1/status")).status_code == 401
        assert (await client.get("/_admin/v1/status", headers={"Authorization": "Bearer wrong"})).json() == {"error": {"code": "admin_unauthorized", "message": "unauthorized"}}
        response = await client.get("/_admin/v1/status", headers={"Authorization": "Bearer synthetic-token"})
        assert response.status_code == 200
        session = await client.get("/_admin/v1/session", headers={"Authorization": "Bearer synthetic-token"})
        assert session.json() == {"permissions": ["gateway.operations.view", "gateway.reload", "gateway.diagnostics.view"]}
    audit = next(record.message for record in caplog.records if record.message.startswith("admin operation=status"))
    assert "active_revision=" in audit
    assert "persisted_revision=" in audit
    assert "restart_required=False" in audit


@pytest.mark.asyncio
async def test_policy_routes_are_authenticated_and_strict(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("LLM_PROXY_ADMIN_TOKEN", "synthetic-token")
    path = tmp_path / "enabled.yaml"; path.write_text(_config(True), encoding="utf-8")
    app = create_app(ConfigurationStore(path, ConfigurationLoader(), 1)); headers = {"Authorization": "Bearer synthetic-token"}
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://proxy.test") as client:
        assert (await client.get("/_admin/v1/configuration/policies")).status_code == 401
        snapshot = await client.get("/_admin/v1/configuration/policies", headers=headers)
        assert snapshot.status_code == 200 and set(snapshot.json()) == {"revision", "policies"}
        assert (await client.post("/_admin/v1/configuration/policies/validate", headers=headers, json={"policies": [], "extra": True})).status_code == 422
        valid = await client.post("/_admin/v1/configuration/policies/validate", headers=headers, json={"policies": []})
        assert valid.status_code == 200 and set(valid.json()) == {"valid", "revision", "errors"}
        assert valid.json() == {"valid": True, "revision": snapshot.json()["revision"], "errors": []}
        malformed = await client.put("/_admin/v1/configuration/policies", headers=headers, json={"expected_revision": snapshot.json()["revision"], "policies": [], "extra": True})
        assert malformed.status_code == 422 and malformed.json()["error"]["code"] == "invalid_configuration"
        stale = await client.put("/_admin/v1/configuration/policies", headers=headers, json={"expected_revision": "stale", "policies": []})
        assert stale.status_code == 409 and stale.json()["error"]["code"] == "revision_conflict"
        applied = await client.put("/_admin/v1/configuration/policies", headers=headers, json={"expected_revision": snapshot.json()["revision"], "policies": []})
        assert applied.status_code == 200
        assert set(applied.json()) == {"active_revision", "persisted_revision", "restart_required", "policies"}
        assert applied.json()["active_revision"] == applied.json()["persisted_revision"] == snapshot.json()["revision"]
        assert "base_url" not in str(snapshot.json()) and "synthetic-token" not in str(snapshot.json())
        service = app.state.management_configuration_service
        def failing_apply(error):
            def fail(*_): raise error
            return fail
        monkeypatch.setattr(service, "apply_policies", failing_apply(ActivationFailed("activation failed")))
        activation = await client.put("/_admin/v1/configuration/policies", headers=headers, json={"expected_revision": snapshot.json()["revision"], "policies": []})
        assert activation.status_code == 500 and activation.json()["error"]["code"] == "activation_failed"
        monkeypatch.setattr(service, "apply_policies", failing_apply(ConfigurationStateIndeterminate("state unknown")))
        indeterminate = await client.put("/_admin/v1/configuration/policies", headers=headers, json={"expected_revision": snapshot.json()["revision"], "policies": []})
        assert indeterminate.status_code == 500 and indeterminate.json()["error"]["code"] == "configuration_state_indeterminate"


@pytest.mark.asyncio
async def test_administration_and_model_profile_routes_are_typed_redacted_and_strict(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("LLM_PROXY_ADMIN_TOKEN", "synthetic-token")
    path = tmp_path / "enabled.yaml"; path.write_text(_config(True), encoding="utf-8")
    app = create_app(ConfigurationStore(path, ConfigurationLoader(), 1)); headers = {"Authorization": "Bearer synthetic-token"}
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://proxy.test") as client:
        administration = await client.get("/_admin/v1/configuration/administration", headers=headers)
        assert administration.status_code == 200
        body = administration.json()
        assert body["providers"][0]["configured"] is True
        assert body["providers"][0]["lifecycle"] == "mixed"
        assert body["providers"][0]["reload_safe_fields"] == ["listing_enabled"]
        assert body["interfaces"][0]["lifecycle"] == "mixed"
        assert "base_url" not in str(body) and "synthetic-token" not in str(body)
        snapshot = await client.get("/_admin/v1/configuration/model-profiles", headers=headers)
        assert set(snapshot.json()) == {"revision", "models"}
        assert (await client.post("/_admin/v1/configuration/model-profiles/validate", headers=headers, json={"models": snapshot.json()["models"], "extra": True})).status_code == 422
        validation = await client.post("/_admin/v1/configuration/model-profiles/validate", headers=headers, json={"models": snapshot.json()["models"]})
        assert validation.json()["valid"] is True
        applied = await client.put("/_admin/v1/configuration/model-profiles", headers=headers, json={"expected_revision": snapshot.json()["revision"], "models": snapshot.json()["models"]})
        assert applied.status_code == 200 and applied.json()["activation_class"] == "reload_safe"


def test_enabled_management_requires_token_and_remote_opt_in(tmp_path: Path, monkeypatch):
    path = tmp_path / "config.yaml"
    path.write_text(_config(True), encoding="utf-8")
    monkeypatch.delenv("LLM_PROXY_ADMIN_TOKEN", raising=False)
    with pytest.raises(RuntimeError, match="LLM_PROXY_ADMIN_TOKEN"):
        create_app(ConfigurationStore(path, ConfigurationLoader(), 1))

    monkeypatch.setenv("LLM_PROXY_ADMIN_TOKEN", "synthetic-token")
    path.write_text(_config(True, host="0.0.0.0"), encoding="utf-8")
    with pytest.raises(RuntimeError, match="allow_remote"):
        create_app(ConfigurationStore(path, ConfigurationLoader(), 1))

    path.write_text(_config(True, host="0.0.0.0", allow_remote=True), encoding="utf-8")
    assert create_app(ConfigurationStore(path, ConfigurationLoader(), 1))
