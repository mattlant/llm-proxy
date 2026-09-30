from __future__ import annotations

import httpx
import pytest

from llm_proxy.app import create_app
from llm_proxy.configuration.loader import ConfigurationLoader
from llm_proxy.configuration.store import ConfigurationStore
from tests.integration.test_admin_interface import _config
from tests.integration.test_openai_interface import Factory


@pytest.mark.asyncio
async def test_failed_admin_validation_does_not_change_openai_completion(tmp_path, monkeypatch):
    monkeypatch.setenv("LLM_PROXY_ADMIN_TOKEN", "synthetic-token")
    path = tmp_path / "config.yaml"
    path.write_text(_config(True), encoding="utf-8")
    app = create_app(ConfigurationStore(path, ConfigurationLoader(), 1), Factory())
    headers = {"Authorization": "Bearer synthetic-token"}
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://proxy.test") as client:
        validation = await client.post("/_admin/v1/configuration/validate", headers=headers, json={})
        completion = await client.post("/v1/chat/completions", json={"model": "test", "messages": [{"role": "user", "content": "hello"}]})

    assert validation.status_code == 200
    assert validation.json()["valid"] is False
    assert completion.status_code == 200
    assert completion.json()["choices"][0]["message"]["content"] == "hello"
