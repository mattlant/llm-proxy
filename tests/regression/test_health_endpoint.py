from __future__ import annotations


async def test_health_reports_configuration(client):
    response = await client.get("/_proxy/health")

    assert response.status_code == 200
    health = response.json()
    assert health == {
        "ok": True,
        "config_file": health["config_file"],
        "loaded_at": health["loaded_at"],
        "interfaces": {"openai": True, "ollama": True, "anthropic": True},
        "providers": {
            "test-provider": {
                    "extension": "ollama",
            }
        },
        "models": {
            "test-model": {
                "interfaces": ["openai", "ollama"],
                "provider": "test-provider",
            }
        },
    }
    assert "authentication" not in health
    assert "token" not in health
