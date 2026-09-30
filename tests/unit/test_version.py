from __future__ import annotations

import tomllib
from pathlib import Path
from types import SimpleNamespace

from llm_proxy import __version__
from llm_proxy.app import create_app
from llm_proxy.management.service import ManagementQueryService
from llm_proxy.providers.ollama.extension import OllamaExtension


def test_runtime_versions_match_package_metadata() -> None:
    with Path("pyproject.toml").open("rb") as file:
        package_version = tomllib.load(file)["project"]["version"]

    assert package_version == "0.4.0"
    assert __version__ == package_version
    assert create_app().version == package_version
    assert OllamaExtension.metadata.package_version == package_version


def test_management_status_uses_package_version() -> None:
    service = ManagementQueryService.__new__(ManagementQueryService)
    service._store = SimpleNamespace(snapshot=SimpleNamespace(config=SimpleNamespace(interfaces={}, server=SimpleNamespace(host="127.0.0.1")), loaded_at=None), last_rejected_reload=None)
    service._configuration = SimpleNamespace(active_revision="active", persisted_revision="active")

    assert service.status()["gateway_version"] == __version__
