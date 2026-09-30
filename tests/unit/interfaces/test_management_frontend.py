from __future__ import annotations

import httpx
from fastapi import FastAPI

from llm_proxy.interfaces.management_frontend import MANAGEMENT_STATIC_DIRECTORY, UNAVAILABLE_MESSAGE, mount_management_frontend


async def test_package_static_directory_is_fixed() -> None:
    assert MANAGEMENT_STATIC_DIRECTORY.as_posix().endswith("llm_proxy/static/management")


async def test_missing_assets_return_deterministic_unavailable_response() -> None:
    app = FastAPI()
    mount_management_frontend(app, static_directory=MANAGEMENT_STATIC_DIRECTORY / "missing")
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        response = await client.get("/management/")
        assert response.status_code == 503
        assert response.text == UNAVAILABLE_MESSAGE
        assert (await client.head("/management/anything.js")).status_code == 503


async def test_static_assets_are_served_without_traversal(tmp_path) -> None:
    (tmp_path / "index.html").write_text("<html>management</html>")
    (tmp_path / "asset-manifest.json").write_text('{"assets": ["index.html", "main.12345678.js"]}')
    (tmp_path / "main.12345678.js").write_text("console.log('ok')")
    app = FastAPI()
    mount_management_frontend(app, tmp_path)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        assert (await client.get("/management/")).text == "<html>management</html>"
        assert (await client.get("/management/main.12345678.js")).status_code == 200
        assert (await client.get("/management/../pyproject.toml", follow_redirects=False)).status_code in {404, 503}
        assert (await client.get("/management/%2e%2e/pyproject.toml", follow_redirects=False)).status_code in {404, 503}


async def test_damaged_asset_sets_are_unavailable(tmp_path) -> None:
    (tmp_path / "index.html").write_text("<html>management</html>")
    (tmp_path / "asset-manifest.json").write_text('{"assets": ["index.html", "missing.12345678.js"]}')
    app = FastAPI()
    mount_management_frontend(app, tmp_path)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        assert (await client.get("/management/")).status_code == 503


async def test_malformed_manifest_is_unavailable(tmp_path) -> None:
    (tmp_path / "index.html").write_text("<html>management</html>")
    (tmp_path / "asset-manifest.json").write_text("not json")
    app = FastAPI()
    mount_management_frontend(app, tmp_path)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        assert (await client.get("/management/")).text == UNAVAILABLE_MESSAGE
