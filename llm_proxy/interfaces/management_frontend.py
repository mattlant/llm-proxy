from __future__ import annotations

import json
import mimetypes
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.exceptions import HTTPException
from fastapi.responses import PlainTextResponse, Response
from fastapi.staticfiles import StaticFiles

MANAGEMENT_STATIC_DIRECTORY = Path(__file__).resolve().parents[1] / "static" / "management"
UNAVAILABLE_MESSAGE = "Management frontend assets are unavailable."


def _readable_file(path: Path) -> bool:
    try:
        return path.is_file() and bool(path.read_bytes())
    except OSError:
        return False


def _valid(directory: Path) -> bool:
    manifest_path = directory / "asset-manifest.json"
    if not _readable_file(directory / "index.html") or not _readable_file(manifest_path):
        return False
    try:
        assets = json.loads(manifest_path.read_text(encoding="utf-8"))["assets"]
    except (json.JSONDecodeError, KeyError, OSError, TypeError):
        return False
    if not isinstance(assets, list) or "index.html" not in assets:
        return False
    for asset in assets:
        if not isinstance(asset, str) or not asset or Path(asset).is_absolute() or ".." in Path(asset).parts:
            return False
        if not _readable_file(directory / asset):
            return False
    return any(asset.endswith(".js") and "." in asset for asset in assets)


class ManagementStaticFiles(StaticFiles):
    """Use bounded eager responses for package assets in ASGI transports and production."""

    def file_response(self, full_path, stat_result, scope, status_code: int = 200) -> Response:
        content = Path(full_path).read_bytes()
        media_type, _ = mimetypes.guess_type(str(full_path))
        return Response(content=content, status_code=status_code, media_type=media_type)

    async def check_config(self) -> None:
        return None

    async def get_response(self, path: str, scope) -> Response:
        if scope["method"] not in {"GET", "HEAD"}:
            raise HTTPException(status_code=405)
        full_path, stat_result = self.lookup_path(path)
        if stat_result and full_path:
            if Path(full_path).is_file():
                return self.file_response(full_path, stat_result, scope)
            if Path(full_path).is_dir() and self.html:
                index_path, index_stat = self.lookup_path(str(Path(path) / "index.html"))
                if index_stat and index_path and Path(index_path).is_file():
                    return self.file_response(index_path, index_stat, scope)
        raise HTTPException(status_code=404)


def mount_management_frontend(app: FastAPI, static_directory: Path = MANAGEMENT_STATIC_DIRECTORY) -> None:
    if _valid(static_directory):
        app.mount("/management", ManagementStaticFiles(directory=static_directory, html=True, follow_symlink=False), name="management_frontend")
        return

    async def unavailable(_: Request) -> PlainTextResponse:
        return PlainTextResponse(UNAVAILABLE_MESSAGE, status_code=503)

    app.add_api_route("/management", unavailable, methods=["GET", "HEAD"], include_in_schema=False)
    app.add_api_route("/management/{asset_path:path}", unavailable, methods=["GET", "HEAD"], include_in_schema=False)
