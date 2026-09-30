from __future__ import annotations

import hmac
import os

from fastapi import Request

from .errors import AdminUnauthorized


class AdminAuthenticator:
    def __init__(self, token_env: str) -> None:
        token = os.getenv(token_env)
        if not token:
            raise RuntimeError(f"management enabled but {token_env} is unavailable")
        self._token = token

    async def authenticate(self, request: Request) -> None:
        authorization = request.headers.get("authorization", "")
        scheme, _, token = authorization.partition(" ")
        if scheme.lower() != "bearer" or not token or not hmac.compare_digest(token, self._token):
            raise AdminUnauthorized("unauthorized")

    def permissions(self) -> list[str]:
        return ["gateway.operations.view", "gateway.reload", "gateway.diagnostics.view"]
