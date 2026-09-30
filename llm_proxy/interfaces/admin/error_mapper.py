from __future__ import annotations

from fastapi import Request
from fastapi.responses import JSONResponse

from llm_proxy.management.errors import ManagementError


async def management_error(_: Request, error: ManagementError) -> JSONResponse:
    return JSONResponse({"error": {"code": error.code, "message": str(error)}}, status_code=error.status_code, headers={"WWW-Authenticate": "Bearer"} if error.status_code == 401 else None)
