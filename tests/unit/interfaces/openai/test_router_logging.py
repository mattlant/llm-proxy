import logging

import pytest
from starlette.requests import Request

from llm_proxy.interfaces.openai.router import chat_completions


@pytest.mark.asyncio
async def test_debug_logs_unhandled_chat_completion_exception(caplog: pytest.LogCaptureFixture) -> None:
    async def receive():
        return {"type": "http.request", "body": b"{}", "more_body": False}

    request = Request({"type": "http", "method": "POST", "path": "/v1/chat/completions", "headers": [], "app": object()}, receive)

    with caplog.at_level(logging.DEBUG, logger="llm-proxy"):
        response = await chat_completions(request)

    assert response.status_code == 500
    assert "OpenAI chat completion failed" in caplog.text
