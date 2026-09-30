from __future__ import annotations

import json
from pathlib import Path

import httpx
import yaml

from llm_proxy.app import create_app
from llm_proxy.configuration.loader import ConfigurationLoader
from llm_proxy.configuration.store import ConfigurationStore
from llm_proxy.domain.content import TextContent
from llm_proxy.domain.events import ResponseCompleted, ResponseStarted, TextCompleted, TextDelta, TextStarted
from llm_proxy.domain.messages import Message, MessageRole
from llm_proxy.domain.responses import CompletionResponse, FinishReason, Usage


class Gateway:
    def __init__(self) -> None:
        self.execution = None

    async def complete(self, execution):
        self.execution = execution
        return CompletionResponse("response", "upstream", Message(MessageRole.ASSISTANT, (TextContent("ready"),)), FinishReason.END_TURN, None, Usage(1, 2))

    def stream(self, execution):
        self.execution = execution

        async def events():
            yield ResponseStarted("response", "upstream")
            yield TextStarted("text")
            yield TextDelta("text", "ready")
            yield TextCompleted("text")
            yield ResponseCompleted(FinishReason.END_TURN, None, Usage(1, 2))

        return events()


class Factory:
    def __init__(self) -> None:
        self.gateway = Gateway()

    def get(self, instance_name):
        return self.gateway


def app_for_session(config_file, factory: Factory):
    config = yaml.safe_load(config_file.read_text())
    config["models"]["anthropic-model"] = {
        "upstream_model": "anthropic-upstream",
        "provider": "test-provider",
        "interfaces": ["anthropic"],
        "aliases": {"anthropic": ["local-opus"]},
    }
    config_file.write_text(yaml.safe_dump(config, sort_keys=False))
    return create_app(ConfigurationStore(config_file, ConfigurationLoader(), 1), factory)


async def test_captured_claude_code_session_request_rejects_unimplemented_reasoning(config_file) -> None:
    factory = Factory()
    app = app_for_session(config_file, factory)
    payload = json.loads(Path("tests/fixtures/claude_code/session-request.json").read_text())
    headers = {
        "anthropic-version": "2023-06-01",
        "anthropic-beta": "context-management-2025-06-27,prompt-caching-scope-2026-01-05",
        "x-api-key": "any-token",
    }
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://proxy.test") as client:
        response = await client.post("/v1/messages", headers=headers, json=payload)

    assert response.status_code == 400
    assert response.json()["error"]["message"] == "selected execution does not support reasoning"
    assert factory.gateway.execution is None
