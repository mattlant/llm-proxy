from __future__ import annotations

import httpx
import yaml

from llm_proxy.app import create_app
from llm_proxy.configuration.loader import ConfigurationLoader
from llm_proxy.configuration.store import ConfigurationStore
from llm_proxy.domain.content import ReasoningContent, TextContent
from llm_proxy.domain.events import ReasoningCompleted, ReasoningDelta, ReasoningStarted, ResponseCompleted, ResponseStarted, TextCompleted, TextDelta, TextStarted
from llm_proxy.domain.messages import Message, MessageRole
from llm_proxy.domain.responses import CompletionResponse, FinishReason, Usage


def app_for_anthropic(config_file, factory):
    config = yaml.safe_load(config_file.read_text())
    config["models"]["anthropic-model"] = {
        "upstream_model": "anthropic-upstream",
        "provider": "test-provider",
        "interfaces": ["anthropic"],
        "aliases": {"anthropic": ["local-opus"]},
    }
    config_file.write_text(yaml.safe_dump(config, sort_keys=False))
    return create_app(ConfigurationStore(config_file, ConfigurationLoader(), 1), factory)


def headers(*, beta: str | None = None) -> dict[str, str]:
    values = {"anthropic-version": "2023-06-01", "x-api-key": "any-token"}
    if beta is not None:
        values["anthropic-beta"] = beta
    return values


def payload(*, stream: bool = False) -> dict:
    return {
        "model": "local-opus",
        "max_tokens": 1024,
        "messages": [{"role": "user", "content": "hello"}],
        "stream": stream,
        "context_management": {"edits": [{"type": "clear_thinking_20251015", "keep": "all"}]},
    }


class Gateway:
    def __init__(self) -> None:
        self.execution = None

    async def complete(self, execution):
        self.execution = execution
        return CompletionResponse("response", "upstream", Message(MessageRole.ASSISTANT, (ReasoningContent("hidden"), TextContent("visible"))), FinishReason.END_TURN, None, Usage(1, 2))

    def stream(self, execution):
        self.execution = execution

        async def events():
            yield ResponseStarted("response", "upstream")
            yield ReasoningStarted("reasoning")
            yield ReasoningDelta("reasoning", "hidden")
            yield ReasoningCompleted("reasoning")
            yield TextStarted("text")
            yield TextDelta("text", "visible")
            yield TextCompleted("text")
            yield ResponseCompleted(FinishReason.END_TURN, None, Usage(1, 2))

        return events()


class Factory:
    def __init__(self) -> None:
        self.gateway = Gateway()

    def get(self, instance_name):
        return self.gateway


async def test_context_management_is_a_truthful_noop_without_reasoning(config_file) -> None:
    factory = Factory()
    app = app_for_anthropic(config_file, factory)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://proxy.test") as client:
        response = await client.post("/v1/messages", headers=headers(beta="context-management-2025-06-27"), json=payload())

    assert response.status_code == 200
    assert response.json()["content"] == [{"type": "text", "text": "visible"}]
    assert response.json()["context_management"] == {"applied_edits": []}
    assert factory.gateway.execution.request.controls.reasoning is None


async def test_streaming_context_management_has_no_thinking_events(config_file) -> None:
    app = app_for_anthropic(config_file, Factory())
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://proxy.test") as client:
        response = await client.post("/v1/messages", headers=headers(), json=payload(stream=True))

    assert response.status_code == 200
    assert "thinking" not in response.text
    assert '"context_management":{"applied_edits":[]}' in response.text


async def test_rejects_unknown_context_management_beta_when_one_is_supplied(config_file) -> None:
    app = app_for_anthropic(config_file, Factory())
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://proxy.test") as client:
        response = await client.post("/v1/messages", headers=headers(beta="context-management-20990101"), json=payload())

    assert response.status_code == 400
    assert response.json()["error"]["type"] == "invalid_request_error"
