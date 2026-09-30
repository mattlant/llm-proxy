from __future__ import annotations

import json

import httpx

from llm_proxy.app import create_app
from llm_proxy.configuration.loader import ConfigurationLoader
from llm_proxy.configuration.store import ConfigurationStore
from llm_proxy.domain.content import TextContent
from llm_proxy.domain.events import ResponseCompleted, ResponseStarted, TextDelta
from llm_proxy.domain.messages import Message, MessageRole
from llm_proxy.domain.responses import CompletionResponse, FinishReason, Usage


class Gateway:
    async def complete(self, execution):
        return CompletionResponse("response-1", execution.upstream_model, Message(MessageRole.ASSISTANT, (TextContent("hello"),)), FinishReason.END_TURN, None, Usage(1, 2))

    def stream(self, execution):
        return self._events(execution)

    async def _events(self, execution):
        yield ResponseStarted("response-1", execution.upstream_model)
        yield TextDelta("text-1", "hello")
        yield ResponseCompleted(FinishReason.END_TURN, None, Usage(1, 2))


class Factory:
    def get(self, instance_name):
        return Gateway()


async def test_native_chat_and_generate_use_canonical_service(config_file) -> None:
    app = create_app(ConfigurationStore(config_file, ConfigurationLoader(), 1), Factory())
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://proxy.test") as client:
        chat = await client.post("/api/chat", json={"model": "test-model", "stream": False, "messages": [{"role": "user", "content": "hello"}]})
        generate = await client.post("/api/generate", json={"model": "test-model", "stream": False, "prompt": "hello"})
        streamed = await client.post("/api/chat", json={"model": "test-model", "stream": True, "messages": [{"role": "user", "content": "hello"}]})
        generated_stream = await client.post("/api/generate", json={"model": "test-model", "stream": True, "prompt": "hello"})

    assert chat.json()["message"]["content"] == "hello"
    assert generate.json()["response"] == "hello"
    assert [json.loads(line) for line in streamed.text.splitlines()][-1]["done"] is True
    assert [json.loads(line) for line in generated_stream.text.splitlines()][0]["response"] == "hello"


async def test_show_resolves_ollama_model_metadata(config_file) -> None:
    app = create_app(ConfigurationStore(config_file, ConfigurationLoader(), 1), Factory())
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://proxy.test") as client:
        response = await client.post("/api/show", json={"model": "test-model"})

    assert response.status_code == 200
    assert response.json()["model"] == "test-model"
    assert response.json()["capabilities"] == ["completion", "tools"]
