from __future__ import annotations

import httpx
import yaml

from llm_proxy.app import create_app
from llm_proxy.configuration.loader import ConfigurationLoader
from llm_proxy.configuration.store import ConfigurationStore
from llm_proxy.domain.content import TextContent, ToolCallContent
from llm_proxy.domain.errors import ProviderUnavailableError
from llm_proxy.domain.messages import Message, MessageRole
from llm_proxy.domain.responses import CompletionResponse, FinishReason, Usage


def anthropic_app(config_file, factory):
    config = yaml.safe_load(config_file.read_text())
    config["models"]["anthropic-model"] = {
        "upstream_model": "anthropic-upstream",
        "provider": "test-provider",
        "interfaces": ["anthropic"],
        "aliases": {"anthropic": ["local-opus"]},
    }
    config_file.write_text(yaml.safe_dump(config, sort_keys=False))
    return create_app(ConfigurationStore(config_file, ConfigurationLoader(), 1), factory)


def headers() -> dict[str, str]:
    return {"anthropic-version": "2023-06-01", "x-api-key": "any-token"}


class Gateway:
    async def complete(self, execution):
        return CompletionResponse("response-1", execution.upstream_model, Message(MessageRole.ASSISTANT, (TextContent("hello"),)), FinishReason.END_TURN, None, Usage(1, 2))

    def stream(self, execution):
        raise AssertionError("not used")


class Factory:
    def __init__(self) -> None:
        self.gateway = Gateway()

    def get(self, instance_name):
        return self.gateway


async def test_messages_non_streaming_uses_anthropic_alias_and_authentication(config_file) -> None:
    app = anthropic_app(config_file, Factory())
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://proxy.test") as client:
        response = await client.post("/v1/messages", headers=headers(), json={"model": "local-opus", "max_tokens": 10, "messages": [{"role": "user", "content": "hello"}]})

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("application/json")
    assert response.json()["model"] == "local-opus"
    assert response.json()["content"] == [{"type": "text", "text": "hello"}]


async def test_messages_requires_anthropic_headers_and_authentication(config_file) -> None:
    app = anthropic_app(config_file, Factory())
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://proxy.test") as client:
        missing_version = await client.post("/v1/messages", headers={"x-api-key": "token"}, json={"model": "local-opus", "max_tokens": 10, "messages": [{"role": "user", "content": "hello"}]})
        missing_token = await client.post("/v1/messages", headers={"anthropic-version": "2023-06-01"}, json={"model": "local-opus", "max_tokens": 10, "messages": [{"role": "user", "content": "hello"}]})

    assert missing_version.status_code == 400
    assert missing_version.json()["error"]["type"] == "invalid_request_error"
    assert missing_token.status_code == 401
    assert missing_token.json()["error"]["type"] == "authentication_error"


async def test_messages_provider_failure_before_stream_uses_anthropic_error(config_file) -> None:
    class FailingGateway(Gateway):
        async def complete(self, execution):
            raise ProviderUnavailableError("http://internal-provider")

    class FailingFactory:
        def get(self, instance_name):
            return FailingGateway()

    app = anthropic_app(config_file, FailingFactory())
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://proxy.test") as client:
        response = await client.post("/v1/messages", headers=headers(), json={"model": "local-opus", "max_tokens": 10, "messages": [{"role": "user", "content": "hello"}]})

    assert response.status_code == 503
    assert response.json() == {"type": "error", "error": {"type": "overloaded_error", "message": "Upstream provider is unavailable"}}
