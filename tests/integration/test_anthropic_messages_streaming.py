from __future__ import annotations

import httpx
import yaml

from llm_proxy.app import create_app
from llm_proxy.configuration.loader import ConfigurationLoader
from llm_proxy.configuration.store import ConfigurationStore
from llm_proxy.domain.errors import ProviderUnavailableError
from llm_proxy.domain.events import ResponseCompleted, ResponseFailed, ResponseStarted, TextCompleted, TextDelta, TextStarted
from llm_proxy.domain.responses import FinishReason, Usage



def anthropic_app(config_file, factory):
    config = yaml.safe_load(config_file.read_text())
    config["models"]["anthropic-model"] = {
        "upstream_model": "anthropic-upstream", "provider": "test-provider",
        "interfaces": ["anthropic"], "aliases": {"anthropic": ["local-opus"]},
    }
    config_file.write_text(yaml.safe_dump(config, sort_keys=False))
    return create_app(ConfigurationStore(config_file, ConfigurationLoader(), 1), factory)


def headers() -> dict[str, str]:
    return {"anthropic-version": "2023-06-01", "x-api-key": "any-token"}


class Gateway:
    async def complete(self, execution):
        raise AssertionError("not used")

    def stream(self, execution):
        return self._events(execution)

    async def _events(self, execution):
        yield ResponseStarted("response-1", execution.upstream_model)
        yield TextStarted("text-1")
        yield TextDelta("text-1", "hello")
        yield TextCompleted("text-1")
        yield ResponseCompleted(FinishReason.END_TURN, None, Usage(1, 2))


class FailingGateway(Gateway):
    async def _events(self, execution):
        yield ResponseStarted("response-1", execution.upstream_model)
        yield ResponseFailed(ProviderUnavailableError("internal provider detail"))


class Factory:
    def __init__(self, gateway) -> None:
        self.gateway = gateway

    def get(self, instance_name):
        return self.gateway


async def test_messages_streams_anthropic_sse(config_file) -> None:
    app = anthropic_app(config_file, Factory(Gateway()))
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://proxy.test") as client:
        response = await client.post("/v1/messages", headers=headers(), json={"model": "local-opus", "max_tokens": 10, "stream": True, "messages": [{"role": "user", "content": "hello"}]})

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    assert response.headers["cache-control"] == "no-cache"
    assert "event: message_start" in response.text
    assert response.text.endswith('event: message_stop\ndata: {"type":"message_stop"}\n\n')


async def test_messages_streams_anthropic_error_after_start(config_file) -> None:
    app = anthropic_app(config_file, Factory(FailingGateway()))
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://proxy.test") as client:
        response = await client.post("/v1/messages", headers=headers(), json={"model": "local-opus", "max_tokens": 10, "stream": True, "messages": [{"role": "user", "content": "hello"}]})

    assert "event: error" in response.text
    assert "message_stop" not in response.text
