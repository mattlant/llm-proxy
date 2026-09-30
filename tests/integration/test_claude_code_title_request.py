from __future__ import annotations

import json
from pathlib import Path

import httpx
import yaml

from llm_proxy.app import create_app
from llm_proxy.configuration.loader import ConfigurationLoader
from llm_proxy.configuration.store import ConfigurationStore
from llm_proxy.domain.content import TextContent
from llm_proxy.domain.messages import Message, MessageRole
from llm_proxy.domain.responses import CompletionResponse, FinishReason, Usage


def headers() -> dict[str, str]:
    return {"anthropic-version": "2023-06-01", "x-api-key": "any-token"}


def title_request() -> dict:
    return json.loads(Path("tests/fixtures/claude_code/title-request.json").read_text())


class Gateway:
    def __init__(self) -> None:
        self.execution = None

    async def complete(self, execution):
        self.execution = execution
        return CompletionResponse("response-1", execution.upstream_model, Message(MessageRole.ASSISTANT, (TextContent('{"title":"smoke"}'),)), FinishReason.END_TURN, None, Usage(1, 2))

    def stream(self, execution):
        raise AssertionError("not used")


class Factory:
    def __init__(self) -> None:
        self.gateway = Gateway()
        self.looked_up = False

    def get(self, instance_name):
        self.looked_up = True
        return self.gateway


def app_for_title(config_file, factory: Factory, *, structured_output: str):
    config = yaml.safe_load(config_file.read_text())
    config["models"]["anthropic-model"] = {
        "upstream_model": "anthropic-upstream",
        "provider": "test-provider",
        "interfaces": ["anthropic"],
        "aliases": {"anthropic": ["local-opus"]},
        "compatibility": {"structured_output": structured_output},
    }
    config_file.write_text(yaml.safe_dump(config, sort_keys=False))
    return create_app(ConfigurationStore(config_file, ConfigurationLoader(), 1), factory)


async def test_title_request_preserves_json_schema_to_execution(config_file) -> None:
    factory = Factory()
    app = app_for_title(config_file, factory, structured_output="openai_json_schema")
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://proxy.test") as client:
        response = await client.post("/v1/messages", headers=headers(), json=title_request())

    assert response.status_code == 200
    assert response.json()["content"] == [{"type": "text", "text": '{"title":"smoke"}'}]
    assert factory.gateway.execution.request.controls.output_constraint.schema["properties"]["title"]["type"] == "string"


async def test_title_request_with_unsupported_model_fails_before_gateway_creation(config_file) -> None:
    factory = Factory()
    app = app_for_title(config_file, factory, structured_output="unsupported")
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://proxy.test") as client:
        response = await client.post("/v1/messages", headers=headers(), json=title_request())

    assert response.status_code == 400
    assert response.json()["error"] == {"type": "invalid_request_error", "message": "selected model does not support JSON schema structured output"}
    assert factory.looked_up is False
