from __future__ import annotations

import httpx
import yaml

from llm_proxy.app import create_app
from llm_proxy.configuration.loader import ConfigurationLoader
from llm_proxy.configuration.store import ConfigurationStore
from llm_proxy.domain.content import TextContent, ToolCallContent
from llm_proxy.domain.messages import Message, MessageRole
from llm_proxy.domain.responses import CompletionResponse, FinishReason, Usage



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
    def __init__(self) -> None:
        self.executions = []

    async def complete(self, execution):
        self.executions.append(execution)
        if len(self.executions) == 1:
            message = Message(MessageRole.ASSISTANT, (TextContent("Reading"), ToolCallContent("call-1", "Read", {"path": "README"})))
            return CompletionResponse("response-1", execution.upstream_model, message, FinishReason.TOOL_USE, None, Usage(1, 2))
        return CompletionResponse("response-2", execution.upstream_model, Message(MessageRole.ASSISTANT, (TextContent("Done"),)), FinishReason.END_TURN, None, Usage(3, 4))

    def stream(self, execution):
        raise AssertionError("not used")


class Factory:
    def __init__(self, gateway) -> None:
        self.gateway = gateway

    def get(self, instance_name):
        return self.gateway


async def test_tool_use_and_tool_result_continue_as_canonical_history(config_file) -> None:
    gateway = Gateway()
    app = anthropic_app(config_file, Factory(gateway))
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://proxy.test") as client:
        first = await client.post("/v1/messages", headers=headers(), json={"model": "local-opus", "max_tokens": 10, "messages": [{"role": "user", "content": "read it"}], "tools": [{"name": "Read", "input_schema": {}}]})
        second = await client.post("/v1/messages", headers=headers(), json={"model": "local-opus", "max_tokens": 10, "messages": [
            {"role": "user", "content": "read it"},
            {"role": "assistant", "content": first.json()["content"]},
            {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "call-1", "content": "contents", "is_error": True}]},
        ], "tools": [{"name": "Read", "input_schema": {}}]})

    assert first.json()["content"][1] == {"type": "tool_use", "id": "call-1", "name": "Read", "input": {"path": "README"}}
    result = gateway.executions[1].request.messages[-1].content[0]
    assert result.tool_call_id == "call-1"
    assert result.is_error is True
    assert second.json()["content"] == [{"type": "text", "text": "Done"}]
