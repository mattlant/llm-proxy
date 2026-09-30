from __future__ import annotations

import httpx
import yaml

from llm_proxy.app import create_app
from llm_proxy.configuration.loader import ConfigurationLoader
from llm_proxy.configuration.store import ConfigurationStore
from llm_proxy.domain.content import TextContent, ToolCallContent
from llm_proxy.domain.messages import Message, MessageRole
from llm_proxy.domain.responses import CompletionResponse, FinishReason, Usage


class Gateway:
    def __init__(self) -> None:
        self.executions = []

    async def complete(self, execution):
        self.executions.append(execution)
        if len(self.executions) == 1:
            return CompletionResponse("first", "upstream", Message(MessageRole.ASSISTANT, (ToolCallContent("call-1", "Read", {"path": "fixture"}),)), FinishReason.TOOL_USE, None, Usage(1, 2))
        return CompletionResponse("second", "upstream", Message(MessageRole.ASSISTANT, (TextContent("complete"),)), FinishReason.END_TURN, None, Usage(3, 4))

    def stream(self, execution):
        raise AssertionError("not used")


class Factory:
    def __init__(self) -> None:
        self.gateway = Gateway()

    def get(self, instance_name):
        return self.gateway


def app_for_loop(config_file, factory: Factory):
    config = yaml.safe_load(config_file.read_text())
    config["models"]["anthropic-model"] = {
        "upstream_model": "anthropic-upstream", "provider": "test-provider",
        "interfaces": ["anthropic"], "aliases": {"anthropic": ["local-opus"]},
    }
    config_file.write_text(yaml.safe_dump(config, sort_keys=False))
    return create_app(ConfigurationStore(config_file, ConfigurationLoader(), 1), factory)


async def test_tool_result_continuation_preserves_id_and_uses_canonical_tool_role(config_file) -> None:
    factory = Factory()
    app = app_for_loop(config_file, factory)
    headers = {"anthropic-version": "2023-06-01", "x-api-key": "any-token"}
    tool = {"name": "Read", "input_schema": {"type": "object"}}
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://proxy.test") as client:
        first = await client.post("/v1/messages", headers=headers, json={"model": "local-opus", "max_tokens": 10, "messages": [{"role": "user", "content": "read"}], "tools": [tool]})
        second = await client.post("/v1/messages", headers=headers, json={
            "model": "local-opus", "max_tokens": 10, "tools": [tool],
            "messages": [
                {"role": "user", "content": "read"},
                {"role": "assistant", "content": first.json()["content"]},
                {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "call-1", "content": "result"}]},
            ],
        })

    assert first.json()["content"] == [{"type": "tool_use", "id": "call-1", "name": "Read", "input": {"path": "fixture"}}]
    assert factory.gateway.executions[1].request.messages[-1].role is MessageRole.TOOL
    assert factory.gateway.executions[1].request.messages[-1].content[0].tool_call_id == "call-1"
    assert second.json()["content"] == [{"type": "text", "text": "complete"}]
