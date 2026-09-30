from __future__ import annotations

import logging
import httpx

from llm_proxy.app import create_app
from llm_proxy.configuration.loader import ConfigurationLoader
from llm_proxy.configuration.store import ConfigurationStore
from llm_proxy.domain.content import TextContent
from llm_proxy.domain.errors import ProviderUnavailableError
from llm_proxy.domain.events import ResponseCompleted, ResponseStarted, TextDelta, ToolCallArgumentsDelta, ToolCallStarted
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
    def __init__(self) -> None:
        self.gateway = Gateway()

    def get(self, instance_name):
        return self.gateway


async def test_openai_completion_uses_canonical_service_with_injected_gateway(config_file) -> None:
    store = ConfigurationStore(config_file, ConfigurationLoader(), 1)
    app = create_app(store, Factory())
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://proxy.test") as client:
        response = await client.post("/v1/chat/completions", json={"model": "test-model", "messages": [{"role": "user", "content": "hello"}]})

    assert response.status_code == 200
    assert response.json()["model"] == "test-model"
    assert response.json()["choices"][0]["message"]["content"] == "hello"


async def test_canonical_request_emits_correlated_operational_lifecycle(config_file, caplog) -> None:
    app = create_app(ConfigurationStore(config_file, ConfigurationLoader(), 1), Factory())
    caplog.set_level(logging.DEBUG, logger="llm-proxy")
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://proxy.test") as client:
        response = await client.post("/v1/chat/completions", json={"model": "test-model", "messages": [{"role": "user", "content": "not logged"}]})

    assert response.status_code == 200
    messages = [record.getMessage() for record in caplog.records if record.name == "llm-proxy" and record.getMessage().startswith("completion_")]
    assert [message.split()[0] for message in messages] == ["completion_started", "completion_decision", "completion_finished"]
    request_id = messages[0].split("request_id=")[1].split()[0]
    assert all(f"request_id={request_id}" in message for message in messages)
    assert "mode=canonical" in messages[1]
    assert "not_logged" not in "\n".join(messages)


async def test_openai_copilot_compatible_request_reaches_gateway(config_file) -> None:
    class RecordingGateway(Gateway):
        def __init__(self) -> None:
            self.execution = None

        async def complete(self, execution):
            self.execution = execution
            return await super().complete(execution)

    gateway = RecordingGateway()

    class RecordingFactory:
        def get(self, instance_name):
            return gateway

    app = create_app(ConfigurationStore(config_file, ConfigurationLoader(), 1), RecordingFactory())
    payload = {
        "model": "test-model",
        "messages": [
            {"role": "developer", "content": "Be concise."},
            {"role": "user", "content": "hello"},
        ],
        "stream": False,
        "max_tokens": 256,
        "stream_options": {"include_usage": True},
        "parallel_tool_calls": True,
    }

    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://proxy.test") as client:
        response = await client.post("/v1/chat/completions", json=payload)

    assert response.status_code == 200
    assert gateway.execution is not None
    assert gateway.execution.request.messages[0].role.value == "developer"


async def test_openai_structured_user_text_reaches_gateway(config_file) -> None:
    class RecordingGateway(Gateway):
        def __init__(self) -> None:
            self.execution = None

        async def complete(self, execution):
            self.execution = execution
            return await super().complete(execution)

    gateway = RecordingGateway()

    class RecordingFactory:
        def get(self, instance_name):
            return gateway

    app = create_app(ConfigurationStore(config_file, ConfigurationLoader(), 1), RecordingFactory())
    payload = {
        "model": "test-model",
        "messages": [{"role": "user", "content": [
            {"type": "text", "text": "hello"},
            {"type": "text", "text": " world"},
        ]}],
    }

    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://proxy.test") as client:
        response = await client.post("/v1/chat/completions", json=payload)

    assert response.status_code == 200
    assert gateway.execution is not None
    assert gateway.execution.request.messages[0].content[0].text == "hello world"


async def test_openai_unsupported_structured_user_content_does_not_reach_gateway(config_file) -> None:
    class RecordingGateway(Gateway):
        def __init__(self) -> None:
            self.called = False

        async def complete(self, execution):
            self.called = True
            return await super().complete(execution)

    gateway = RecordingGateway()

    class RecordingFactory:
        def get(self, instance_name):
            return gateway

    app = create_app(ConfigurationStore(config_file, ConfigurationLoader(), 1), RecordingFactory())
    payload = {
        "model": "test-model",
        "messages": [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": "x"}}]}],
    }

    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://proxy.test") as client:
        response = await client.post("/v1/chat/completions", json=payload)

    assert response.status_code == 400
    assert gateway.called is False


async def test_openai_stream_projects_canonical_events(config_file) -> None:
    store = ConfigurationStore(config_file, ConfigurationLoader(), 1)
    app = create_app(store, Factory())
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://proxy.test") as client:
        response = await client.post("/v1/chat/completions", json={"model": "test-model", "stream": True, "messages": [{"role": "user", "content": "hello"}]})

    assert response.status_code == 200
    assert '"content":"hello"' in response.text
    assert response.text.endswith("data: [DONE]\n\n")


async def test_openai_tool_stream_projects_native_tool_deltas(config_file) -> None:
    class ToolGateway(Gateway):
        async def _events(self, execution):
            yield ResponseStarted("response-1", execution.upstream_model)
            yield ToolCallStarted("tool-1", "call-1", "read")
            yield ToolCallArgumentsDelta("tool-1", "call-1", '{"path":')
            yield ResponseCompleted(FinishReason.TOOL_USE, None, Usage(1, 2))

    class ToolFactory:
        def get(self, instance_name):
            return ToolGateway()

    app = create_app(ConfigurationStore(config_file, ConfigurationLoader(), 1), ToolFactory())
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://proxy.test") as client:
        response = await client.post("/v1/chat/completions", json={"model": "test-model", "stream": True, "messages": [{"role": "user", "content": "hello"}]})

    assert '"tool_calls":[{"index":0,"id":"call-1"' in response.text
    assert '"finish_reason":"tool_calls"' in response.text


async def test_openai_provider_failure_uses_protocol_error(config_file) -> None:
    class FailingGateway(Gateway):
        async def complete(self, execution):
            raise ProviderUnavailableError("internal provider detail")

    class FailingFactory:
        def get(self, instance_name):
            return FailingGateway()

    app = create_app(ConfigurationStore(config_file, ConfigurationLoader(), 1), FailingFactory())
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://proxy.test") as client:
        response = await client.post("/v1/chat/completions", json={"model": "test-model", "messages": [{"role": "user", "content": "hello"}]})

    assert response.status_code == 503
    assert response.json()["error"]["message"] == "upstream provider request failed"
