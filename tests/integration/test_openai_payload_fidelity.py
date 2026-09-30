from __future__ import annotations

import base64
import json
import logging

import httpx

from llm_proxy.app import create_app
from llm_proxy.configuration.loader import ConfigurationLoader
from llm_proxy.configuration.store import ConfigurationStore
from llm_proxy.domain.content import TextContent
from llm_proxy.domain.events import ResponseCompleted, ResponseStarted, TextDelta
from llm_proxy.domain.messages import Message, MessageRole
from llm_proxy.domain.responses import CompletionResponse, FinishReason, Usage


def traced_bytes(value: dict) -> bytes:
    if value["encoding"] == "utf-8":
        return value["data"].encode("utf-8")
    assert value["encoding"] == "base64"
    return base64.b64decode(value["data"])


class Gateway:
    async def complete(self, execution):
        return CompletionResponse("response", execution.upstream_model, Message(MessageRole.ASSISTANT, (TextContent("answer"),)), FinishReason.END_TURN, None, Usage(1, 2))

    def stream(self, execution):
        async def events():
            yield ResponseStarted("response", execution.upstream_model)
            yield TextDelta("text", "answer")
            yield ResponseCompleted(FinishReason.END_TURN, None, Usage(1, 2))
        return events()


class Gateways:
    def get(self, instance_name):
        return Gateway()


def traced_app(config_file, gateways, *, include_content=True):
    config_file.write_text(config_file.read_text(encoding="utf-8").replace("  log_level: INFO", f"  log_level: INFO\n  payload_trace_mode: both\n  payload_trace_include_content: {'true' if include_content else 'false'}"), encoding="utf-8")
    return create_app(ConfigurationStore(config_file, ConfigurationLoader(), 1), gateways)


def terminal_outcomes(caplog) -> list[str]:
    return [json.loads(record.message)["outcome"] for record in caplog.records if record.name == "llm-proxy.payload" and '"stage":"transaction_end"' in record.message]


async def test_enabled_trace_correlates_json_and_sse_client_boundaries(config_file, caplog) -> None:
    app = traced_app(config_file, Gateways())
    caplog.set_level(logging.DEBUG, logger="llm-proxy.payload")
    json_body = b'{ "model" : "test-model", "messages" : [{"role":"user","content":"hello"}], "api_key":"hidden" }'
    stream_body = b'{ "model":"test-model", "stream":true, "messages":[{"role":"user","content":"hello"}] }'
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://proxy.test") as client:
        json_response = await client.post("/v1/chat/completions?trace=one", content=json_body, headers=[("X-Dupe", "one"), ("x-dupe", "two"), ("content-type", "application/json")])
        stream_response = await client.post("/v1/chat/completions", content=stream_body, headers={"content-type": "application/json"})
        assert json_response.status_code == 200
        assert stream_response.text.endswith("data: [DONE]\n\n")
    records = [json.loads(record.message) for record in caplog.records if record.name == "llm-proxy.payload"]
    by_transaction: dict[str, list[dict]] = {}
    for record in records:
        by_transaction.setdefault(record["transaction_id"], []).append(record)
    assert len(by_transaction) == 2
    assert all(values[0]["stage"] == "client_request_raw" and values[-1]["stage"] == "transaction_end" for values in by_transaction.values())
    assert all([item["sequence"] for item in values] == list(range(1, len(values) + 1)) for values in by_transaction.values())
    assert any(record["stage"] == "client_stream_record" and record["payload"] == "[DONE]" for record in records)
    assert any(record.get("payload", {}).get("api_key") == "[REDACTED]" for record in records if isinstance(record.get("payload"), dict))
    json_transaction = next(values for values in by_transaction.values() if traced_bytes(values[0]["body"]) == json_body)
    stream_transaction = next(values for values in by_transaction.values() if traced_bytes(values[0]["body"]) == stream_body)
    assert json_transaction[0]["body"]["encoding"] == "utf-8"
    assert b"".join(traced_bytes(record["body"]) for record in json_transaction if record["stage"] == "client_response_raw") == json_response.content
    assert b"".join(traced_bytes(record["body"]) for record in stream_transaction if record["stage"] == "client_stream_bytes_raw") == stream_response.content
    raw_headers = json_transaction[0]["http"]["headers"]
    assert [(pair["name"]["data"].encode(), pair["value"]["data"].encode()) for pair in raw_headers if pair["name"]["data"].lower() == "x-dupe"] == [(b"x-dupe", b"one"), (b"x-dupe", b"two")]


async def test_trace_can_omit_content_while_retaining_structure_and_raw_metadata(config_file, caplog) -> None:
    app = traced_app(config_file, Gateways(), include_content=False)
    caplog.set_level(logging.DEBUG, logger="llm-proxy.payload")
    body = {"model": "test-model", "messages": [{"role": "user", "content": "secret prompt"}], "stream": False}
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://proxy.test") as client:
        response = await client.post("/v1/chat/completions", json=body)
    assert response.status_code == 200
    records = [json.loads(record.message) for record in caplog.records if record.name == "llm-proxy.payload"]
    raw_records = [record for record in records if record.get("representation") == "raw_http"]
    structured_records = [record for record in records if "payload" in record]
    assert raw_records and all("body" not in record and record["body_omitted"] is True for record in raw_records if record["stage"] != "client_stream_start_raw")
    request = next(record for record in structured_records if record["stage"] == "client_request")
    assert request["payload"]["messages"] == [{"role": "user", "content": "[OMITTED]"}]
    assert any(record["stage"] == "client_response" and record["payload"]["choices"][0]["message"]["content"] == "[OMITTED]" for record in structured_records)


async def test_handled_error_response_is_traced_as_error(config_file, caplog) -> None:
    class FailingGateway(Gateway):
        async def complete(self, execution):
            from llm_proxy.domain.errors import ProviderUnavailableError
            raise ProviderUnavailableError("unavailable")

    class FailingGateways:
        def get(self, instance_name):
            return FailingGateway()

    caplog.set_level(logging.DEBUG, logger="llm-proxy.payload")
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=traced_app(config_file, FailingGateways())), base_url="http://proxy.test") as client:
        assert (await client.post("/v1/chat/completions", json={"model": "test-model", "messages": [{"role": "user", "content": "hello"}]})).status_code == 503
    assert terminal_outcomes(caplog) == ["error"]


async def test_stream_without_done_is_traced_as_incomplete(config_file, caplog) -> None:
    class IncompleteGateway(Gateway):
        def stream(self, execution):
            async def events():
                yield ResponseStarted("response", execution.upstream_model)
                yield TextDelta("text", "partial")
            return events()

    class IncompleteGateways:
        def get(self, instance_name):
            return IncompleteGateway()

    caplog.set_level(logging.DEBUG, logger="llm-proxy.payload")
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=traced_app(config_file, IncompleteGateways())), base_url="http://proxy.test") as client:
        response = await client.post("/v1/chat/completions", json={"model": "test-model", "stream": True, "messages": [{"role": "user", "content": "hello"}]})
    assert response.status_code == 200
    assert not response.text.endswith("data: [DONE]\n\n")
    assert terminal_outcomes(caplog) == ["incomplete"]
