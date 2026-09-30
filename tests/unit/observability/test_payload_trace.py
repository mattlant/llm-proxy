from __future__ import annotations

import base64
import hashlib
import json
import logging

from llm_proxy.observability.payload_trace import PayloadTraceRecorder, TraceMode, redact_payload


def test_redaction_is_recursive_and_preserves_payload_content() -> None:
    assert redact_payload({"access_token": "secret", "nested": [{"api_key": "also-secret", "reasoning": "visible"}]}) == {"access_token": "[REDACTED]", "nested": [{"api_key": "[REDACTED]", "reasoning": "visible"}]}


def test_redaction_preserves_non_credential_token_fields() -> None:
    assert redact_payload({"prompt_tokens": 4, "completion_tokens": 6, "total_tokens": 10, "max_tokens": 12, "token": "not-a-recognized-credential-key"}) == {"prompt_tokens": 4, "completion_tokens": 6, "total_tokens": 10, "max_tokens": 12, "token": "not-a-recognized-credential-key"}


def test_content_omission_is_scoped_to_message_and_thinking_semantics() -> None:
    payload = {
        "messages": [
            {"role": "user", "content": "secret prompt", "metadata": {"content": "diagnostic"}},
            {"role": "assistant", "thinking": "secret thinking", "reasoning": "secret reasoning", "reasoning_content": "secret reasoning content", "tool_calls": [{"function": {"name": "lookup"}}]},
            {"role": "tool", "content": {"result": "secret result"}},
        ],
        "choices": [{"delta": {"content": "secret delta"}}],
        "thinking_block": {"type": "thinking", "thinking": "secret block", "id": "diagnostic"},
        "diagnostic": {"content": "preserved", "thinking": "preserved", "reasoning": "preserved", "reasoning_content": "preserved"},
        "marker": "[DONE]",
    }
    omitted = redact_payload(payload, include_content=False)
    assert omitted["messages"] == [
        {"role": "user", "content": "[OMITTED]", "metadata": {"content": "diagnostic"}},
        {"role": "assistant", "thinking": "[OMITTED]", "reasoning": "[OMITTED]", "reasoning_content": "[OMITTED]", "tool_calls": [{"function": {"name": "lookup"}}]},
        {"role": "tool", "content": "[OMITTED]"},
    ]
    assert omitted["choices"][0]["delta"]["content"] == "secret delta"
    assert omitted["thinking_block"] == {"type": "thinking", "thinking": "[OMITTED]", "id": "diagnostic"}
    assert omitted["diagnostic"]["content"] == "preserved"
    assert omitted["marker"] == "[DONE]"


def test_content_omission_keeps_raw_metadata_without_body(caplog) -> None:
    caplog.set_level(logging.DEBUG, logger="llm-proxy.payload")
    recorder = PayloadTraceRecorder()
    context = recorder.begin(TraceMode.RAW, include_content=False)
    body = b'{"messages":[{"role":"user","content":"secret"}]}'
    recorder.record_raw_http("raw", body, context=context, method="POST", path="/v1/chat/completions")
    record = json.loads(caplog.records[0].message)
    assert "body" not in record
    assert record["body_omitted"] is True
    assert record["body_byte_length"] == len(body)
    assert record["body_sha256"] == hashlib.sha256(body).hexdigest()
    assert record["http"]["method"] == "POST"


def test_disabled_recorder_is_a_noop(caplog) -> None:
    recorder = PayloadTraceRecorder()
    assert recorder.begin(TraceMode.DISABLED) is None
    recorder.record("client_request", {"content": "ignored"})
    assert not caplog.records


def test_records_correlated_sequence_and_safe_values(caplog) -> None:
    caplog.set_level(logging.DEBUG, logger="llm-proxy.payload")
    recorder = PayloadTraceRecorder()
    context = recorder.begin(TraceMode.BOTH)
    recorder.record("client_request", {"authorization": "never"}, context=context)
    recorder.record("client_response", object(), context=context)
    recorder.end("success", context=context)
    records = [json.loads(record.message) for record in caplog.records]
    assert [record["sequence"] for record in records] == [1, 2, 3]
    assert {record["transaction_id"] for record in records} == {context.transaction_id}
    assert records[0]["payload"]["authorization"] == "[REDACTED]"
    assert records[-1]["outcome"] == "success"


def test_raw_records_preserve_bytes_headers_and_one_sequence(caplog) -> None:
    caplog.set_level(logging.DEBUG, logger="llm-proxy.payload")
    recorder = PayloadTraceRecorder()
    context = recorder.begin(TraceMode.BOTH)
    body = b'\xff {"authorization":"visible"}'
    recorder.record_raw_http("client_request_raw", body, context=context, method="POST", url="http://proxy.test/v1/chat/completions?a=1", path="/v1/chat/completions", query=b"a=1", headers=((b"X-Case", b"one"), (b"X-Case", b"two")))
    recorder.record("client_request", {"authorization": "hidden"}, context=context)
    recorder.end("success", context=context)
    records = [json.loads(record.message) for record in caplog.records]
    assert [record["sequence"] for record in records] == [1, 2, 3]
    raw = records[0]
    assert base64.b64decode(raw["body"]["data"]) == body
    assert raw["body"]["encoding"] == "base64"
    assert raw["body"]["byte_length"] == len(body)
    assert raw["body"]["sha256"] == hashlib.sha256(body).hexdigest()
    assert raw["http"]["query"] == {"encoding": "utf-8", "data": "a=1", "byte_length": 3, "sha256": hashlib.sha256(b"a=1").hexdigest()}
    assert [item["value"]["data"] for item in raw["http"]["headers"]] == ["one", "two"]
    assert records[1]["payload"]["authorization"] == "[REDACTED]"


def test_raw_records_use_readable_utf8_and_lossless_binary_fallback(caplog) -> None:
    caplog.set_level(logging.DEBUG, logger="llm-proxy.payload")
    recorder = PayloadTraceRecorder()
    context = recorder.begin(TraceMode.RAW)
    recorder.record_raw_http("raw", "json: ✅".encode(), context=context, headers=((b"X-Text", "héllo".encode()), (b"X-Binary", b"\xff")))
    record = json.loads(caplog.records[0].message)
    assert record["body"] == {"encoding": "utf-8", "data": "json: ✅", "byte_length": 9, "sha256": hashlib.sha256("json: ✅".encode()).hexdigest()}
    assert record["http"]["headers"][0]["value"]["data"] == "héllo"
    binary = record["http"]["headers"][1]["value"]
    assert binary["encoding"] == "base64"
    assert base64.b64decode(binary["data"]) == b"\xff"


def test_modes_gate_structured_and_raw_records(caplog) -> None:
    caplog.set_level(logging.DEBUG, logger="llm-proxy.payload")
    for mode, expected in ((TraceMode.STRUCTURED, ["structured"]), (TraceMode.RAW, ["raw"]), (TraceMode.BOTH, ["structured", "raw"])):
        caplog.clear()
        recorder = PayloadTraceRecorder()
        context = recorder.begin(mode)
        recorder.record("structured", {}, context=context)
        recorder.record_raw_http("raw", b"x", context=context)
        assert [json.loads(record.message)["stage"] for record in caplog.records] == expected
