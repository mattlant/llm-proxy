from __future__ import annotations

import pytest

from llm_proxy.providers.ollama.stream_parser import SseDecoder, SseRecord


def test_decodes_single_record() -> None:
    decoder = SseDecoder()

    assert decoder.feed(b"data: {\"delta\":\"hello\"}\n\n") == (
        SseRecord(None, '{"delta":"hello"}'),
    )


def test_decodes_record_fragmented_across_chunks() -> None:
    decoder = SseDecoder()

    assert decoder.feed(b"data: hello") == ()
    assert decoder.feed(b" world\n") == ()
    assert decoder.feed(b"\n") == (SseRecord(None, "hello world"),)


def test_decodes_multiple_records_in_one_chunk() -> None:
    decoder = SseDecoder()

    assert decoder.feed(b"data: one\n\ndata: two\n\n") == (
        SseRecord(None, "one"),
        SseRecord(None, "two"),
    )


def test_handles_utf8_code_point_split_across_chunks() -> None:
    decoder = SseDecoder()
    encoded = "data: café\n\n".encode()

    assert decoder.feed(encoded[:-3]) == ()
    assert decoder.feed(encoded[-3:]) == (SseRecord(None, "café"),)


def test_joins_multiline_data_and_preserves_event() -> None:
    decoder = SseDecoder()

    assert decoder.feed(b"event: message\ndata: first\ndata: second\n\n") == (
        SseRecord("message", "first\nsecond"),
    )


def test_ignores_comments_and_ping_lines() -> None:
    decoder = SseDecoder()

    assert decoder.feed(b": keep-alive\n\ndata: payload\n\n") == (
        SseRecord(None, "payload"),
    )


def test_handles_crlf_records() -> None:
    decoder = SseDecoder()

    assert decoder.feed(b"event: message\r\ndata: payload\r\n\r\n") == (
        SseRecord("message", "payload"),
    )


def test_finish_emits_final_unterminated_data() -> None:
    decoder = SseDecoder()

    assert decoder.feed(b"data: final") == ()
    assert decoder.finish() == (SseRecord(None, "final"),)


def test_done_record_and_invalid_utf8() -> None:
    decoder = SseDecoder()
    assert decoder.feed(b"data: [DONE]\n\n") == (SseRecord(None, "[DONE]"),)

    with pytest.raises(UnicodeDecodeError):
        SseDecoder().feed(b"data: \xff\n\n")
