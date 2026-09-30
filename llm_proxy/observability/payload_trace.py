from __future__ import annotations

import base64
import contextvars
import hashlib
import json
import logging
import threading
import uuid
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Mapping, Sequence


_REDACTED = "[REDACTED]"
_OMITTED = "[OMITTED]"
_SENSITIVE_KEYS = frozenset({"authorization", "api_key", "apikey", "access_token", "refresh_token", "password", "secret", "cookie"})
_MESSAGE_ROLES = frozenset({"system", "developer", "user", "assistant", "tool"})
_THINKING_BLOCK_TYPES = frozenset({"thinking", "redacted_thinking"})
_MESSAGE_CONTENT_KEYS = frozenset({"content", "thinking", "reasoning", "reasoning_content"})
_context: contextvars.ContextVar[PayloadTraceContext | None] = contextvars.ContextVar("payload_trace_context", default=None)
_recorder: contextvars.ContextVar[PayloadTraceRecorder | None] = contextvars.ContextVar("payload_trace_recorder", default=None)


class TraceMode(StrEnum):
    DISABLED = "disabled"
    STRUCTURED = "structured"
    RAW = "raw"
    BOTH = "both"


@dataclass(frozen=True, slots=True)
class PayloadTraceContext:
    transaction_id: str
    mode: TraceMode
    include_content: bool


def redact_payload(value: Any, *, include_content: bool = True) -> Any:
    return _redact_payload(value, include_content=include_content)


def _redact_payload(value: Any, *, include_content: bool) -> Any:
    if isinstance(value, Mapping):
        is_message = value.get("role") in _MESSAGE_ROLES
        is_thinking_block = value.get("type") in _THINKING_BLOCK_TYPES
        return {
            str(key): (
                _REDACTED
                if str(key).casefold() in _SENSITIVE_KEYS
                else _OMITTED
                if not include_content and ((is_message and str(key) in _MESSAGE_CONTENT_KEYS) or (is_thinking_block and str(key) == "thinking"))
                else _redact_payload(item, include_content=include_content)
            )
            for key, item in value.items()
        }
    if isinstance(value, list | tuple):
        return [_redact_payload(item, include_content=include_content) for item in value]
    if value is None or isinstance(value, str | int | float | bool):
        return value
    return str(value)


class PayloadTraceRecorder:
    def __init__(self, logger: logging.Logger | None = None) -> None:
        self._logger = logger or logging.getLogger("llm-proxy.payload")
        self._sequences: dict[str, int] = {}
        self._sequence_lock = threading.Lock()

    def begin(self, mode: TraceMode, include_content: bool = True) -> PayloadTraceContext | None:
        if mode is TraceMode.DISABLED:
            return None
        context = PayloadTraceContext(str(uuid.uuid4()), mode, include_content)
        with self._sequence_lock:
            self._sequences[context.transaction_id] = 0
        return context

    def bind(self, context: PayloadTraceContext | None):
        if context is None:
            return None
        return _context.set(context), _recorder.set(self)

    @staticmethod
    def reset(tokens: tuple[contextvars.Token[PayloadTraceContext | None], contextvars.Token[PayloadTraceRecorder | None]] | None) -> None:
        if tokens is not None:
            _recorder.reset(tokens[1])
            _context.reset(tokens[0])

    def _emit(self, trace: PayloadTraceContext, record: dict[str, Any]) -> None:
        try:
            with self._sequence_lock:
                sequence = self._sequences.get(trace.transaction_id, 0) + 1
                self._sequences[trace.transaction_id] = sequence
            record.update(event="payload_trace", transaction_id=trace.transaction_id, sequence=sequence)
            self._logger.debug(json.dumps(record, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str))
        except Exception:
            pass

    def record(self, stage: str, payload: Any = None, *, interface: str = "openai", provider: str | None = None, stream: bool = False, context: PayloadTraceContext | None = None, outcome: str | None = None) -> None:
        trace = context or _context.get()
        if trace is None or trace.mode not in {TraceMode.STRUCTURED, TraceMode.BOTH}:
            return
        record: dict[str, Any] = {"stage": stage, "interface": interface, "provider": provider, "stream": stream, "payload": redact_payload(payload, include_content=trace.include_content)}
        if outcome is not None:
            record["outcome"] = outcome
        self._emit(trace, record)

    def record_raw_http(self, stage: str, body: bytes | None = None, *, interface: str = "openai", provider: str | None = None, stream: bool = False, method: str | None = None, url: str | None = None, path: str | None = None, query: bytes = b"", status_code: int | None = None, headers: Sequence[tuple[bytes, bytes]] = (), context: PayloadTraceContext | None = None) -> None:
        trace = context or _context.get()
        if trace is None or trace.mode not in {TraceMode.RAW, TraceMode.BOTH}:
            return
        record: dict[str, Any] = {"stage": stage, "interface": interface, "provider": provider, "stream": stream, "representation": "raw_http"}
        if method is not None or url is not None or path is not None or status_code is not None or headers:
            record["http"] = {"method": method, "url": url, "path": path, "query": self._encode_bytes(query), "status_code": status_code, "headers": [{"name": self._encode_bytes(name), "value": self._encode_bytes(value)} for name, value in headers]}
        if body is not None:
            if trace.include_content:
                record["body"] = self._encode_bytes(body)
            else:
                record["body_omitted"] = True
                record["body_byte_length"] = len(body)
                record["body_sha256"] = hashlib.sha256(body).hexdigest()
        self._emit(trace, record)

    @staticmethod
    def _encode_bytes(value: bytes) -> dict[str, Any]:
        try:
            data = value.decode("utf-8")
            encoding = "utf-8"
        except UnicodeDecodeError:
            data = base64.b64encode(value).decode("ascii")
            encoding = "base64"
        return {"encoding": encoding, "data": data, "byte_length": len(value), "sha256": hashlib.sha256(value).hexdigest()}

    def end(self, outcome: str, *, context: PayloadTraceContext | None = None) -> None:
        trace = context or _context.get()
        if trace is not None:
            self._emit(trace, {"stage": "transaction_end", "interface": "openai", "provider": None, "stream": False, "outcome": outcome})


def current_recorder() -> PayloadTraceRecorder | None:
    return _recorder.get()


def current_trace_context() -> PayloadTraceContext | None:
    return _context.get()
