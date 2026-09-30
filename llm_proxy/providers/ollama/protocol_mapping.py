from __future__ import annotations

from typing import Any

from llm_proxy.domain.errors import ProviderProtocolError
from llm_proxy.domain.responses import FinishReason


_FINISH_REASONS = {
    "stop": FinishReason.END_TURN,
    "length": FinishReason.MAX_TOKENS,
    "tool_calls": FinishReason.TOOL_USE,
    "content_filter": FinishReason.REFUSAL,
    "refusal": FinishReason.REFUSAL,
}


def map_finish_reason(value: Any) -> FinishReason:
    if value not in _FINISH_REASONS:
        raise ProviderProtocolError(f"unknown upstream finish reason: {value!r}")
    return _FINISH_REASONS[value]


def map_usage_count(value: Any, name: str, current: int | None = None) -> int | None:
    if value is None:
        return current
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ProviderProtocolError(f"upstream usage field {name} must be a non-negative integer")
    return value
