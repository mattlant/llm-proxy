"""Header ownership rules for compatible wire relays."""
from __future__ import annotations

from llm_proxy.provider_extensions.wire import HeaderPairs

_HOP = {b"keep-alive", b"proxy-authenticate", b"proxy-authorization", b"te", b"trailer", b"transfer-encoding", b"upgrade"}


def _connection_names(headers: HeaderPairs) -> set[bytes]:
    return {part.strip().lower() for name, value in headers if name.lower() == b"connection" for part in value.split(b",") if part.strip()}


def _filter(headers: HeaderPairs, denied: set[bytes]) -> HeaderPairs:
    dynamic = _connection_names(headers)
    return tuple((name, value) for name, value in headers if name.lower() not in _HOP | dynamic | denied | {b"connection"})


def filter_request_headers(headers: HeaderPairs, defaults: HeaderPairs = ()) -> HeaderPairs:
    result = _filter(headers, {b"host", b"content-length", b"cookie", b"proxy-authorization"})
    existing = {name.lower() for name, _ in result}
    return result + tuple(item for item in defaults if item[0].lower() not in existing)


def filter_response_headers(headers: HeaderPairs) -> HeaderPairs:
    return _filter(headers, {b"content-length", b"access-control-allow-origin", b"access-control-allow-credentials", b"access-control-expose-headers", b"set-cookie", b"authorization", b"proxy-authorization"})
