from llm_proxy.infrastructure.wire_headers import filter_request_headers, filter_response_headers


def test_request_header_filter_preserves_authorization_and_ordered_duplicates() -> None:
    headers = ((b"X-Trace", b"one"), (b"Authorization", b"Bearer client"), (b"X-Trace", b"two"))
    assert filter_request_headers(headers) == headers


def test_request_header_filter_removes_connection_nominated_headers() -> None:
    headers = ((b"Connection", b"x-private"), (b"X-Private", b"no"), (b"Authorization", b"Bearer client"))
    assert filter_request_headers(headers) == ((b"Authorization", b"Bearer client"),)


def test_request_header_filter_keeps_proxy_authorization_filtered() -> None:
    headers = ((b"Authorization", b"Bearer client"), (b"Proxy-Authorization", b"Basic secret"))
    assert filter_request_headers(headers) == ((b"Authorization", b"Bearer client"),)


def test_request_header_filter_preserves_defaults_without_duplicates() -> None:
    headers = ((b"Authorization", b"Bearer client"), (b"content-type", b"application/json"))
    defaults = ((b"Content-Type", b"ignored"), (b"Accept", b"application/json"))
    assert filter_request_headers(headers, defaults) == (headers[0], headers[1], defaults[1])


def test_response_header_filter_removes_cors_and_cookies() -> None:
    headers = ((b"x-request-id", b"abc"), (b"set-cookie", b"secret"), (b"access-control-allow-origin", b"*"))
    assert filter_response_headers(headers) == ((b"x-request-id", b"abc"),)
