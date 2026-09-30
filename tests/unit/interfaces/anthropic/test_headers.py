from __future__ import annotations

import pytest

from llm_proxy.domain.errors import InvalidCompletionRequest
from llm_proxy.interfaces.anthropic.headers import extract_anthropic_headers


def test_extracts_version_credentials_and_normalized_beta_values() -> None:
    headers = extract_anthropic_headers({
        "Content-Type": "application/json; charset=utf-8",
        "anthropic-version": "2023-06-01",
        "anthropic-beta": "tools-2025-01-01, tools-2025-01-01, future-feature",
        "x-api-key": "token",
        "Authorization": "Bearer token",
    })

    assert headers.version == "2023-06-01"
    assert headers.beta == ("tools-2025-01-01", "future-feature")
    assert headers.token == "token"


def test_accepts_bearer_and_rejects_conflicting_credentials() -> None:
    bearer = extract_anthropic_headers({"content-type": "application/json", "anthropic-version": "2099-12-31", "authorization": "Bearer token"})
    assert bearer.token == "token"

    conflicting = extract_anthropic_headers({"content-type": "application/json", "anthropic-version": "2023-06-01", "x-api-key": "one", "authorization": "Bearer two"})
    with pytest.raises(InvalidCompletionRequest, match="conflicting"):
        _ = conflicting.token


@pytest.mark.parametrize("headers", [
    {"content-type": "application/json"},
    {"content-type": "application/json", "anthropic-version": ""},
    {"content-type": "text/plain", "anthropic-version": "2023-06-01"},
])
def test_rejects_missing_invalid_version_or_non_json_content_type(headers) -> None:
    with pytest.raises(InvalidCompletionRequest):
        extract_anthropic_headers(headers)
