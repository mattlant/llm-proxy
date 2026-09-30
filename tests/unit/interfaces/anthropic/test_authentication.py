from __future__ import annotations

import pytest

from llm_proxy.configuration.models import AuthenticationConfig
from llm_proxy.interfaces.anthropic.authentication import AnthropicAuthenticationError, AnthropicAuthenticator
from llm_proxy.interfaces.anthropic.headers import extract_anthropic_headers


def headers(token: str | None):
    result = {"content-type": "application/json", "anthropic-version": "2023-06-01"}
    if token is not None:
        result["x-api-key"] = token
    return extract_anthropic_headers(result)


def test_authenticator_supports_any_and_fixed_tokens() -> None:
    authenticator = AnthropicAuthenticator()
    authenticator.authenticate(headers("any-token"), AuthenticationConfig(True, None))
    authenticator.authenticate(headers("secret"), AuthenticationConfig(False, "secret"))

    with pytest.raises(AnthropicAuthenticationError, match="Invalid authentication credentials"):
        authenticator.authenticate(headers(None), AuthenticationConfig(True, None))
    with pytest.raises(AnthropicAuthenticationError, match="Invalid authentication credentials"):
        authenticator.authenticate(headers("wrong"), AuthenticationConfig(False, "secret"))


def test_authenticator_accepts_bearer_tokens() -> None:
    value = extract_anthropic_headers({"content-type": "application/json", "anthropic-version": "2023-06-01", "authorization": "Bearer secret"})

    AnthropicAuthenticator().authenticate(value, AuthenticationConfig(False, "secret"))
