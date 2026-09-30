from __future__ import annotations

import hmac

from llm_proxy.configuration.models import AuthenticationConfig

from .headers import AnthropicHeaders


class AnthropicAuthenticationError(Exception):
    """Authentication failed without retaining credential material."""


class AnthropicAuthenticator:
    def authenticate(self, headers: AnthropicHeaders, config: AuthenticationConfig) -> None:
        token = headers.token
        if token is None:
            raise AnthropicAuthenticationError("Invalid authentication credentials")
        if config.allow_any_token:
            return
        if config.token is None or not hmac.compare_digest(token, config.token):
            raise AnthropicAuthenticationError("Invalid authentication credentials")
