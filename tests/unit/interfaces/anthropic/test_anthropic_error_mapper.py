from __future__ import annotations

import pytest

from llm_proxy.application.errors import ModelNotExposedError, ModelNotFoundError
from llm_proxy.domain.errors import ContextLimitExceededError, InvalidCompletionRequest, ProviderProtocolError, ProviderUnavailableError
from llm_proxy.interfaces.anthropic.authentication import AnthropicAuthenticationError
from llm_proxy.interfaces.anthropic.error_mapper import AnthropicErrorMapper


@pytest.mark.parametrize(("error", "status", "kind", "message"), [
    (InvalidCompletionRequest("bad request"), 400, "invalid_request_error", "bad request"),
    (AnthropicAuthenticationError("secret"), 401, "authentication_error", "Invalid authentication credentials"),
    (ModelNotFoundError("anthropic", "missing"), 404, "not_found_error", "Model not found"),
    (ModelNotExposedError("anthropic", "hidden"), 404, "not_found_error", "Model not found"),
    (ContextLimitExceededError("limit"), 413, "request_too_large", "Request exceeds the model context limit"),
    (ProviderUnavailableError("http://private"), 503, "overloaded_error", "Upstream provider is unavailable"),
    (ProviderProtocolError("/internal/path"), 502, "api_error", "Upstream provider request failed"),
])
def test_maps_errors_without_leaking_internal_details(error, status, kind, message) -> None:
    actual_status, body = AnthropicErrorMapper().map_error(error)

    assert actual_status == status
    assert body == {"type": "error", "error": {"type": kind, "message": message}}
