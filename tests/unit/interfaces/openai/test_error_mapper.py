from __future__ import annotations

from llm_proxy.application.errors import ModelNotFoundError
from llm_proxy.domain.errors import DeveloperRoleCompatibilityError, ProviderProtocolError, ProviderUnavailableError
from llm_proxy.interfaces.openai.error_mapper import OpenAIErrorMapper


def test_maps_model_and_provider_errors_without_provider_details() -> None:
    mapper = OpenAIErrorMapper()

    status, payload = mapper.map_error(ModelNotFoundError("openai", "missing"))
    assert status == 404
    assert payload["error"]["type"] == "invalid_request_error"

    status, payload = mapper.map_error(ProviderProtocolError("/internal/path", details={"body": "secret"}))
    assert status == 502
    assert payload["error"]["message"] == "upstream provider request failed"

    status, payload = mapper.map_error(ProviderUnavailableError("connection refused"))
    assert status == 503
    assert payload["error"]["message"] == "upstream provider request failed"


def test_maps_developer_role_compatibility_error_as_invalid_request() -> None:
    status, payload = OpenAIErrorMapper().map_error(DeveloperRoleCompatibilityError())

    assert status == 400
    assert payload["error"]["type"] == "invalid_request_error"
    assert payload["error"]["message"] == "selected execution does not support developer messages"
