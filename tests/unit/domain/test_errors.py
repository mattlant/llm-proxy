from __future__ import annotations

import pytest

from llm_proxy.domain.errors import (
    CompletionCancelledError,
    CompletionError,
    ContextLimitExceededError,
    InvalidCompletionRequest,
    InvalidToolArgumentsError,
    ProviderProtocolError,
    ProviderUnavailableError,
    UnknownModelError,
)


@pytest.mark.parametrize(
    "error_type",
    [
        InvalidCompletionRequest,
        UnknownModelError,
        InvalidToolArgumentsError,
        ProviderUnavailableError,
        ProviderProtocolError,
        ContextLimitExceededError,
        CompletionCancelledError,
    ],
)
def test_domain_errors_share_completion_error_contract(error_type):
    error = error_type(
        "failure",
        tool_call_id="call-1",
        model="model",
        provider="provider",
        details={"attempt": 1},
    )

    assert isinstance(error, CompletionError)
    assert str(error) == "failure"
    assert error.tool_call_id == "call-1"
    assert error.model == "model"
    assert error.provider == "provider"
    assert error.details == {"attempt": 1}
