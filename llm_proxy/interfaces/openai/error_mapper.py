from __future__ import annotations

from typing import Any

from llm_proxy.application.errors import CapabilityExecutionError, CapabilityTimeoutError, InterfaceDisabledError, ModelNotExposedError, ModelNotFoundError
from llm_proxy.domain.errors import ContextLimitExceededError, DeveloperRoleCompatibilityError, InvalidCompletionRequest, ProviderProtocolError, ProviderUnavailableError
from llm_proxy.application.failure_classification import FailureCategory, FailureClassifier


class OpenAIErrorMapper:
    def map_error(self, error: Exception) -> tuple[int, dict[str, Any]]:
        status, kind = self._status_and_type(error)
        code = "extension_capability_timeout" if isinstance(error, CapabilityTimeoutError) else "extension_capability_failed" if isinstance(error, CapabilityExecutionError) else None
        return status, {"error": {"message": self._message(error), "type": kind, "param": None, "code": code}}

    @staticmethod
    def _status_and_type(error: Exception) -> tuple[int, str]:
        category = FailureClassifier.classify(error).category
        status = {FailureCategory.INVALID_REQUEST: 400, FailureCategory.MODEL_NOT_FOUND: 404,
                  FailureCategory.CONTEXT_LIMIT: 413, FailureCategory.PROVIDER_UNAVAILABLE: 503,
                  FailureCategory.CAPABILITY_FAILED: 503, FailureCategory.CAPABILITY_TIMEOUT: 504,
                  FailureCategory.PROVIDER_PROTOCOL: 502}.get(category, 500)
        return status, "invalid_request_error" if category in {FailureCategory.INVALID_REQUEST, FailureCategory.MODEL_NOT_FOUND, FailureCategory.CONTEXT_LIMIT} else "api_error"

    @staticmethod
    def _message(error: Exception) -> str:
        if isinstance(error, (ProviderUnavailableError, ProviderProtocolError)):
            return "upstream provider request failed"
        if isinstance(error, CapabilityTimeoutError): return "extension capability timed out"
        if isinstance(error, CapabilityExecutionError): return "extension capability failed"
        return str(error) or "request failed"
