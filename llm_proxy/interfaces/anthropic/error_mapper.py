from __future__ import annotations

from typing import Any

from llm_proxy.application.errors import CapabilityExecutionError, CapabilityTimeoutError, InterfaceDisabledError, ModelNotExposedError, ModelNotFoundError
from llm_proxy.domain.errors import ContextLimitExceededError, InvalidCompletionRequest, InvalidToolArgumentsError, ProviderProtocolError, ProviderUnavailableError
from llm_proxy.application.failure_classification import FailureCategory, FailureClassifier

from .authentication import AnthropicAuthenticationError


class AnthropicErrorMapper:
    def map_error(self, error: Exception) -> tuple[int, dict[str, Any]]:
        status, kind, message = self._details(error)
        return status, {"type": "error", "error": {"type": kind, "message": message}}

    @staticmethod
    def _details(error: Exception) -> tuple[int, str, str]:
        if isinstance(error, AnthropicAuthenticationError):
            return 401, "authentication_error", "Invalid authentication credentials"
        category = FailureClassifier.classify(error).category
        if category is FailureCategory.INVALID_REQUEST:
            return 400, "invalid_request_error", str(error) or "Invalid request"
        if category is FailureCategory.MODEL_NOT_FOUND:
            return 404, "not_found_error", "Model not found"
        if category is FailureCategory.CONTEXT_LIMIT:
            return 413, "request_too_large", "Request exceeds the model context limit"
        if category is FailureCategory.PROVIDER_UNAVAILABLE:
            return 503, "overloaded_error", "Upstream provider is unavailable"
        if category is FailureCategory.CAPABILITY_FAILED: return 503, "overloaded_error", "Extension capability failed"
        if category is FailureCategory.CAPABILITY_TIMEOUT: return 504, "api_error", "Extension capability timed out"
        if category is FailureCategory.PROVIDER_PROTOCOL:
            return 502, "api_error", "Upstream provider request failed"
        if isinstance(error, InvalidToolArgumentsError):
            return 500, "api_error", "Upstream provider returned invalid tool arguments"
        return 500, "api_error", "Internal server error"
