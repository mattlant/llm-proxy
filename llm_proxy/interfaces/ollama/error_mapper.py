from __future__ import annotations

from typing import Any

from llm_proxy.application.errors import CapabilityExecutionError, CapabilityTimeoutError, InterfaceDisabledError, ModelNotExposedError, ModelNotFoundError
from llm_proxy.domain.errors import ContextLimitExceededError, DeveloperRoleCompatibilityError, InvalidCompletionRequest, ProviderProtocolError, ProviderUnavailableError
from llm_proxy.application.failure_classification import FailureCategory, FailureClassifier


class OllamaErrorMapper:
    def map_error(self, error: Exception) -> tuple[int, dict[str, Any]]:
        category = FailureClassifier.classify(error).category
        if isinstance(error, DeveloperRoleCompatibilityError):
            return 400, {"error": str(error)}
        if category is FailureCategory.INVALID_REQUEST:
            return 400, {"error": str(error) or "invalid request"}
        if category is FailureCategory.MODEL_NOT_FOUND:
            return 404, {"error": "model not found"}
        if category is FailureCategory.CONTEXT_LIMIT:
            return 413, {"error": "context limit exceeded"}
        if category in {FailureCategory.PROVIDER_UNAVAILABLE, FailureCategory.PROVIDER_PROTOCOL}:
            return (503 if category is FailureCategory.PROVIDER_UNAVAILABLE else 502), {"error": "upstream provider request failed"}
        if category is FailureCategory.CAPABILITY_FAILED: return 503, {"error": "extension capability failed"}
        if category is FailureCategory.CAPABILITY_TIMEOUT: return 504, {"error": "extension capability timed out"}
        return 500, {"error": "request failed"}
