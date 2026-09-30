from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from llm_proxy.application.errors import CapabilityExecutionError, CapabilityTimeoutError, InterfaceDisabledError, ModelNotExposedError, ModelNotFoundError
from llm_proxy.domain.errors import ContextLimitExceededError, DeveloperRoleCompatibilityError, InvalidCompletionRequest, ProviderProtocolError, ProviderUnavailableError


class FailureCategory(StrEnum):
    INVALID_REQUEST = "invalid_request"
    MODEL_NOT_FOUND = "model_not_found"
    CONTEXT_LIMIT = "context_limit"
    PROVIDER_UNAVAILABLE = "provider_unavailable"
    PROVIDER_PROTOCOL = "provider_protocol"
    CAPABILITY_FAILED = "capability_failed"
    CAPABILITY_TIMEOUT = "capability_timeout"
    INTERNAL = "internal"


@dataclass(frozen=True, slots=True)
class FailureClassification:
    category: FailureCategory


class FailureClassifier:
    @staticmethod
    def classify(error: Exception) -> FailureClassification:
        if isinstance(error, (DeveloperRoleCompatibilityError, InvalidCompletionRequest, ValueError)):
            return FailureClassification(FailureCategory.INVALID_REQUEST)
        if isinstance(error, (ModelNotFoundError, ModelNotExposedError, InterfaceDisabledError)):
            return FailureClassification(FailureCategory.MODEL_NOT_FOUND)
        if isinstance(error, ContextLimitExceededError):
            return FailureClassification(FailureCategory.CONTEXT_LIMIT)
        if isinstance(error, ProviderUnavailableError):
            return FailureClassification(FailureCategory.PROVIDER_UNAVAILABLE)
        if isinstance(error, ProviderProtocolError):
            return FailureClassification(FailureCategory.PROVIDER_PROTOCOL)
        if isinstance(error, CapabilityExecutionError):
            return FailureClassification(FailureCategory.CAPABILITY_FAILED)
        if isinstance(error, CapabilityTimeoutError):
            return FailureClassification(FailureCategory.CAPABILITY_TIMEOUT)
        return FailureClassification(FailureCategory.INTERNAL)
