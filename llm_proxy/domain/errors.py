from __future__ import annotations

from typing import Any, Mapping


class CompletionError(Exception):
    def __init__(
        self,
        message: str = "",
        *,
        tool_call_id: str | None = None,
        model: str | None = None,
        provider: str | None = None,
        details: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.tool_call_id = tool_call_id
        self.model = model
        self.provider = provider
        self.details = dict(details or {})


class InvalidCompletionRequest(CompletionError):
    pass


class DeveloperRoleCompatibilityError(InvalidCompletionRequest):
    def __init__(self) -> None:
        super().__init__("selected execution does not support developer messages")


class UnknownModelError(CompletionError):
    pass


class InvalidToolArgumentsError(CompletionError):
    pass


class ProviderUnavailableError(CompletionError):
    pass


class ProviderProtocolError(CompletionError):
    pass


class ContextLimitExceededError(CompletionError):
    pass


class CompletionCancelledError(CompletionError):
    pass
