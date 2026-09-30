"""Public canonical execution values for provider extensions."""
from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import dataclass
from enum import Enum
from typing import Protocol

from llm_proxy.domain.events import CompletionEvent
from llm_proxy.domain.messages import DeveloperRoleMode
from llm_proxy.domain.requests import CompletionRequest, ContextManagementResult, SamplingParameters
from llm_proxy.domain.responses import CompletionResponse


try:
    from enum import StrEnum
except ImportError:  # pragma: no cover
    class StrEnum(str, Enum):
        pass


class SourceInterface(StrEnum):
    """Stable inbound protocol identity, without exposing adapter internals."""

    OPENAI = "openai"
    OLLAMA = "ollama"
    ANTHROPIC = "anthropic"


@dataclass(frozen=True, slots=True)
class ProviderExecutionPolicy:
    """Immutable model-specific mapping policy safe to expose to providers.

    It deliberately excludes model profiles, provider configuration, aliases,
    resolver state, and inbound-interface configuration.
    """

    structured_output_mode: str = "unsupported"
    expose_thinking: bool = False
    native_tools: bool = True
    developer_role_mode: DeveloperRoleMode = DeveloperRoleMode.PRESERVE

    def __post_init__(self) -> None:
        if self.structured_output_mode not in {"unsupported", "openai_json_schema", "ollama_format"}:
            raise ValueError("structured_output_mode is unsupported")
        if not isinstance(self.expose_thinking, bool) or not isinstance(self.native_tools, bool):
            raise TypeError("provider execution policy flags must be booleans")
        if not isinstance(self.developer_role_mode, DeveloperRoleMode):
            raise TypeError("developer_role_mode must be a DeveloperRoleMode")


@dataclass(frozen=True, slots=True)
class CompletionExecution:
    request: CompletionRequest
    upstream_model: str
    provider_instance_name: str
    parameters: SamplingParameters
    response_model: str
    context_management: ContextManagementResult | None = None
    policy: ProviderExecutionPolicy = ProviderExecutionPolicy()
    source_interface: SourceInterface = SourceInterface.OPENAI

    def __post_init__(self) -> None:
        if not isinstance(self.request, CompletionRequest):
            raise TypeError("request must be a CompletionRequest")
        for name in ("upstream_model", "provider_instance_name", "response_model"):
            if not isinstance(getattr(self, name), str) or not getattr(self, name).strip():
                raise ValueError(f"{name} must be a non-blank string")
        if not isinstance(self.parameters, SamplingParameters):
            raise TypeError("parameters must be SamplingParameters")
        if self.context_management is not None and not isinstance(self.context_management, ContextManagementResult):
            raise TypeError("context_management must be a ContextManagementResult or null")
        if not isinstance(self.policy, ProviderExecutionPolicy):
            raise TypeError("policy must be a ProviderExecutionPolicy")
        if not isinstance(self.source_interface, SourceInterface):
            raise TypeError("source_interface must be a SourceInterface")


class CompletionGateway(Protocol):
    async def complete(self, execution: CompletionExecution) -> CompletionResponse: ...
    def stream(self, execution: CompletionExecution) -> AsyncIterator[CompletionEvent]: ...
