"""An external package which relies only on the supported public SDK."""
from __future__ import annotations

import asyncio

from llm_proxy.provider_extensions import (
    CompletionExecution, CompletionGateway, CompletionResponse, FinishReason, Message,
    MessageRole, ProviderCapabilities, ProviderExtensionMetadata, ProviderFactory,
    ProviderHealth, ProviderHealthStatus, ProviderInstanceConfig, ResponseCompleted,
    ResponseStarted, SdkCompatibility, TextCompleted, TextContent, TextDelta, TextStarted,
    Usage, ProviderUnavailableError,
    ManagementCommandDescriptor, ManagementCommandMutability, ManagementCommandPermission,
)


class DeterministicGateway(CompletionGateway):
    management_commands = (
        ManagementCommandDescriptor("get_state", "Get instance-local state", {"type": "object", "additionalProperties": False}, {"type": "object", "required": ["value"], "properties": {"value": {"type": "integer"}}, "additionalProperties": False}, ManagementCommandMutability.READ_ONLY, ManagementCommandPermission.INSPECT),
        ManagementCommandDescriptor("set_state", "Set instance-local state", {"type": "object", "required": ["value"], "properties": {"value": {"type": "integer"}}, "additionalProperties": False}, {"type": "object", "required": ["value"], "properties": {"value": {"type": "integer"}}, "additionalProperties": False}, ManagementCommandMutability.STATE_CHANGING, ManagementCommandPermission.OPERATE),
        ManagementCommandDescriptor("reset_state", "Reset instance-local state", {"type": "object", "additionalProperties": False}, {"type": "object", "required": ["value"], "properties": {"value": {"type": "integer"}}, "additionalProperties": False}, ManagementCommandMutability.STATE_CHANGING, ManagementCommandPermission.OPERATE),
    )

    def __init__(self, *, fail_completion: bool = False, await_cancellation: bool = False, initial_state: int = 0) -> None:
        self._fail_completion = fail_completion
        self._await_cancellation = await_cancellation
        self._state = initial_state

    async def execute_management_command(self, command_name, payload, context):
        if command_name == "get_state":
            return {"value": self._state}
        if command_name == "set_state":
            self._state = payload["value"]
        elif command_name == "reset_state":
            self._state = 0
        return {"value": self._state}

    async def complete(self, execution: CompletionExecution) -> CompletionResponse:
        if self._fail_completion:
            raise ProviderUnavailableError("deterministic provider unavailable", provider="fixture-instance")
        return CompletionResponse("external-response-1", execution.response_model, Message(MessageRole.ASSISTANT, (TextContent("external response"),)), FinishReason.END_TURN, None, Usage(1, 2))

    async def stream(self, execution: CompletionExecution):
        if self._await_cancellation:
            await asyncio.sleep(3600)
        yield ResponseStarted("external-response-1", execution.response_model)
        yield TextStarted("text-1")
        yield TextDelta("text-1", "external stream")
        yield TextCompleted("text-1")
        yield ResponseCompleted(FinishReason.END_TURN, None, Usage(1, 2))


class DeterministicFactory(ProviderFactory):
    def create(self, config: ProviderInstanceConfig) -> CompletionGateway:
        if config.config.get("fail_construction"):
            raise ValueError("deterministic construction failure")
        return DeterministicGateway(
            fail_completion=bool(config.config.get("fail_completion")),
            await_cancellation=bool(config.config.get("await_cancellation")),
            initial_state=int(config.config.get("initial_state", 0)),
        )


class DeterministicExtension:
    metadata = ProviderExtensionMetadata("example.deterministic", "Deterministic external provider", "example-deterministic-provider", "0.1.0")
    compatibility = SdkCompatibility(minimum="0.1.0", maximum="0.3.999")
    capabilities = ProviderCapabilities(streaming=True, usage=True, health=True, management_commands=True)
    factory = DeterministicFactory()

    async def health(self, config: ProviderInstanceConfig) -> ProviderHealth:
        return ProviderHealth(ProviderHealthStatus.HEALTHY)
