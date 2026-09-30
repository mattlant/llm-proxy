"""Single application-owned authority for canonical semantic support."""
from __future__ import annotations

from collections.abc import Mapping
from typing import Protocol, runtime_checkable

from llm_proxy.configuration.models import InterfaceName, StructuredOutputMode
from llm_proxy.domain.errors import InvalidCompletionRequest
from llm_proxy.domain.events import ResponseCompleted
from llm_proxy.domain.requests import CompletionRequest
from llm_proxy.domain.responses import CompletionResponse, FinishReason
from llm_proxy.domain.semantic_support import (
    EffectiveSemanticHandling,
    SemanticHandlingOutcome,
    SemanticIdentity,
    SemanticResolutionScope,
    SemanticSupportPhase,
)
from llm_proxy.provider_extensions import ProviderSemanticDisposition

from .provider_registry import ProviderExtensionOrigin, ProviderRuntimeSupport


@runtime_checkable
class TargetSemanticProjection(Protocol):
    def finish_reason_outcome(self, reason: FinishReason) -> SemanticHandlingOutcome:
        ...


class _DefaultProjection:
    def __init__(self, lossy: frozenset[FinishReason]) -> None:
        self._lossy = lossy

    def finish_reason_outcome(self, reason: FinishReason) -> SemanticHandlingOutcome:
        return SemanticHandlingOutcome.LOSSY if reason in self._lossy else SemanticHandlingOutcome.SUPPORTED


class SemanticProjectionRegistry:
    def __init__(self, projections: Mapping[InterfaceName, TargetSemanticProjection] | None = None) -> None:
        if projections is None:
            projections = {
                InterfaceName.OPENAI: _DefaultProjection(frozenset({FinishReason.PAUSE_TURN})),
                InterfaceName.OLLAMA: _DefaultProjection(frozenset({FinishReason.PAUSE_TURN, FinishReason.REFUSAL})),
                InterfaceName.ANTHROPIC: _DefaultProjection(frozenset()),
            }
        normalized = dict(projections)
        if set(normalized) != set(InterfaceName):
            raise ValueError("semantic projection registry must contain exactly one projection per interface")
        if not all(isinstance(value, TargetSemanticProjection) for value in normalized.values()):
            raise TypeError("semantic projections must implement TargetSemanticProjection")
        self._projections = dict(normalized)

    def get(self, interface: InterfaceName) -> TargetSemanticProjection:
        try:
            return self._projections[interface]
        except KeyError as error:
            raise ValueError(f"no semantic projection for interface '{interface}'") from error


def support_definitions() -> tuple[SemanticIdentity, ...]:
    """Return the complete application-owned semantic vocabulary in order."""
    return tuple(SemanticIdentity)


class SemanticSupportResolver:
    def __init__(self, projections: SemanticProjectionRegistry | None = None) -> None:
        self._projections = projections or SemanticProjectionRegistry()

    def resolve_request(self, request: CompletionRequest, context, runtime_support: ProviderRuntimeSupport | None = None) -> tuple[EffectiveSemanticHandling, ...]:
        controls = request.controls
        resolved = context.resolved
        results: list[EffectiveSemanticHandling] = []
        if controls.context_management is not None:
            results.append(self._request(SemanticIdentity.CONTEXT_MANAGEMENT, SemanticHandlingOutcome.IGNORED_NOOP, "application_context_noop"))
        if controls.effort is not None:
            results.append(self._request(SemanticIdentity.EFFORT, SemanticHandlingOutcome.ADVISORY, "application_advisory"))
        if controls.prompt_cache is not None:
            results.append(self._request(SemanticIdentity.PROMPT_CACHE, SemanticHandlingOutcome.ADVISORY, "application_advisory"))
        if controls.output_constraint is not None:
            if resolved.compatibility.structured_output is StructuredOutputMode.UNSUPPORTED:
                raise InvalidCompletionRequest("selected model does not support JSON schema structured output")
            results.append(self._request(SemanticIdentity.STRUCTURED_OUTPUT, SemanticHandlingOutcome.SUPPORTED, "model_compatibility", SemanticResolutionScope.ROUTE_RESOLVED))
        if controls.reasoning is not None:
            reasoning = self._reasoning(runtime_support)
            results.append(reasoning)
            if reasoning.outcome is SemanticHandlingOutcome.REJECTED:
                return tuple(results)
        return tuple(results)

    def resolve_response(self, response_or_completed: CompletionResponse | ResponseCompleted, interface: InterfaceName) -> tuple[EffectiveSemanticHandling, ...]:
        if not isinstance(interface, InterfaceName):
            raise TypeError("interface must be an InterfaceName")
        reason = response_or_completed.finish_reason
        outcome = self._projections.get(interface).finish_reason_outcome(reason)
        if not isinstance(outcome, SemanticHandlingOutcome):
            raise ValueError("target finish-reason projection returned an invalid outcome")
        return (EffectiveSemanticHandling(
            SemanticIdentity.FINISH_REASON,
            SemanticSupportPhase.RESPONSE,
            SemanticResolutionScope.RUNTIME_OBSERVED,
            outcome,
            f"target_{interface.value}",
        ),)

    @staticmethod
    def _request(semantic: SemanticIdentity, outcome: SemanticHandlingOutcome, reason: str, scope: SemanticResolutionScope = SemanticResolutionScope.STATIC) -> EffectiveSemanticHandling:
        return EffectiveSemanticHandling(semantic, SemanticSupportPhase.REQUEST, scope, outcome, reason)

    @classmethod
    def _reasoning(cls, runtime_support: ProviderRuntimeSupport | None) -> EffectiveSemanticHandling:
        support = runtime_support or ProviderRuntimeSupport(ProviderExtensionOrigin.UNKNOWN)
        declaration = support.declaration
        if declaration is not None:
            disposition = declaration.support[SemanticIdentity.REASONING]
            if disposition is ProviderSemanticDisposition.IMPLEMENTED:
                return cls._request(SemanticIdentity.REASONING, SemanticHandlingOutcome.SUPPORTED, "provider_declaration", SemanticResolutionScope.ROUTE_RESOLVED)
            if disposition is ProviderSemanticDisposition.UNSUPPORTED:
                return cls._request(SemanticIdentity.REASONING, SemanticHandlingOutcome.REJECTED, "provider_declaration", SemanticResolutionScope.ROUTE_RESOLVED)
            raise ValueError("invalid provider reasoning disposition")
        if support.origin is ProviderExtensionOrigin.ENTRY_POINT:
            return cls._request(SemanticIdentity.REASONING, SemanticHandlingOutcome.UNKNOWN, "legacy_extension_passthrough", SemanticResolutionScope.ROUTE_RESOLVED)
        raise InvalidCompletionRequest("selected execution does not support reasoning")


__all__ = ["SemanticProjectionRegistry", "SemanticSupportResolver", "TargetSemanticProjection", "support_definitions"]
