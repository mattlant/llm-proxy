from __future__ import annotations

from types import SimpleNamespace

import pytest

from llm_proxy.application.provider_registry import ProviderExtensionOrigin, ProviderRuntimeSupport
from llm_proxy.application.semantic_support import SemanticSupportResolver
from llm_proxy.configuration.models import InterfaceName, ModelCompatibility, StructuredOutputMode
from llm_proxy.domain.content import TextContent
from llm_proxy.domain.errors import InvalidCompletionRequest
from llm_proxy.domain.messages import Message, MessageRole
from llm_proxy.domain.requests import CompletionRequest, ContextManagementRequest, ContextEditRequest, ContextEditKind, EffortLevel, PromptCachePreference, ReasoningDisplay, ReasoningMode, ReasoningPreference
from llm_proxy.domain.responses import FinishReason
from llm_proxy.domain.semantic_support import SemanticHandlingOutcome, SemanticIdentity, SemanticResolutionScope
from llm_proxy.provider_extensions import ProviderSemanticDeclaration, ProviderSemanticDisposition


def request(**controls):
    from llm_proxy.domain.requests import ExecutionControls
    return CompletionRequest("model", (Message(MessageRole.USER, (TextContent("hello"),)),), controls=ExecutionControls(**controls))


def context(structured_output=StructuredOutputMode.UNSUPPORTED):
    return SimpleNamespace(resolved=SimpleNamespace(compatibility=ModelCompatibility(structured_output=structured_output)))


def test_request_resolution_composes_application_owned_outcomes_in_fixed_order() -> None:
    subject = SemanticSupportResolver()
    result = subject.resolve_request(request(
        context_management=ContextManagementRequest((ContextEditRequest(ContextEditKind.CLEAR_THINKING, "all"),)),
        effort=EffortLevel.LOW,
        prompt_cache=PromptCachePreference(1),
    ), context())

    assert [(item.semantic, item.outcome, item.reason) for item in result] == [
        (SemanticIdentity.CONTEXT_MANAGEMENT, SemanticHandlingOutcome.IGNORED_NOOP, "application_context_noop"),
        (SemanticIdentity.EFFORT, SemanticHandlingOutcome.ADVISORY, "application_advisory"),
        (SemanticIdentity.PROMPT_CACHE, SemanticHandlingOutcome.ADVISORY, "application_advisory"),
    ]


def test_reasoning_bridge_is_only_entry_point_scoped() -> None:
    subject = SemanticSupportResolver()
    reasoning = ReasoningPreference(ReasoningMode.ADAPTIVE, ReasoningDisplay.OMITTED)
    bridge = subject.resolve_request(request(reasoning=reasoning), context(), ProviderRuntimeSupport(ProviderExtensionOrigin.ENTRY_POINT))
    assert bridge[0].outcome is SemanticHandlingOutcome.UNKNOWN
    assert bridge[0].reason == "legacy_extension_passthrough"

    with pytest.raises(InvalidCompletionRequest, match="does not support reasoning"):
        subject.resolve_request(request(reasoning=reasoning), context(), ProviderRuntimeSupport(ProviderExtensionOrigin.BUILT_IN))


@pytest.mark.parametrize(("disposition", "outcome"), [
    (ProviderSemanticDisposition.IMPLEMENTED, SemanticHandlingOutcome.SUPPORTED),
    (ProviderSemanticDisposition.UNSUPPORTED, SemanticHandlingOutcome.REJECTED),
])
def test_provider_declaration_translates_to_application_outcome(disposition, outcome) -> None:
    declaration = ProviderSemanticDeclaration({SemanticIdentity.REASONING: disposition})
    result = SemanticSupportResolver().resolve_request(
        request(reasoning=ReasoningPreference(ReasoningMode.ADAPTIVE, ReasoningDisplay.OMITTED)),
        context(),
        ProviderRuntimeSupport(ProviderExtensionOrigin.BUILT_IN, declaration),
    )
    assert result[0].outcome is outcome
    assert result[0].scope is SemanticResolutionScope.ROUTE_RESOLVED


def test_structured_output_is_resolved_from_model_compatibility() -> None:
    from llm_proxy.domain.requests import JsonSchemaOutputConstraint
    result = SemanticSupportResolver().resolve_request(
        request(output_constraint=JsonSchemaOutputConstraint({"type": "object"})),
        context(StructuredOutputMode.OPENAI_JSON_SCHEMA),
    )
    assert result[0].outcome is SemanticHandlingOutcome.SUPPORTED
    with pytest.raises(InvalidCompletionRequest, match="JSON schema structured output"):
        SemanticSupportResolver().resolve_request(request(output_constraint=JsonSchemaOutputConstraint({"type": "object"})), context())


@pytest.mark.parametrize(("interface", "reason", "outcome"), [
    (InterfaceName.OPENAI, FinishReason.PAUSE_TURN, SemanticHandlingOutcome.LOSSY),
    (InterfaceName.OLLAMA, FinishReason.REFUSAL, SemanticHandlingOutcome.LOSSY),
    (InterfaceName.ANTHROPIC, FinishReason.REFUSAL, SemanticHandlingOutcome.SUPPORTED),
])
def test_response_resolution_uses_target_fidelity(interface, reason, outcome) -> None:
    from llm_proxy.domain.events import ResponseCompleted
    from llm_proxy.domain.responses import Usage
    result = SemanticSupportResolver().resolve_response(ResponseCompleted(reason, None, Usage(1, 1)), interface)
    assert result == (result[0],)
    assert result[0].semantic is SemanticIdentity.FINISH_REASON
    assert result[0].outcome is outcome
