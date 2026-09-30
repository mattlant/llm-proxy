from __future__ import annotations

from dataclasses import dataclass

from llm_proxy.domain.requests import CompletionRequest, ContextManagementResult


@dataclass(frozen=True, slots=True)
class ContextEditApplication:
    request: CompletionRequest
    result: ContextManagementResult | None


def apply_context_edits(request: CompletionRequest) -> ContextEditApplication:
    """Apply only deterministic canonical history edits.

    Local Anthropic projections never emit signed thinking blocks, so the observed
    clear-thinking strategy has no canonical history to transform yet.
    """
    if request.controls.context_management is None:
        return ContextEditApplication(request, None)
    return ContextEditApplication(request, ContextManagementResult())
