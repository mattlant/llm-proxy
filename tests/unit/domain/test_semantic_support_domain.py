from __future__ import annotations

import pytest

from llm_proxy.domain.semantic_support import (
    EffectiveSemanticHandling,
    SemanticHandlingOutcome,
    SemanticIdentity,
    SemanticResolutionScope,
    SemanticSupportPhase,
)


def test_effective_handling_is_immutable_and_validated() -> None:
    value = EffectiveSemanticHandling(
        SemanticIdentity.REASONING,
        SemanticSupportPhase.REQUEST,
        SemanticResolutionScope.ROUTE_RESOLVED,
        SemanticHandlingOutcome.UNKNOWN,
        "legacy_extension_passthrough",
    )

    assert value.reason == "legacy_extension_passthrough"
    with pytest.raises((AttributeError, TypeError)):
        value.reason = "changed"  # type: ignore[misc]


@pytest.mark.parametrize("kwargs", [
    {"semantic": "reasoning"},
    {"phase": "request"},
    {"scope": "static"},
    {"outcome": "supported"},
    {"reason": ""},
])
def test_effective_handling_rejects_invalid_values(kwargs) -> None:
    values = {
        "semantic": SemanticIdentity.REASONING,
        "phase": SemanticSupportPhase.REQUEST,
        "scope": SemanticResolutionScope.STATIC,
        "outcome": SemanticHandlingOutcome.SUPPORTED,
        "reason": "test",
    }
    values.update(kwargs)
    with pytest.raises((TypeError, ValueError)):
        EffectiveSemanticHandling(**values)
