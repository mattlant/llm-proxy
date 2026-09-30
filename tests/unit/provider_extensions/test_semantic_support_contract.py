from __future__ import annotations

import pytest

from llm_proxy.domain.semantic_support import SemanticIdentity
from llm_proxy.provider_extensions import (
    ProviderCapabilities,
    ProviderExtensionMetadata,
    ProviderFactory,
    ProviderInstanceConfig,
    ProviderSemanticDeclaration,
    ProviderSemanticDisposition,
    SdkCompatibility,
    provider_semantic_declaration,
)


def test_legacy_extension_without_declaration_remains_valid() -> None:
    class Legacy:
        metadata = ProviderExtensionMetadata("legacy", "Legacy", "legacy", "1")
        compatibility = SdkCompatibility("0.1.0")
        capabilities = ProviderCapabilities()
        factory = object()

    assert provider_semantic_declaration(Legacy()) is None


@pytest.mark.parametrize("disposition", list(ProviderSemanticDisposition))
def test_declaration_contains_only_provider_reasoning_facts(disposition) -> None:
    declaration = ProviderSemanticDeclaration({SemanticIdentity.REASONING: disposition})

    assert declaration.support[SemanticIdentity.REASONING] is disposition
    with pytest.raises(TypeError):
        declaration.support[SemanticIdentity.REASONING] = disposition  # type: ignore[index]


def test_declaration_rejects_application_owned_outcomes_and_identities() -> None:
    with pytest.raises(ValueError):
        ProviderSemanticDeclaration({SemanticIdentity.EFFORT: ProviderSemanticDisposition.IMPLEMENTED})
    with pytest.raises(TypeError):
        ProviderSemanticDeclaration({SemanticIdentity.REASONING: "supported"})  # type: ignore[arg-type]


def test_accessor_rejects_malformed_opt_in_declaration() -> None:
    class Malformed:
        semantic_support = object()

    with pytest.raises(TypeError):
        provider_semantic_declaration(Malformed())
