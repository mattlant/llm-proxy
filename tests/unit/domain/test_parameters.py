import pytest

from llm_proxy.domain.parameters import CompatibilityParameters, EffectiveParameters, ParameterCatalog, ParameterDefinition, ParameterOverlay
from llm_proxy.domain.requests import SamplingParameters


def test_legacy_conversion_preserves_colliding_compatibility_value() -> None:
    legacy = SamplingParameters(temperature=0.2, extra={"temperature": 0.7, "seed": {"values": [4]}})

    assert EffectiveParameters.from_legacy(legacy).to_legacy() == legacy


def test_overlay_removal_then_later_assignment_wins_in_each_domain() -> None:
    initial = EffectiveParameters.from_legacy(SamplingParameters(temperature=0.2, extra={"seed": 1}))
    removed = initial.apply(ParameterOverlay(remove_canonical=frozenset({"temperature"}), remove_compatibility=frozenset({"seed"})))
    restored = removed.apply(ParameterOverlay(canonical={"temperature": 0.4}, compatibility={"seed": 2}))

    assert removed.to_legacy() == SamplingParameters()
    assert restored.to_legacy() == SamplingParameters(temperature=0.4, extra={"seed": 2})


def test_compatibility_values_are_json_and_names_are_strings() -> None:
    with pytest.raises(TypeError):
        CompatibilityParameters({1: "nope"})  # type: ignore[dict-item]
    with pytest.raises(TypeError):
        CompatibilityParameters({"nope": object()})


def test_catalog_rejects_duplicate_or_unknown_canonical_identities() -> None:
    definition = ParameterDefinition("known", int, lambda _: None)
    with pytest.raises(ValueError, match="duplicate"):
        ParameterCatalog((definition, definition))
    with pytest.raises(ValueError, match="unknown"):
        EffectiveParameters(canonical={"unknown": 1})
