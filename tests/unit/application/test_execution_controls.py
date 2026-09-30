from __future__ import annotations

import pytest

from llm_proxy.domain.requests import (
    EffortLevel,
    ExecutionControls,
    JsonSchemaOutputConstraint,
    ReasoningDisplay,
    ReasoningMode,
    ReasoningPreference,
)


def test_execution_controls_are_immutable_and_defensively_copy_schema() -> None:
    schema = {"type": "object", "properties": {"value": {"type": "string"}}}
    controls = ExecutionControls(
        effort=EffortLevel.HIGH,
        output_constraint=JsonSchemaOutputConstraint(schema),
    )
    schema["properties"]["value"]["type"] = "number"

    assert controls.effort is EffortLevel.HIGH
    assert controls.output_constraint.schema["properties"]["value"]["type"] == "string"
    with pytest.raises(TypeError):
        controls.output_constraint.schema["type"] = "array"


@pytest.mark.parametrize("value", [{"type": {"invalid"}}, {1: "invalid"}])
def test_json_schema_constraint_rejects_non_json_values(value) -> None:
    with pytest.raises(TypeError):
        JsonSchemaOutputConstraint(value)


def test_reasoning_preference_enforces_mode_budget_invariants() -> None:
    preference = ReasoningPreference(ReasoningMode.ENABLED, ReasoningDisplay.OMITTED, budget_tokens=128)

    assert preference.budget_tokens == 128
    with pytest.raises(ValueError, match="requires"):
        ReasoningPreference(ReasoningMode.ENABLED, ReasoningDisplay.OMITTED)
    with pytest.raises(ValueError, match="does not accept"):
        ReasoningPreference(ReasoningMode.ADAPTIVE, ReasoningDisplay.OMITTED, budget_tokens=128)
