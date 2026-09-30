from llm_proxy.domain.requests import SamplingParameters
from llm_proxy.interfaces.openai.parameters import mutation_operations, parameters_from_wire_payload


def test_wire_parameter_adapter_preserves_unknown_values() -> None:
    assert parameters_from_wire_payload({"model": "m", "temperature": 0.2, "vendor": {"mode": "x"}}) == SamplingParameters(temperature=0.2, extra={"vendor": {"mode": "x"}})


def test_wire_parameter_adapter_rejects_target_collision() -> None:
    try:
        mutation_operations({}, SamplingParameters(extra={"temperature": 0.2}))
    except ValueError as error:
        assert "temperature" in str(error)
    else:
        raise AssertionError("expected collision rejection")


def test_wire_parameter_adapter_keeps_non_facade_catalog_values_compatible() -> None:
    assert parameters_from_wire_payload({"model": "m", "n": 2}) == SamplingParameters(extra={"n": 2})
    assert mutation_operations({"n": 1}, SamplingParameters(extra={"n": 2}))[0].pointer == "/n"


def test_wire_parameter_adapter_thaws_nested_compatibility_values() -> None:
    operation = mutation_operations({}, SamplingParameters(extra={"vendor": {"mode": "x"}}))[0]
    assert operation.value == {"mode": "x"}
    assert type(operation.value) is dict
