from types import SimpleNamespace

from llm_proxy.domain.requests import SamplingParameters
from llm_proxy.interfaces.anthropic.parameters import parameters_from_source


def test_anthropic_parameter_adapter_maps_canonical_values() -> None:
    source = SimpleNamespace(temperature=0.2, top_p=None, top_k=None, max_tokens=10, stop_sequences=["END"])
    assert parameters_from_source(source) == SamplingParameters(temperature=0.2, max_tokens=10, stop_sequences=("END",))
