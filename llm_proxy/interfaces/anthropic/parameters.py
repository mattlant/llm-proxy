from llm_proxy.domain.parameters import EffectiveParameters
from llm_proxy.domain.requests import SamplingParameters

ANTHROPIC_PARAMETER_ALIASES = {"stop_sequences": "stop_sequences"}


def parameters_from_source(source: object) -> SamplingParameters:
    values = {name: getattr(source, name) for name in ("temperature", "top_p", "top_k", "max_tokens") if getattr(source, name) is not None}
    if source.stop_sequences:
        values["stop_sequences"] = tuple(source.stop_sequences)
    return EffectiveParameters(values).to_legacy()
