from __future__ import annotations

from llm_proxy.domain.parameters import EffectiveParameters
from llm_proxy.domain.requests import SamplingParameters


OLLAMA_PARAMETER_ALIASES = {"num_predict": "max_tokens", "stop": "stop_sequences"}


def parameters_from_options(source: object | None) -> SamplingParameters:
    if source is None:
        return SamplingParameters()
    values = {canonical: getattr(source, wire) for wire, canonical in OLLAMA_PARAMETER_ALIASES.items() if wire != "stop" and getattr(source, wire) is not None}
    for name in ("temperature", "top_p", "top_k", "min_p", "repeat_penalty", "repeat_last_n"):
        value = getattr(source, name)
        if value is not None:
            values[name] = value
    stop = source.stop
    if stop:
        values["stop_sequences"] = (stop,) if isinstance(stop, str) else tuple(stop)
    return EffectiveParameters(values).to_legacy()
