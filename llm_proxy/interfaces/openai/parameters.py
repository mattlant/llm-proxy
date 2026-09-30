from __future__ import annotations

from llm_proxy.domain.parameters import CORE_PARAMETER_CATALOG, CompatibilityParameters, EffectiveParameters, ParameterOverlay
from llm_proxy.domain.requests import SamplingParameters
from llm_proxy.provider_extensions import WirePatchOperation


OPENAI_PARAMETER_ALIASES = {"stop": "stop_sequences"}


def _wire_value(value):
    if hasattr(value, "items"):
        return {key: _wire_value(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_wire_value(item) for item in value]
    return value


def parameters_from_source(source: object) -> SamplingParameters:
    values = {
        name: getattr(source, name)
        for name in CORE_PARAMETER_CATALOG.names
        if name != "stop_sequences" and hasattr(source, name) and getattr(source, name) is not None
    }
    stop = getattr(source, "stop", ())
    if stop:
        values["stop_sequences"] = (stop,) if isinstance(stop, str) else tuple(stop)
    return EffectiveParameters(values).to_legacy()


def parameters_from_wire_payload(payload: dict) -> SamplingParameters:
    values = {name: payload[name] for name in CORE_PARAMETER_CATALOG.names if name != "stop_sequences" and payload.get(name) is not None}
    if any(isinstance(value, bool) or not isinstance(value, (int, float)) for value in values.values()):
        raise ValueError("OpenAI request has invalid sampling parameter")
    stop = payload.get("stop", ())
    if stop == (): stop = ()
    elif isinstance(stop, str): stop = (stop,)
    elif isinstance(stop, list) and all(isinstance(item, str) for item in stop): stop = tuple(stop)
    elif stop is None: stop = ()
    else: raise ValueError("OpenAI request has invalid stop")
    excluded = {"model", "stream", "messages", "tools", "metadata", "stop", *CORE_PARAMETER_CATALOG.names}
    return EffectiveParameters(values, CompatibilityParameters({key: value for key, value in payload.items() if key not in excluded})).apply(ParameterOverlay(canonical={"stop_sequences": stop} if stop else {})).to_legacy()


def mutation_operations(payload: dict, parameters: SamplingParameters) -> list[WirePatchOperation]:
    effective = EffectiveParameters.from_legacy(parameters)
    operations = []
    for name, value in effective.canonical.items():
        target = "stop" if name == "stop_sequences" else name
        rendered = list(value) if name == "stop_sequences" else _wire_value(value)
        if payload.get(target) != rendered:
            operations.append(WirePatchOperation(f"/{target}", rendered))
    facade_wire_names = set(SamplingParameters.__dataclass_fields__) - {"extra"}
    collisions = (facade_wire_names | {"stop"}).intersection(effective.compatibility.values)
    if collisions:
        raise ValueError(f"extra parameters collide with standard fields: {', '.join(sorted(collisions))}")
    operations.extend(WirePatchOperation(f"/{name}", _wire_value(value)) for name, value in effective.compatibility.values.items() if payload.get(name) != value)
    return operations
