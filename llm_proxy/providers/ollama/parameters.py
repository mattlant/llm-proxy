from __future__ import annotations

import copy
from typing import Any

from llm_proxy.domain.parameters import CORE_PARAMETER_CATALOG, EffectiveParameters
from llm_proxy.domain.requests import SamplingParameters


class OllamaParameterProjection:
    """The built-in Ollama target's support and collision boundary."""

    _wire_names = {
        "temperature", "top_p", "top_k", "min_p", "repeat_penalty",
        "repeat_last_n", "max_tokens", "stop", "stop_sequences",
    }

    def project(self, parameters: SamplingParameters | EffectiveParameters) -> dict[str, Any]:
        effective = parameters if isinstance(parameters, EffectiveParameters) else EffectiveParameters.from_legacy(parameters)
        values: dict[str, Any] = {}
        for name, value in effective.canonical.items():
            if name == "stop_sequences":
                values["stop"] = list(value)
            elif name in CORE_PARAMETER_CATALOG.names:
                values[name] = value
            else:  # defensive: catalog entries must be explicitly projectable
                raise ValueError(f"unsupported Ollama parameter: {name}")
        collisions = self._wire_names.intersection(effective.compatibility.values)
        if collisions:
            raise ValueError(f"extra parameters collide with standard fields: {', '.join(sorted(collisions))}")
        values.update(copy.deepcopy(dict(effective.compatibility.values)))
        return values
