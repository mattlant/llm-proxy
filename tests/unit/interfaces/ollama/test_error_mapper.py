from llm_proxy.application.errors import CapabilityExecutionError, CapabilityTimeoutError
from llm_proxy.extensions.side_effect import SideEffectFailureCategory
from llm_proxy.interfaces.ollama.error_mapper import OllamaErrorMapper


def test_maps_capability_failures_without_extension_details() -> None:
    mapper = OllamaErrorMapper()

    assert mapper.map_error(CapabilityExecutionError(SideEffectFailureCategory.FAILED)) == (503, {"error": "extension capability failed"})
    assert mapper.map_error(CapabilityTimeoutError()) == (504, {"error": "extension capability timed out"})
