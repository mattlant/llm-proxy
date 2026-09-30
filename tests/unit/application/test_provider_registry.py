from __future__ import annotations

import importlib.metadata
from importlib.metadata import EntryPoint
from pathlib import Path
import shutil

import pytest

from llm_proxy.application.provider_registry import ProviderExtensionOrigin, ProviderGatewayRegistry, ProviderRegistryError, activate_provider_instances, discover_extensions
from llm_proxy.application.runtime import ActivationPolicy
from llm_proxy.configuration.models import ProviderConfig
from llm_proxy.providers.ollama.extension import OllamaExtension
from llm_proxy.provider_extensions import ProviderCapabilities, ProviderExtensionMetadata, ProviderListedModel, ProviderListingError, ProviderListingFailureCategory, SdkCompatibility, validate_sdk_compatibility


@pytest.mark.asyncio
async def test_listing_registry_bounds_timeout_validates_results_and_isolates_instances():
    import asyncio
    class Slow:
        async def list_models(self):
            await asyncio.sleep(1)
            return ()
    class Duplicate:
        async def list_models(self): return (ProviderListedModel("same"), ProviderListedModel("same"))
    class Healthy:
        async def list_models(self): return (ProviderListedModel("healthy"),)
    gateways = ProviderGatewayRegistry({"slow": Slow(), "duplicate": Duplicate(), "healthy": Healthy()}, capabilities={name: ProviderCapabilities(model_listing=True) for name in ("slow", "duplicate", "healthy")})
    with pytest.raises(ProviderListingError) as timeout:
        await gateways.list_models("slow", 1)
    assert timeout.value.category is ProviderListingFailureCategory.TIMEOUT
    with pytest.raises(ProviderListingError) as invalid:
        await gateways.list_models("duplicate", 1)
    assert invalid.value.category is ProviderListingFailureCategory.INVALID_RESPONSE
    assert await gateways.list_models("healthy", 1) == (ProviderListedModel("healthy"),)


def test_sdk_05_retains_external_sdk_03_extensions_without_listing_capability():
    compatibility = validate_sdk_compatibility(SdkCompatibility("0.3.0", "0.3.999"))

    assert compatibility.compatible
    assert not ProviderCapabilities().model_listing


def provider(name: str = "local", extension: str = "ollama", enabled: bool = True) -> ProviderConfig:
    return ProviderConfig(name, extension, enabled, {"base_url": "http://provider.test", "outbound_interface": "openai"})


def test_builtin_activation_is_immutable_and_uses_instance_lookup() -> None:
    extensions = discover_extensions((("built-in ollama", OllamaExtension()),), entry_points=lambda: ())
    gateways = activate_provider_instances(extensions, {"local": provider()})

    assert gateways.instance_names == ("local",)
    assert gateways.get("local").provider_name == "local"
    assert extensions.origin("ollama") is ProviderExtensionOrigin.BUILT_IN
    assert gateways.support("local").origin is ProviderExtensionOrigin.BUILT_IN
    with pytest.raises(TypeError):
        extensions.entries["other"] = OllamaExtension()  # type: ignore[index]


def test_unused_load_failure_is_diagnostic_but_missing_configured_extension_blocks_activation() -> None:
    class BrokenPoint:
        name = "broken"
        value = "fixture:broken"
        def load(self):
            raise ImportError("nope")

    extensions = discover_extensions((("built-in ollama", OllamaExtension()),), entry_points=lambda: (BrokenPoint(),))
    assert extensions.diagnostics[0].message == "load failed: ImportError"
    with pytest.raises(ProviderRegistryError, match="missing"):
        activate_provider_instances(extensions, {"missing": provider("missing", "missing")})


def test_duplicate_extension_ids_are_rejected_deterministically() -> None:
    with pytest.raises(ProviderRegistryError, match="duplicate provider extension ID 'ollama'"):
        discover_extensions((("z built-in", OllamaExtension()), ("a external", OllamaExtension())), entry_points=lambda: ())


def test_duplicate_ids_fail_before_compatibility_filtering_and_independent_of_order() -> None:
    class Factory:
        def create(self, config): pass

    class Compatible:
        metadata = ProviderExtensionMetadata("duplicate", "Duplicate", "fixture", "1")
        compatibility = SdkCompatibility("0.1.0")
        capabilities = ProviderCapabilities()
        factory = Factory()

    class Incompatible(Compatible):
        compatibility = SdkCompatibility("9.0.0")

    for extensions in ((("a", Compatible()), ("z", Incompatible())), (("z", Incompatible()), ("a", Compatible()))):
        with pytest.raises(ProviderRegistryError, match="duplicate provider extension ID 'duplicate'"):
            discover_extensions(extensions, entry_points=lambda: ())


def test_incompatible_unused_extension_is_isolated() -> None:
    class Incompatible:
        metadata = ProviderExtensionMetadata("example.incompatible", "Incompatible", "fixture", "1")
        compatibility = SdkCompatibility("9.0.0")
        capabilities = ProviderCapabilities()
        factory = object()

    extensions = discover_extensions((("fixture", Incompatible()),), entry_points=lambda: ())
    assert not extensions.entries
    assert extensions.diagnostics[0].extension_id == "example.incompatible"


def test_configured_incompatible_extension_blocks_activation() -> None:
    class Incompatible:
        metadata = ProviderExtensionMetadata("example.incompatible", "Incompatible", "fixture", "1")
        compatibility = SdkCompatibility("9.0.0")
        capabilities = ProviderCapabilities()
        factory = object()

    extensions = discover_extensions((("fixture", Incompatible()),), entry_points=lambda: ())
    with pytest.raises(ProviderRegistryError, match="requires unavailable extension 'example.incompatible'"):
        activate_provider_instances(
            extensions,
            {"incompatible": ProviderConfig("incompatible", "example.incompatible", True, {})},
        )


def test_provider_configuration_changes_require_restart() -> None:
    extensions = discover_extensions((('built-in ollama', OllamaExtension()),), entry_points=lambda: ())
    providers = {"local": provider()}
    gateways = activate_provider_instances(extensions, providers)

    with pytest.raises(ProviderRegistryError, match="restart is required"):
        gateways.assert_matches(ActivationPolicy().provider_identity({
            "local": ProviderConfig("local", "ollama", True, {"base_url": "http://other.test"}),
        }))


def test_activation_failure_closes_already_constructed_gateways() -> None:
    closed: list[str] = []

    class Gateway:
        def close(self) -> None:
            closed.append("closed")

    class Factory:
        def create(self, config):
            if config.instance_name == "bad":
                raise ValueError("invalid")
            return Gateway()

    class Extension:
        metadata = ProviderExtensionMetadata("example", "Example", "fixture", "1")
        compatibility = SdkCompatibility("0.1.0")
        capabilities = ProviderCapabilities()
        factory = Factory()

    registry = discover_extensions((("fixture", Extension()),), entry_points=lambda: ())
    with pytest.raises(ProviderRegistryError, match="provider activation failed"):
        activate_provider_instances(
            registry,
            {
                "good": ProviderConfig("good", "example", True, {}),
                "bad": ProviderConfig("bad", "example", True, {}),
            },
        )

    assert closed == ["closed"]


async def test_activation_failure_awaits_async_and_mixed_cleanup_exactly_once() -> None:
    closed: list[str] = []

    class AsyncGateway:
        async def aclose(self):
            closed.append("async")

    class SyncGateway:
        def close(self):
            closed.append("sync")

    class Factory:
        def create(self, config):
            if config.instance_name == "bad":
                raise ValueError("invalid")
            return AsyncGateway() if config.instance_name == "async" else SyncGateway()

    class Extension:
        metadata = ProviderExtensionMetadata("cleanup", "Cleanup", "fixture", "1")
        compatibility = SdkCompatibility("0.1.0")
        capabilities = ProviderCapabilities()
        factory = Factory()

    extensions = discover_extensions((("fixture", Extension()),), entry_points=lambda: ())
    with pytest.raises(ProviderRegistryError, match="provider activation failed"):
        activate_provider_instances(extensions, {
            "async": ProviderConfig("async", "cleanup", True, {}),
            "sync": ProviderConfig("sync", "cleanup", True, {}),
            "bad": ProviderConfig("bad", "cleanup", True, {}),
        })
    assert sorted(closed) == ["async", "sync"]


@pytest.mark.parametrize("broken", [
    type("BadMetadata", (), {"metadata": object(), "compatibility": SdkCompatibility("0.1.0"), "capabilities": ProviderCapabilities(), "factory": object()}),
    type("BadCompatibility", (), {"metadata": ProviderExtensionMetadata("bad.compatibility", "Bad", "fixture", "1"), "compatibility": object(), "capabilities": ProviderCapabilities(), "factory": object()}),
    type("BadCapabilities", (), {"metadata": ProviderExtensionMetadata("bad.capabilities", "Bad", "fixture", "1"), "compatibility": SdkCompatibility("0.1.0"), "capabilities": object(), "factory": object()}),
])
def test_malformed_unused_extensions_are_bounded_diagnostics_and_valid_extensions_survive(broken) -> None:
    valid = OllamaExtension()
    registry = discover_extensions((("broken", broken()), ("valid", valid)), entry_points=lambda: ())
    assert tuple(registry.entries) == ("ollama",)
    assert registry.diagnostics[0].source == "broken"
    with pytest.raises(ProviderRegistryError, match="requires unavailable extension"):
        activate_provider_instances(registry, {"bad": ProviderConfig("bad", "missing", True, {})})


async def test_external_entry_point_loads_activates_and_dispatches_without_a_switch(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.syspath_prepend(str(Path(__file__).parents[2] / "fixtures"))
    entry_point = EntryPoint("deterministic", "external_provider.deterministic_provider:DeterministicExtension", "llm_proxy.providers")
    extensions = discover_extensions((), entry_points=lambda: (entry_point,))
    gateways = activate_provider_instances(
        extensions,
        {"fixture": ProviderConfig("fixture", "example.deterministic", True, {"answer": "fixed"})},
    )
    assert extensions.origin("example.deterministic") is ProviderExtensionOrigin.ENTRY_POINT
    assert gateways.support("fixture").origin is ProviderExtensionOrigin.ENTRY_POINT

    from llm_proxy.domain.content import TextContent
    from llm_proxy.domain.messages import Message, MessageRole
    from llm_proxy.domain.requests import CompletionRequest, SamplingParameters
    from llm_proxy.provider_extensions import CompletionExecution

    request = CompletionRequest("public-model", (Message(MessageRole.USER, (TextContent("hello"),)),))
    response = await gateways.get("fixture").complete(
        CompletionExecution(request, "upstream", "fixture", SamplingParameters(), request.model)
    )

    assert response.message.content == (TextContent("external response"),)


def test_installed_external_fixture_exposes_a_real_entry_point(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    fixture_root = Path(__file__).parents[2] / "fixtures"
    target = tmp_path / "site-packages"
    shutil.copytree(fixture_root / "external_provider", target / "external_provider")
    dist_info = target / "example_deterministic_provider-0.1.0.dist-info"
    dist_info.mkdir()
    (dist_info / "METADATA").write_text("Metadata-Version: 2.1\nName: example-deterministic-provider\nVersion: 0.1.0\n")
    (dist_info / "entry_points.txt").write_text(
        "[llm_proxy.providers]\ndeterministic = external_provider.deterministic_provider:DeterministicExtension\n"
    )
    monkeypatch.syspath_prepend(str(target))
    points = [
        point
        for distribution in importlib.metadata.distributions(path=[str(target)])
        for point in distribution.entry_points
        if point.group == "llm_proxy.providers"
    ]

    extensions = discover_extensions((), entry_points=lambda: points)

    assert tuple(extensions.entries) == ("example.deterministic",)
