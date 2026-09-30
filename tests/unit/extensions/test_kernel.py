from __future__ import annotations

import pytest
from types import MappingProxyType

from llm_proxy.configuration.models import CapabilityBinding, ExtensionConfig
from llm_proxy.extensions.contracts import CapabilityRegistration, ExtensionInstanceConfig, ExtensionMetadata, ExtensionSdkCompatibility
from llm_proxy.extensions.kernel import CapabilityFamilyAdapter, ExtensionRegistryError, activate_extension_instances, discover_extensions
from llm_proxy.extensions.side_effect import SIDE_EFFECT_FAMILY, SIDE_EFFECT_VERSION, validate_side_effect

class Capability:
    def validate_arguments(self, arguments): pass
    async def invoke(self, context, arguments): pass

class Instance:
    def __init__(self, calls): self.calls = calls; self.registrations = (CapabilityRegistration(SIDE_EFFECT_FAMILY, SIDE_EFFECT_VERSION, "run", Capability()),)
    async def aclose(self): self.calls.append("closed")

class Extension:
    metadata = ExtensionMetadata("test.extension", "Test", "test", "1")
    compatibility = ExtensionSdkCompatibility("1.0.0", "1.0.0")
    class factory:
        calls = []
        @classmethod
        def create(cls, config):
            instance = Instance(cls.calls); cls.calls.append(config.instance_name); return instance

async def test_activation_registers_typed_capability_and_closes_once() -> None:
    catalog = discover_extensions((("test", Extension()),), entry_points=())
    registry = await activate_extension_instances(catalog, (ExtensionConfig("one", "test.extension", True, {}),), (CapabilityFamilyAdapter(SIDE_EFFECT_FAMILY, SIDE_EFFECT_VERSION, validate_side_effect),))
    assert isinstance(registry.get("one", SIDE_EFFECT_FAMILY, 1, "run"), Capability)
    await registry.aclose(); await registry.aclose()
    assert Extension.factory.calls[-1] == "closed"

def test_incompatible_extension_is_safe_diagnostic() -> None:
    class Incompatible(Extension): compatibility = ExtensionSdkCompatibility("2.0.0")
    catalog = discover_extensions((("bad", Incompatible()),), entry_points=())
    assert catalog.entries == {} and catalog.diagnostics[0].category == "incompatible"


async def test_activation_closes_current_instance_when_registration_validation_fails() -> None:
    closed = []

    class InvalidInstance:
        registrations = (CapabilityRegistration("unsupported", 1, "run", Capability()),)
        async def aclose(self): closed.append("current")

    class InvalidExtension(Extension):
        class factory:
            @staticmethod
            def create(config): return InvalidInstance()

    catalog = discover_extensions((("invalid", InvalidExtension()),), entry_points=())
    with pytest.raises(ExtensionRegistryError, match="unsupported"):
        await activate_extension_instances(catalog, (ExtensionConfig("one", "test.extension", True, {}),), (CapabilityFamilyAdapter(SIDE_EFFECT_FAMILY, SIDE_EFFECT_VERSION, validate_side_effect),))
    assert closed == ["current"]


def test_extension_config_and_capability_arguments_are_deeply_immutable() -> None:
    extension = ExtensionConfig("one", "test.extension", True, {"nested": {"items": ["value"]}})
    binding = CapabilityBinding("one", "instance", "side_effect", 1, "run", {"nested": {"items": ["value"]}})

    assert isinstance(extension.config, MappingProxyType)
    assert extension.config["nested"]["items"] == ("value",)
    assert binding.arguments["nested"]["items"] == ("value",)
    with pytest.raises(TypeError): extension.config["nested"]["other"] = "value"
    with pytest.raises(TypeError): binding.arguments["nested"]["other"] = "value"
