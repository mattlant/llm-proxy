from dataclasses import replace

import pytest

from llm_proxy.application.runtime import ActivationPolicy, RuntimeDependencies
from llm_proxy.configuration.loader import ConfigurationLoader
from tests.unit.configuration.test_validator import realistic_config


def config():
    return ConfigurationLoader().load_mapping(realistic_config())


def test_policy_ignores_listing_enabled_but_classifies_startup_fields() -> None:
    policy = ActivationPolicy()
    active = config()
    listing_changed = replace(active, providers={
        name: replace(value, listing_enabled=not value.listing_enabled)
        for name, value in active.providers.items()
    })
    server_changed = replace(active, server=replace(active.server, host="0.0.0.0"))

    assert not policy.classify(active, listing_changed).restart_required
    assert policy.classify(active, server_changed).changed_groups == ("server",)


def test_runtime_dependencies_are_frozen() -> None:
    dependencies = RuntimeDependencies(provider_gateways=object())

    with pytest.raises((AttributeError, TypeError)):
        dependencies.provider_gateways = object()  # type: ignore[misc]


def test_administration_projection_is_policy_derived() -> None:
    projection = ActivationPolicy().administration_lifecycle(config())

    assert projection["server"]["lifecycle"] == "mixed"
    assert "host" in projection["server"]["restart_required_fields"]
    assert "payload_trace_mode" in projection["server"]["reload_safe_fields"]
    assert projection["interfaces"]["lifecycle"] == "mixed"
