from __future__ import annotations

from copy import deepcopy

import pytest

from llm_proxy.configuration.loader import ConfigurationError, ConfigurationLoader
from llm_proxy.configuration.validator import ConfigurationValidator


def realistic_config() -> dict:
    return {
        "server": {
            "host": "127.0.0.1",
            "port": 11435,
            "timeout_seconds": 0,
            "config_reload_seconds": 1,
            "log_level": "INFO",
        },
        "interfaces": {
            "openai": {"enabled": True},
            "ollama": {"enabled": True},
            "anthropic": {
                "enabled": True,
                "authentication": {"allow_any_token": True, "token": None},
            },
        },
        "providers": {
            "rtx-3090": {
                "extension": "ollama",
                "config": {"base_url": "http://provider.test:11434", "outbound_interface": "openai"},
                "enabled": True,
            }
        },
        "models": {
            "qwable": {
                "upstream_model": "hf.co/lordx64/Qwable-v2-GGUF:Q4_K_M",
                "provider": "rtx-3090",
                "interfaces": ["anthropic"],
                "aliases": {"anthropic": ["local-opus"]},
                "parameters": {"temperature": 0.7},
            },
            "qwopus": {
                "upstream_model": "hf.co/barozp/Qwopus3.6-35B-A3B-v1-GGUF:Q4_K_M",
                "provider": "rtx-3090",
                "interfaces": ["openai", "ollama"],
                "parameters": {"temperature": 1.0, "repeat_penalty": 1.12},
            },
            "ornith": {
                "upstream_model": "hf.co/deepreinforce-ai/Ornith-1.0-35B-GGUF:Q4_K_M",
                "provider": "rtx-3090",
                "interfaces": ["openai", "ollama"],
                "parameters": {"temperature": 0.6},
            },
        },
        "policies": [],
    }


def load(raw: dict):
    return ConfigurationLoader().load_mapping(raw)


def assert_invalid(raw: dict, path: str, message: str):
    with pytest.raises(ConfigurationError, match=rf"{path}.*{message}"):
        ConfigurationValidator().validate(load(raw))


def test_realistic_qwable_qwopus_ornith_configuration_is_valid():
    config = load(realistic_config())

    ConfigurationValidator().validate(config)


def test_at_least_one_interface_provider_and_model_are_required():
    raw = realistic_config()
    raw["interfaces"] = {"openai": {"enabled": False}}
    assert_invalid(raw, "interfaces", "at least one interface")

    raw = realistic_config()
    raw["providers"]["rtx-3090"]["enabled"] = False
    assert_invalid(raw, "providers", "at least one provider")

    raw = realistic_config()
    raw["models"] = {}
    assert_invalid(raw, "models", "at least one model")


def test_model_provider_must_exist_and_be_enabled():
    raw = realistic_config()
    raw["models"]["qwable"]["provider"] = "missing"
    assert_invalid(raw, r"models\.qwable\.provider", "unknown provider")

    raw = realistic_config()
    raw["providers"]["rtx-3090"]["enabled"] = False
    raw["providers"]["fallback"] = {
        "extension": "ollama",
        "config": {"base_url": "http://provider.test:11434", "outbound_interface": "openai"},
        "enabled": True,
    }
    assert_invalid(raw, r"models\.qwable\.provider", "disabled provider")


def test_model_interfaces_must_exist_and_be_enabled():
    raw = realistic_config()
    raw["models"]["qwable"]["interfaces"] = ["missing"]
    with pytest.raises(ConfigurationError, match=r"models\.qwable\.interfaces.*unsupported value"):
        load(raw)

    raw = realistic_config()
    raw["interfaces"]["anthropic"]["enabled"] = False
    assert_invalid(raw, r"models\.qwable\.interfaces", "disabled interface")


def test_aliases_must_belong_to_an_exposed_interface():
    raw = realistic_config()
    raw["models"]["qwable"]["aliases"] = {"openai": ["local-opus"]}
    assert_invalid(raw, r"models\.qwable\.aliases\.openai", "not exposed")


def test_profile_names_colliding_under_case_folding_are_rejected():
    raw = realistic_config()
    raw["models"]["QWABLE"] = deepcopy(raw["models"]["qwable"])
    assert_invalid(raw, r"models\.QWABLE", "collides")


def test_aliases_are_unique_per_interface_under_case_folding():
    raw = realistic_config()
    raw["models"]["qwopus"]["aliases"] = {"openai": ["local-opus"]}
    raw["models"]["ornith"]["aliases"] = {"openai": ["LOCAL-OPUS"]}
    assert_invalid(raw, r"models\.ornith\.aliases\.openai", "collides")


def test_alias_cannot_shadow_visible_profile_on_same_interface():
    raw = realistic_config()
    raw["models"]["qwable"]["aliases"]["anthropic"] = ["qwable"]
    # The profile itself rejects its own name as an alias during typed construction.
    with pytest.raises(ConfigurationError, match=r"models\.qwable: profile name cannot"):
        load(raw)

    raw = realistic_config()
    raw["models"]["qwable"]["aliases"]["anthropic"] = ["qwopus"]
    raw["models"]["qwopus"]["interfaces"] = ["anthropic"]
    assert_invalid(raw, r"models\.qwable\.aliases\.anthropic", "shadows visible profile")


def test_alias_cannot_use_provider_qualified_separator():
    raw = realistic_config()
    raw["models"]["qwable"]["aliases"] = {"anthropic": ["provider::model"]}
    assert_invalid(raw, r"models\.qwable\.aliases\.anthropic", "must not contain")


def test_provider_configuration_is_immutable_and_extension_owned():
    config = load(realistic_config())
    provider = config.providers["rtx-3090"]
    assert provider.extension_id == "ollama"
    with pytest.raises(TypeError):
        provider.config["base_url"] = "http://other.test"  # type: ignore[index]


def test_policy_model_provider_and_interface_references_are_validated():
    raw = realistic_config()
    raw["policies"] = [{
        "name": "valid",
        "enabled": True,
        "match": {"interfaces": ["anthropic"]},
        "actions": {"model": "local-opus", "provider": "rtx-3090"},
    }]
    ConfigurationValidator().validate(load(raw))

    raw = realistic_config()
    raw["policies"] = [{"name": "bad", "enabled": True, "match": {}, "actions": {"model": "missing"}}]
    assert_invalid(raw, r"policies\[0\]\.actions\.model", "unknown model")

    raw = realistic_config()
    raw["policies"] = [{"name": "bad", "enabled": True, "match": {}, "actions": {"provider": "missing"}}]
    assert_invalid(raw, r"policies\[0\]\.actions\.provider", "unknown provider")
