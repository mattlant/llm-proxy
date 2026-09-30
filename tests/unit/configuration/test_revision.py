from llm_proxy.configuration.loader import ConfigurationLoader
from llm_proxy.configuration.revision import configuration_revision, serialize_configuration
from tests.unit.configuration.test_validator import realistic_config


def test_revision_is_deterministic_and_serialization_round_trips():
    config = ConfigurationLoader().load_mapping(realistic_config())
    serialized = serialize_configuration(config)
    assert configuration_revision(config) == configuration_revision(ConfigurationLoader().load_text(serialized))


def test_meaningful_change_has_a_new_revision():
    raw = realistic_config()
    first = configuration_revision(ConfigurationLoader().load_mapping(raw))
    raw["models"]["qwable"]["parameters"]["temperature"] = 0.8
    assert configuration_revision(ConfigurationLoader().load_mapping(raw)) != first


def test_additive_matchers_round_trip_and_change_revision():
    raw = realistic_config()
    raw["policies"] = [{
        "name": "matchers",
        "enabled": True,
        "match": {
            "model_contains_any": ["qw", "qw"], "model_contains_all": ["able"],
            "model_wildcard_any": ["qw*"], "model_wildcard_all": ["*able"],
            "tool_wildcard_any": ["read_*"], "tool_wildcard_all": ["*_file"],
        },
        "actions": {"parameters": {"temperature": 0.2}},
    }]
    config = ConfigurationLoader().load_mapping(raw)

    assert ConfigurationLoader().load_text(serialize_configuration(config)).policies[0].match == config.policies[0].match
    previous = configuration_revision(config)
    raw["policies"][0]["match"]["tool_wildcard_all"] = ["write_*"]
    assert configuration_revision(ConfigurationLoader().load_mapping(raw)) != previous
