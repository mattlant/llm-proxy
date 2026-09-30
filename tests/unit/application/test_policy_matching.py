from llm_proxy.application.policy_matching import (
    contains_casefold,
    equals_casefold,
    equals_sensitive,
    matches_all,
    matches_any,
    wildcard_casefold,
    wildcard_sensitive,
)


def test_quantifiers_cover_frozensets_empty_sources_and_empty_predicates():
    source = frozenset({"Alpha", "Beta"})

    assert matches_any(source, ("alpha",), equals_casefold)
    assert matches_all(source, ("alpha", "BETA"), equals_casefold)
    assert not matches_any((), ("alpha",), equals_casefold)
    assert not matches_all((), ("alpha",), equals_casefold)
    assert not matches_any(source, (), equals_casefold)
    assert matches_all(source, (), equals_casefold)


def test_scalar_comparators_preserve_case_and_whole_value_contracts():
    assert equals_casefold("Qwopus", "qwopus")
    assert not equals_sensitive("Read_File", "read_file")
    assert contains_casefold("Qwopus3.6-35B", "35b")
    assert wildcard_casefold("Qwopus3.6-35B", "*QWOPUS*35b")
    assert not wildcard_casefold("Qwopus3.6-35B", "Qwopus")
    assert wildcard_sensitive("copilot/github/read", "copilot/github/r[ea]ad")
    assert wildcard_sensitive("copilot/github/read", "copilot/*")
    assert not wildcard_sensitive("Copilot/github/read", "copilot/github/*")
