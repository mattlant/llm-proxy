from __future__ import annotations

import pytest

from llm_proxy.provider_extensions import ManagementCommandContext, ManagementCommandDescriptor, ManagementCommandMutability, ManagementCommandPermission


def test_management_command_values_are_immutable_and_validate_metadata():
    descriptor = ManagementCommandDescriptor("get_state", "Return state", {"type": "object"}, {"type": "object"}, ManagementCommandMutability.READ_ONLY, ManagementCommandPermission.INSPECT)
    with pytest.raises(TypeError):
        descriptor.input_schema["type"] = "string"  # type: ignore[index]
    assert ManagementCommandContext("primary", "opaque_123").provider_instance == "primary"
    with pytest.raises(ValueError):
        ManagementCommandDescriptor("Bad Name", "x", {}, {}, ManagementCommandMutability.READ_ONLY, ManagementCommandPermission.INSPECT)
    with pytest.raises(ValueError):
        ManagementCommandDescriptor("set_state", "x", {}, {}, ManagementCommandMutability.STATE_CHANGING, ManagementCommandPermission.INSPECT)
