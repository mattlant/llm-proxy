import pytest

from llm_proxy.management.command_registry import CommandRegistryError, validate_command_executor
from llm_proxy.provider_extensions import ManagementCommandDescriptor, ManagementCommandMutability, ManagementCommandPermission


def test_command_capability_requires_executor_and_rejects_invalid_schema():
    with pytest.raises(CommandRegistryError, match="requires a command executor"):
        validate_command_executor(object(), True)

    class Gateway:
        management_commands = (ManagementCommandDescriptor("get_state", "Return state", {"$ref": "https://invalid.test/schema"}, {"type": "object"}, ManagementCommandMutability.READ_ONLY, ManagementCommandPermission.INSPECT),)
        async def execute_management_command(self, command_name, payload, context): pass

    with pytest.raises(CommandRegistryError, match="invalid command schema"):
        validate_command_executor(Gateway(), True)
