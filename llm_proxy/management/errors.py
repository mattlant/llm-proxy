from __future__ import annotations


class ManagementError(Exception):
    code = "management_error"
    status_code = 400

    def __init__(self, message: str = "management request failed") -> None:
        super().__init__(message)


class AdminUnauthorized(ManagementError):
    code = "admin_unauthorized"
    status_code = 401

class RevisionConflict(ManagementError):
    code = "revision_conflict"
    status_code = 409

class InvalidConfiguration(ManagementError):
    code = "invalid_configuration"
    status_code = 422

class PersistenceFailed(ManagementError):
    code = "persistence_failed"
    status_code = 500

class ActivationFailed(ManagementError):
    code = "activation_failed"
    status_code = 500

class ConfigurationStateConflict(ManagementError):
    code = "configuration_state_conflict"
    status_code = 409

class ConfigurationStateIndeterminate(ManagementError):
    code = "configuration_state_indeterminate"
    status_code = 500


class RestartRequired(ManagementError):
    code = "restart_required"
    status_code = 409


class HealthUnavailable(ManagementError):
    code = "health_unavailable"
    status_code = 503


class ProviderInstanceNotFound(ManagementError):
    code = "provider_instance_not_found"
    status_code = 404


class ManagementCommandsUnavailable(ManagementError):
    code = "management_commands_unavailable"
    status_code = 404


class CommandNotFound(ManagementError):
    code = "command_not_found"
    status_code = 404


class CommandUnauthorized(ManagementError):
    code = "command_unauthorized"
    status_code = 403


class InvalidCommandInput(ManagementError):
    code = "invalid_command_input"
    status_code = 422


class InvalidCommandOutput(ManagementError):
    code = "invalid_command_output"
    status_code = 502


class CommandTimeout(ManagementError):
    code = "command_timeout"
    status_code = 504


class CommandCancelled(ManagementError):
    code = "command_cancelled"
    status_code = 499


class CommandFailed(ManagementError):
    code = "command_failed"
    status_code = 502
