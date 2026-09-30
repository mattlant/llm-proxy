from __future__ import annotations


class ModelResolutionError(Exception):
    def __init__(self, interface: str, requested_model: str) -> None:
        self.interface = interface
        self.requested_model = requested_model
        super().__init__(f"{interface}: {requested_model}")


class ModelNotFoundError(ModelResolutionError):
    pass


class ModelNotExposedError(ModelResolutionError):
    pass


class InterfaceDisabledError(ModelResolutionError):
    pass

class CapabilityExecutionError(Exception):
    def __init__(self, category) -> None: self.category = category

class CapabilityTimeoutError(Exception):
    pass
