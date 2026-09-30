"""Deterministic SDK compatibility validation for future registry startup."""
from __future__ import annotations

from dataclasses import dataclass

from packaging.version import InvalidVersion, Version

from .contracts import PROVIDER_SDK_VERSION, SdkCompatibility


@dataclass(frozen=True, slots=True)
class CompatibilityResult:
    compatible: bool
    diagnostic: str


def validate_sdk_compatibility(declared: SdkCompatibility, installed: str = PROVIDER_SDK_VERSION) -> CompatibilityResult:
    try:
        current = Version(installed)
        minimum = Version(declared.minimum)
        maximum = Version(declared.maximum) if declared.maximum is not None else None
    except InvalidVersion as error:
        return CompatibilityResult(False, f"invalid SDK compatibility version: {error}")
    if maximum is not None and minimum > maximum:
        return CompatibilityResult(False, "minimum SDK version exceeds maximum SDK version")
    if current < minimum:
        return CompatibilityResult(False, f"SDK {current} is older than minimum supported {minimum}")
    # SDK 0.4 only adds optional wire contracts.  Existing 0.3 extensions
    # retain their canonical-only behavior without needing a rebuild.
    optional_wire_upgrade = current == Version("0.5.0") and maximum is not None and maximum >= Version("0.3.0")
    if maximum is not None and current > maximum and not optional_wire_upgrade:
        return CompatibilityResult(False, f"SDK {current} is newer than maximum supported {maximum}")
    return CompatibilityResult(True, f"SDK {current} is compatible")
