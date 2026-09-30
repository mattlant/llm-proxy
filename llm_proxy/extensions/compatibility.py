from __future__ import annotations

from dataclasses import dataclass
from packaging.version import InvalidVersion, Version

from .contracts import EXTENSION_SDK_VERSION, ExtensionSdkCompatibility

@dataclass(frozen=True, slots=True)
class ExtensionCompatibilityResult:
    compatible: bool
    reason: str | None = None

def validate_extension_sdk_compatibility(declared: ExtensionSdkCompatibility, installed: str = EXTENSION_SDK_VERSION) -> ExtensionCompatibilityResult:
    try:
        current, minimum = Version(installed), Version(declared.minimum)
        maximum = Version(declared.maximum) if declared.maximum is not None else None
    except InvalidVersion:
        return ExtensionCompatibilityResult(False, "invalid SDK version")
    if maximum is not None and minimum > maximum:
        return ExtensionCompatibilityResult(False, "invalid SDK range")
    if current < minimum or maximum is not None and current > maximum:
        return ExtensionCompatibilityResult(False, "SDK version is incompatible")
    return ExtensionCompatibilityResult(True)
