from llm_proxy.extensions import CapabilityRegistration, ExtensionInstanceConfig, ExtensionMetadata, ExtensionSdkCompatibility, SIDE_EFFECT_FAMILY, SIDE_EFFECT_VERSION, SideEffectError, SideEffectFailureCategory

class Capability:
    def validate_arguments(self, arguments):
        if arguments.get("mode") not in {None, "success", "unavailable", "rejected", "failed"}: raise ValueError("invalid mode")
    async def invoke(self, context, arguments):
        mode = arguments.get("mode", "success")
        if mode == "unavailable": raise SideEffectError(SideEffectFailureCategory.UNAVAILABLE)
        if mode == "rejected": raise SideEffectError(SideEffectFailureCategory.REJECTED)
        if mode == "failed": raise SideEffectError(SideEffectFailureCategory.FAILED)

class Instance:
    registrations = (CapabilityRegistration(SIDE_EFFECT_FAMILY, SIDE_EFFECT_VERSION, "deterministic", Capability()),)
    async def aclose(self): pass

class Factory:
    def create(self, config: ExtensionInstanceConfig): return Instance()

class Extension:
    metadata = ExtensionMetadata("deterministic.mock", "Deterministic Mock", "llm-proxy-mock-capability", "0.1.0")
    compatibility = ExtensionSdkCompatibility("1.0.0", "1.0.0")
    factory = Factory()

extension = Extension()
