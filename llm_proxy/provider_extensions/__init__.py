"""Supported public contract for externally packaged provider extensions.

This namespace is intentionally separate from the gateway's application and
provider implementation packages. It is the only ``llm_proxy``
namespace an external provider extension may import.
"""

from .contracts import (
    PROVIDER_SDK_VERSION,
    ProviderListedModel, ProviderListingError, ProviderListingFailureCategory, ProviderModelListingGateway, ProviderHealthGateway,
    CompletionEvent,
    CompletionExecution,
    CompletionGateway,
    ProviderExecutionPolicy,
    SourceInterface,
    CompletionResponse,
    ProviderCapabilities,
    ProviderExtension,
    ProviderExtensionMetadata,
    ProviderFactory,
    ProviderHealth,
    ProviderHealthStatus,
    ProviderInstanceConfig,
    SdkCompatibility,
    ProviderSemanticDeclaration, ProviderSemanticDisposition, ProviderSemanticExtension, provider_semantic_declaration,
)
from .compatibility import CompatibilityResult, validate_sdk_compatibility
from .wire import HeaderPairs, WireExecution, WireGateway, WirePatch, WirePatchOperation, WireProtocolCapability, WireRequest, WireRequestProjection, WireResponse, WireStream
from .management import ManagementCommandContext, ManagementCommandDescriptor, ManagementCommandMutability, ManagementCommandPermission, ProviderManagementCommandExecutor
from llm_proxy.domain.content import TextContent, ToolCallContent
from llm_proxy.domain.errors import ProviderUnavailableError
from llm_proxy.domain.events import ResponseCompleted, ResponseStarted, TextCompleted, TextDelta, TextStarted, ToolCallArgumentsDelta, ToolCallCompleted, ToolCallStarted
from llm_proxy.domain.messages import Message, MessageRole
from llm_proxy.domain.responses import FinishReason, Usage

__all__ = [
    "PROVIDER_SDK_VERSION", "CompletionEvent", "CompletionExecution", "CompletionGateway", "ProviderExecutionPolicy", "SourceInterface", "ProviderListedModel", "ProviderListingError", "ProviderListingFailureCategory", "ProviderModelListingGateway", "ProviderHealthGateway",
    "CompletionResponse", "ProviderCapabilities", "ProviderExtension", "ProviderExtensionMetadata",
    "ProviderFactory", "ProviderHealth", "ProviderHealthStatus", "ProviderInstanceConfig",
    "SdkCompatibility", "CompatibilityResult", "validate_sdk_compatibility", "FinishReason", "Message", "MessageRole", "ResponseCompleted",
    "ProviderSemanticDeclaration", "ProviderSemanticDisposition", "ProviderSemanticExtension", "provider_semantic_declaration",
    "ResponseStarted", "TextCompleted", "TextContent", "TextDelta", "TextStarted", "ToolCallContent", "ToolCallStarted", "ToolCallArgumentsDelta", "ToolCallCompleted", "Usage",
    "ProviderUnavailableError",
    "ManagementCommandContext", "ManagementCommandDescriptor", "ManagementCommandMutability", "ManagementCommandPermission", "ProviderManagementCommandExecutor",
    "HeaderPairs", "WireExecution", "WireGateway", "WirePatch", "WirePatchOperation", "WireProtocolCapability", "WireRequest", "WireRequestProjection", "WireResponse", "WireStream",
]
