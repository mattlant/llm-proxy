# Extension development

`llm-proxy` supports independently packaged extensions discovered through Python entry points.

There are two extension families with deliberately separate contracts:

| Extension type | Entry-point group | Public import surface | Purpose |
| --- | --- | --- | --- |
| Provider extension | `llm_proxy.providers` | `llm_proxy.provider_extensions` | Add a provider gateway and provider-owned capabilities |
| Capability extension | `llm_proxy.extensions` | `llm_proxy.extensions` | Add bounded policy-invoked capabilities |

Extensions are discovered and activated at application startup. Discovery and construction are not request-path operations.

## General rules

An external extension should:

- target Python 3.12.3 or newer;
- be independently installable;
- use only the public SDK namespace for its extension family;
- declare SDK compatibility explicitly;
- expose an entry point rather than requiring repository-relative imports;
- keep instance state isolated between separately configured instances;
- treat activation/configuration as startup-owned unless the extension contract explicitly states otherwise.

Core configuration selects installed extensions by identifier. Installing a package alone does not silently route requests through it.

## Provider extensions

Provider extensions implement model execution behind the gateway.

They are registered through:

```toml
[project.entry-points."llm_proxy.providers"]
my_provider = "my_package.extension:extension"
```

Provider packages import from:

```python
from llm_proxy.provider_extensions import ...
```

They should not depend on application internals, built-in provider implementations, test fixtures, or repository layout.

A provider extension exposes metadata, an SDK compatibility range, capabilities, and a factory that creates configured provider instances.

Conceptually:

```text
installed provider package
    ↓
llm_proxy.providers entry point
    ↓
ProviderExtension
    ├── metadata
    ├── compatibility
    ├── capabilities
    └── factory
            ↓
       configured gateway instance
```

Each configured instance owns its provider-specific state. Factories must not accidentally share mutable runtime state across instances.

### Provider capabilities

`ProviderCapabilities` advertises behavior the provider actually implements. Depending on the extension, capabilities may include:

- completion;
- streaming;
- native and parallel tools;
- usage reporting;
- cancellation;
- health;
- management commands;
- model listing;
- compatible wire protocols.

Declared capabilities are contracts, not hints. A provider should advertise a capability only when its gateway implements the corresponding behavior.

### Model listing

A listing-capable provider implements `ProviderModelListingGateway` and returns immutable `ProviderListedModel` records.

Listing is separate from routing. The application exposes dynamic provider models only when provider passthrough is enabled and the configured provider instance explicitly enables listing.

Provider adapters own native listing-response parsing and surface typed `ProviderListingError` failures.

### Compatible wire protocols

A provider may declare a versioned `WireProtocolCapability`.

Compatible wire execution lets protocol-specific request/response handling remain provider-owned when the provider can preserve that wire contract. It does not allow core to infer compatibility from a provider name or URL.

A declared wire capability is implemented through `WireGateway`. Provider-specific payload, header, relay, and transport behavior remains the provider's responsibility and should be covered by provider-owned tests.

### Management commands

Providers may expose immutable management-command descriptors and implement management execution.

Commands are always targeted to a configured provider instance. Descriptors, input/output schemas, permissions, and results must remain JSON-compatible and deterministic at the public boundary.

The generic compliance suite verifies the shared contract; provider-specific command semantics remain provider-owned.

## Capability extensions

Capability extensions add bounded behavior that policy can invoke before provider dispatch.

They are registered through:

```toml
[project.entry-points."llm_proxy.extensions"]
my_extension = "my_package.extension:extension"
```

Capability packages import from:

```python
from llm_proxy.extensions import ...
```

The current public capability family is `side_effect@1`.

A side-effect capability receives an immutable, payload-free `SideEffectContext`. It does not receive prompts, raw requests, provider credentials, or mutable application internals.

Policy bindings are required by default. A required timeout or failure prevents provider dispatch. Best-effort behavior must be selected explicitly.

Core does not turn a side-effect capability into:

- model routing;
- request rewriting;
- provider construction;
- retries;
- shell execution;
- arbitrary access to internal application state.

Capability-extension activation is startup-owned. Policy binding arguments may reload after configuration validation.

## SDK compatibility

Both extension families declare compatibility with their corresponding public SDK contract.

Provider extensions use `SdkCompatibility` and the provider SDK version exposed by `llm_proxy.provider_extensions`.

Capability extensions use `ExtensionSdkCompatibility` and the extension SDK version exposed by `llm_proxy.extensions`.

Keep the declared range honest. Compatibility metadata is intended to fail clearly when an installed extension and gateway SDK no longer agree.

## Testing provider extensions

Provider-specific protocol, mapping, scenario, and transport tests belong to the provider package.

For the shared provider contract, use the independently packaged [provider compliance suite](../extensions/provider_compliance/README.md).

The compliance suite checks generic obligations such as metadata, SDK compatibility, configuration immutability, instance isolation, normalized completion behavior, typed errors, cleanup, and any optional capabilities the provider declares.

It intentionally does not prove provider-specific protocol fidelity.

## Example provider

The repository contains a deterministic mock provider used to exercise the public provider SDK and compliance boundary.

Its package is independently buildable and uses only the public provider-extension surface. See [the mock provider README](../extensions/mock_provider/README.md) for its package-specific design.

## Packaging checklist

Before treating an extension as independently distributable, verify that:

- its `pyproject.toml` declares Python and `llm-proxy` compatibility;
- its license metadata and license file are present;
- its entry point resolves from an installed wheel;
- no repository-relative imports are required;
- package metadata matches extension metadata;
- separate configured instances do not share mutable state;
- declared optional capabilities have deterministic tests;
- the extension works outside the repository checkout.
