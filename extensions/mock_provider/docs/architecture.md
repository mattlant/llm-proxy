# Architecture

The deterministic mock provider is deliberately structured like an external provider package.

Its dependency boundary is:

```text
llm_proxy_mock_provider
        │
        └──> llm_proxy.provider_extensions
                 public provider SDK only
```

It does not import the gateway application, provider registry, built-in provider implementation, test fixtures, or repository-relative modules.

## Construction

The installed `llm_proxy.providers` entry point exposes one extension object.

The extension declares metadata, SDK compatibility, capabilities, and a `DeterministicMockFactory`. The factory creates a new gateway for each configured provider instance.

```text
entry point
    ↓
DeterministicMockExtension
    ↓
DeterministicMockFactory
    ↓
gateway instance
```

Construction parses and validates the instance's JSON-compatible scenario. Runtime behavior does not depend on external scenario files or network lookups.

## State ownership

Each gateway instance owns its own:

- immutable parsed scenario;
- rule/cursor state;
- captures;
- expectations;
- async lock.

No mutable execution state is shared between configured instances.

That isolation is part of the provider-extension contract and is exercised by the generic compliance suite.

## Concurrency

The async lock protects reservation of deterministic invocation state.

Only the state-selection/reservation step occurs under the lock. Simulated latency and response/event emission occur after the lock is released.

This keeps scenario ordering deterministic without holding the lock across artificial delays or stream production.

## Public-contract focus

The package is intended to prove that an independently installed provider can exercise the same public contracts as a real provider:

- canonical completion and streaming;
- tools and usage;
- typed errors;
- cancellation;
- management commands;
- model listing;
- package identity and SDK compatibility.

Provider-compliance tests observe those contracts through public interfaces rather than reaching into the mock provider's private state.

