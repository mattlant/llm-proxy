# Deterministic mock provider

`llm-proxy-mock-provider` is an independently packaged provider extension used to test the public `llm-proxy` provider SDK with deterministic behavior.

It is intentionally a real extension package rather than a core-internal test double: it installs through normal Python packaging, registers through the `llm_proxy.providers` entry-point group, imports only the public provider-extension SDK, and can be exercised outside the repository checkout.

## Purpose

The package provides repeatable provider behavior for:

- provider-extension discovery and activation;
- canonical completion and streaming;
- native/parallel tool behavior;
- typed failures;
- cancellation;
- management commands;
- model listing;
- package and SDK compatibility;
- two-instance isolation;
- the reusable provider-compliance suite.

It is test infrastructure, not a general-purpose inference provider.

## Install

From this directory:

```bash
python -m pip install .
```

Python 3.12.3 or newer is required.

The installed distribution registers:

```text
llm_proxy.providers
    deterministic_mock
        → llm_proxy_mock_provider.extension:extension
```

No repository-relative imports are required.

## Configuration

Each configured provider instance accepts a JSON-compatible `scenario` document.

A scenario uses `version: 1` and may contain:

- `rules`;
- `defaults`;
- `expectations`;
- `models`.

The `models` array supplies normalized model-listing results when the configured provider instance enables `listing_enabled`.

Scenario state is parsed into immutable configuration at construction time. Runtime invocation state belongs to the created gateway instance.

The package does not load scenario files by path and does not perform network access to resolve scenario configuration.

## Instance isolation

Every factory-created gateway owns its own:

- parsed scenario;
- rule/cursor state;
- captures;
- expectations;
- async lock.

Two configured instances must not share mutable scenario state.

Invocation reservation occurs while holding the instance lock. Simulated latency and event emission occur after the lock is released so deterministic sequencing does not unnecessarily serialize the rest of execution.

See [Architecture](docs/architecture.md) for the implementation boundary.

## Capabilities

The extension currently advertises deterministic support for:

- completion;
- streaming;
- native tools;
- parallel tools;
- usage;
- cancellation;
- management commands;
- model listing.

The package metadata also declares its supported `llm-proxy` SDK range.

Capabilities are intentionally exercised through public contracts rather than private gateway internals.

## Management commands

Mock-provider management commands operate only on the targeted configured instance.

They are used to prove descriptor/schema behavior and exact instance targeting through the same public provider-management contract available to external extensions.

## Testing

Run the package's own tests from this directory using its normal development environment.

The repository also uses the independently packaged [provider compliance suite](../provider_compliance/README.md) to validate the generic provider contract.

Provider-specific scenario behavior remains covered by this package's own tests; generic compliance is not a substitute for scenario tests.

## Build

Build the standalone distribution from this directory:

```bash
python -m build
```

The resulting wheel is self-contained with respect to the mock-provider package and resolves its gateway contract through the installed `llm-proxy` dependency and public provider SDK.

