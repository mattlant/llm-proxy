# Provider extension compliance suite

`llm-proxy-provider-compliance` is an independently packaged pytest suite for authors of `llm-proxy` provider extensions.

It validates the **public generic provider contract**. It does not import the gateway application, provider registry, built-in provider implementations, repository fixtures, or transports.

## What it proves

Every provider adapter is checked for the common contract:

- extension metadata and SDK compatibility;
- immutable configuration;
- factory construction;
- isolation between independently configured instances;
- normalized completion behavior;
- typed provider failures;
- idempotent cleanup;
- installed package identity where applicable.

Optional capability checks activate only when the extension declares the capability.

| Capability | Generic proof |
| --- | --- |
| Always | metadata, SDK range, immutable configuration, construction, isolation, completion, typed error, cleanup |
| `streaming` | canonical event ordering and normalized events |
| `cancellation` + streaming | provider-owned deterministic cancellation probe |
| `health` | canonical `ProviderHealth` probe |
| `management_commands` | immutable descriptors, JSON-compatible schemas/results, exact instance targeting |
| `model_listing` | listing gateway, immutable normalized records, duplicate isolation |

Compatible wire protocols are declared through `ProviderCapabilities.wire_protocols` and `WireProtocolCapability`. The provider owns protocol-specific payload, header, relay, and transport behavior; those details remain provider-owned tests rather than generic compliance assertions.

## Install for development

From the repository root, the normal development setup installs the compliance package alongside the gateway test dependencies:

```bash
python -m pip install -e '.[test]' -e ./extensions/provider_compliance
```

An external provider project can install the published/built compliance distribution as a test dependency instead.

Python 3.12.3 or newer is required.

## Add the suite to a provider

Create a deterministic `ComplianceAdapter` in the provider's tests, expose it through a `provider_compliance_adapter` fixture, and import the reusable suite:

```python
import pytest

from llm_proxy_provider_compliance import ComplianceAdapter
from llm_proxy_provider_compliance.suite import *  # noqa: F403


@pytest.fixture
def provider_compliance_adapter() -> ComplianceAdapter:
    return adapter
```

The adapter supplies the evidence the generic suite needs:

- valid immutable provider configurations;
- independently constructed gateway instances;
- canonical completion and streaming cases;
- expected normalized observations;
- typed error cases;
- package module/distribution identity for installed extensions;
- deterministic isolation and cleanup probes;
- optional health, cancellation, management, and model-listing probes.

The harness does not inspect provider internals to manufacture proof. Isolation, cancellation, cleanup, and provider-specific behavior remain provider-owned deterministic observations.

## Run

Run the provider's compliance test module with pytest:

```bash
pytest path/to/test_compliance.py
```

Keep protocol/mapping/transport tests alongside it. Passing generic compliance does not claim that a provider's native API mapping is correct.

## Clean installed-package proof

This package includes an offline installed-wheel smoke verifier:

```bash
python extensions/provider_compliance/scripts/verify_clean_install.py \
  --wheelhouse /path/to/wheelhouse
```

The wheelhouse must already contain the runtime/test dependencies required for an offline install.

The verifier:

1. builds wheels for the gateway, deterministic mock provider, and compliance package;
2. creates a temporary virtual environment;
3. installs the built packages without using the repository checkout;
4. discovers the mock provider through the real `llm_proxy.providers` entry point;
5. exercises the installed compliance observation path.

This is intentionally separate from ordinary in-tree pytest execution.

## Scope boundary

The compliance suite proves the contract shared by all provider extensions.

It does **not** prove:

- a provider's native HTTP/API protocol;
- request or response mapping specific to that provider;
- scenario semantics;
- transport correctness;
- provider-specific retry policy;
- undeclared capabilities.

Those remain tests owned by the provider package.
