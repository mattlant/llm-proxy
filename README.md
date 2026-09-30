# llm-proxy

**One local gateway. Different clients, providers, models, and protocols.**

`llm-proxy` is a protocol-neutral local LLM gateway for OpenAI-compatible, Ollama, and Anthropic Messages clients.

It sits between clients and model providers so model names, aliases, routing, policy, protocol translation, compatible wire forwarding, diagnostics, and provider-specific behavior can be managed in one place instead of being baked into every client configuration.

## Why I built it

This started as a practical compatibility problem.

Some AI tooling assumes a particular API shape, model name, or provider convention. That is convenient when your setup matches those assumptions and considerably less convenient when it does not. I wanted the client to talk to a stable local endpoint while the gateway decided what model and provider should actually handle the request.

Then local LLMs became good enough to use seriously.

My setup grew from simple request rewriting into multiple local models, OpenAI-compatible inference servers, Ollama, Claude Code, model aliases, per-model capabilities, routing policies, and provider extensions. At that point, a pile of client-specific workarounds was no longer the right abstraction.

`llm-proxy` became the boundary between **what the client asks for** and **how the request is actually executed**.

The same local inference environment also led to [power-control](https://github.com/mattlant/power-control), which manages waking, readiness, leases, power profiles, and suspend for high-power hosts. The projects are complementary: `llm-proxy` handles model/protocol execution while `power-control` handles machine availability. Direct integration is planned, but neither project requires the other today.

## What it does

`llm-proxy` currently provides:

- OpenAI Chat Completions compatibility;
- Ollama chat, generate, model-listing, and model-show compatibility;
- Anthropic Messages compatibility for supported request semantics;
- configurable model profiles and interface-scoped aliases;
- provider selection and policy-driven routing;
- canonical cross-protocol execution when translation is required;
- compatible raw-wire forwarding when a provider declares that capability;
- provider passthrough with stable `provider::upstream-model` identities;
- model capability declarations such as structured-output support;
- startup-discovered provider and capability extensions;
- authenticated management APIs and an optional web management UI;
- hot reload for supported request-time configuration;
- operational, file, structured-payload, and raw-payload diagnostics.

The goal is not to pretend every LLM protocol means exactly the same thing. Where semantics differ, the gateway either maps them deliberately, preserves a compatible provider wire contract, or rejects behavior it cannot represent honestly.

## Architecture

At a high level:

```text
OpenAI client ────────┐
Ollama client ────────┼──> llm-proxy ──> model resolution / policy
Anthropic client ─────┘                      │
                                             ├──> canonical execution
                                             └──> compatible wire forwarding
                                                       │
                                                       ▼
                                                  provider gateway
                                                       │
                                                       ▼
                                                local / remote LLM
```

Model resolution, aliases, policy evaluation, provider selection, and request semantics are application concerns. Provider extensions own provider-specific construction and transport behavior.

The built-in Ollama provider can use an OpenAI-compatible upstream interface where configured. Additional providers can be installed through the public provider-extension entry point.

See [Extension development](docs/extensions.md) for the extension model.

## Requirements

- Python **3.12.3 or newer**
- A configured upstream LLM provider
- Node.js/npm only when building the optional management frontend assets

The project is currently installed from source; it is not dependent on a system service or platform-specific installer.

## Install

Clone the repository, then from the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install .
```

On Windows, activate the virtual environment using the normal Windows activation command instead.

### Management-enabled distribution

The management API is part of the Python application. The web UI is generated frontend content and is **not committed to the repository**.

If you want an installed distribution that serves the management UI at `/management/`, build the frontend assets before building or installing the Python distribution:

```bash
cd management_frontend
npm ci
npm run build:package
cd ..

python -m pip install .
```

For frontend development and verification details, see [management_frontend/README.md](management_frontend/README.md).

Without generated management assets, the core proxy can still be installed and run, but the management web frontend is unavailable.

## Configure

Start with the portable example:

```bash
cp config.example.yaml config.yaml
```

The runtime default is `./config.yaml` in the current working directory. To use another file:

```bash
export LLM_PROXY_CONFIG=/path/to/config.yaml
```

Runtime configuration is never loaded from beside the installed Python package.

The public repository does not include `config.yaml`; create it locally from [config.example.yaml](config.example.yaml) when needed.

Configuration covers:

- server binding, timeouts, reload interval, logging, and payload tracing;
- management API/UI settings;
- enabled client interfaces;
- providers and provider-specific configuration;
- model profiles, aliases, parameters, and compatibility;
- provider-passthrough behavior;
- policies;
- installed capability extensions.

A valid request-time configuration change is atomically reloaded. Provider/extension activation and other startup-owned settings require restart. Invalid edits leave the previous valid configuration active.

## Run

With a valid `config.yaml` in the current directory:

```bash
llm-proxy
```

You can also run the module directly:

```bash
python -m llm_proxy
```

The example configuration listens on:

```text
http://127.0.0.1:11435
```

Health information is available at:

```text
GET /_proxy/health
```

The health response identifies the active configuration path, enabled interfaces and providers, and exposed model profiles. Authentication tokens are not returned.

## Supported client routes

| Interface | Route | Notes |
| --- | --- | --- |
| OpenAI | `POST /v1/chat/completions` | JSON or SSE |
| OpenAI | `GET /v1/models` | Configured and eligible passthrough models |
| Ollama | `POST /api/chat` | JSON or NDJSON |
| Ollama | `POST /api/generate` | JSON or NDJSON |
| Ollama | `GET /api/tags` | Configured and eligible passthrough models |
| Ollama | `POST /api/show` | Profile/alias based |
| Anthropic | `POST /v1/messages` | JSON or SSE |
| Proxy | `GET /_proxy/health` | Runtime diagnostics |

Unsupported provider APIs are not silently proxied. For example, embeddings, OpenAI Responses, and Anthropic batch/files/token-counting APIs are currently outside the supported surface.

## Models, aliases, and provider passthrough

A model profile defines the client-facing model identity and its execution contract:

```text
client model
    ↓
profile / alias resolution
    ↓
policy evaluation
    ↓
provider instance
    ↓
upstream model
```

Aliases are scoped to an interface. A model can therefore present names appropriate to different clients without duplicating the underlying provider configuration.

By default, requests resolve through configured model profiles. Setting:

```yaml
model_access:
  mode: provider_passthrough
```

also permits eligible direct `provider::upstream-model` identities. Configured profiles and aliases still take precedence.

Provider model listing is opt-in per provider instance and is independent of routing.

## Policies

Policies operate on canonical request context and can match on model identity, interface, visible text, tool names, metadata, and wildcard/substring model or tool rules.

Matching policies run in configuration order. Later actions can override earlier model, provider, or selected parameter decisions.

Policies can also invoke installed bounded capabilities before provider dispatch. Required capability failure prevents dispatch; optional behavior must be explicitly configured as best-effort.

The portable example contains a small policy example. See [config.example.yaml](config.example.yaml) for current configuration shape.

## Anthropic and Claude Code

The Anthropic Messages interface is intended to let supported Anthropic clients use models behind `llm-proxy` without pretending unsupported Anthropic semantics exist.

For Claude Code, point the client at the proxy and select aliases exposed through the Anthropic interface:

```bash
export ANTHROPIC_BASE_URL="http://127.0.0.1:11435"
export ANTHROPIC_AUTH_TOKEN="<configured-or-any-non-empty-token>"
export ANTHROPIC_MODEL="<anthropic-model-alias>"
```

The adapter supports streaming text, native tool use, tool-result continuation, selected request controls, and explicitly supported structured-output behavior. Semantics that cannot be represented truthfully are rejected or handled as documented no-ops rather than fabricated.

## Management

Enable the management plane in configuration and provide the token named by `management.token_env` (default `LLM_PROXY_ADMIN_TOKEN`).

The management API is exposed under:

```text
/_admin/v1
```

It requires bearer authentication even on loopback. Non-loopback management requires `management.allow_remote: true` explicitly.

Management supports inspection and configuration updates using revision checks, plus provider-owned management commands where a provider advertises them. Provider secrets/configuration are redacted from inspection responses.

When frontend assets were built into the installed distribution, the web UI is served at:

```text
/management/
```

## Logging and diagnostics

Normal operational logging is deliberately bounded and does not include prompts, tool schemas, credentials, or raw request/response bodies.

Optional file logging writes the same operational stream to per-launch files.

For deeper debugging, payload tracing can record structured or raw application-visible HTTP data. Raw tracing may include credentials, prompts, completions, and reasoning content.

**Treat payload tracing as sensitive diagnostic output and enable it only in a controlled environment.**

See [config.example.yaml](config.example.yaml) for the current logging and tracing settings.

## Extensions

There are two extension surfaces:

- **provider extensions** add provider gateways through the `llm_proxy.providers` entry-point group;
- **capability extensions** add bounded policy-invoked capabilities through the `llm_proxy.extensions` entry-point group.

Extensions are activated at startup and declare compatibility with the public SDK surface they consume.

The repository also contains a separately packaged provider-compliance suite for extension authors.

See:

- [Extension development](docs/extensions.md)
- [Provider compliance suite](extensions/provider_compliance/README.md)

## Development

Install the development/test dependencies plus the separately packaged provider-compliance project:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[test]' -e ./extensions/provider_compliance
```

Run the full Python suite from the repository root:

```bash
PYTHONPATH=tests .venv/bin/python -m pytest -q -p conftest
```

Frontend development and verification live under [management_frontend](management_frontend/README.md).

## Project status

`llm-proxy` is actively developed and used against real local inference workloads.

The current focus is making heterogeneous clients and local models behave predictably through one stable gateway while keeping provider-specific behavior explicit instead of hiding it behind accidental compatibility.

The project is pre-1.0. Protocol and extension surfaces are intentionally tested, but they should not yet be treated as permanently frozen public APIs.

## Related project

[power-control](https://github.com/mattlant/power-control) manages power state for Linux compute hosts: wake, readiness, power profiles, suspend-prevention leases, automatic idle suspend, and remote suspend.

It grew out of the same local inference setup. The longer-term direction is for inference orchestration to combine:

```text
request arrives
    ↓
ensure inference host is awake and ready
    ↓
hold availability while work is active
    ↓
llm-proxy routes and executes the request
    ↓
release availability
    ↓
idle power policy takes over
```

That integration is not required for `llm-proxy` today.

## License

`llm-proxy` is released under the [MIT License](LICENSE).
