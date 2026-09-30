# Management frontend

The management frontend is the optional browser UI for the authenticated `llm-proxy` management plane.

It is a private npm workspace used to build the static assets that the Python application serves at `/management/`. The generated assets are package data; they are intentionally **not committed** to the repository.

## Workspace

The frontend contains one application plus shared internal packages:

```text
management_frontend/
├── apps/
│   └── core-management
├── packages/
│   ├── management-sdk
│   ├── admin-api-client
│   └── ui-primitives
└── scripts/
```

Runtime reads and mutations go through the authenticated `/_admin/v1` API. Frontend packages do not import Python implementation code.

## Install dependencies

From this directory:

```bash
npm ci
```

## Build

Build all frontend workspaces:

```bash
npm run build
```

To produce assets for the Python distribution:

```bash
npm run build:package
```

`build:package`:

1. builds the SDK, admin client, UI primitives, and core management app;
2. validates the application's asset manifest;
3. requires `index.html` and a content-hashed JavaScript asset;
4. replaces `../llm_proxy/static/management` with the validated build output.

The destination is consumed by Python packaging as `llm_proxy` package data.

If the frontend assets are not built before producing/installing a Python distribution, the proxy itself can still run, but the management web UI is unavailable.

## Distribution contract

A management-enabled Python distribution is produced by building the frontend assets first and then building/installing the Python package from the repository root.

For example:

```bash
cd management_frontend
npm ci
npm run build:package
cd ..

python -m pip install .
```

The repository's distribution verifier exercises the stronger end-to-end path:

```bash
npm run verify:distribution
```

That invokes `../scripts/verify_management_distribution.py`, builds through the Python distribution path, installs the result in isolation, and verifies that the management index and referenced hashed asset are served.

## Development verification

Available verification modes are:

```bash
npm run verify:focused
npm run verify:warm
npm run verify:clean
```

### Focused

Runs a reduced developer feedback loop around the management SDK, admin client, and core management application.

### Warm

Runs type checking, tests, lint, build, and the browser test using the current installed dependency state.

### Clean

Runs the workspace clean step, performs `npm ci --ignore-scripts`, then executes the same full validation path as warm verification.

The clean path is the broad frontend workflow and includes the live browser test. Use `verify:distribution` when the question is specifically whether the generated frontend survives Python packaging and is served correctly from an installed distribution.

Individual commands are also available:

```bash
npm run typecheck
npm test
npm run lint
npm run build
npm run test:browser
```

## Authentication and session model

The frontend uses the management plane's existing bearer-token authentication.

The browser session is deliberately ephemeral:

- the token is entered by the user;
- it is cleared from the form immediately;
- it is not passed through the component tree as a prop;
- operational views receive an immutable management context and typed client;
- refreshing the page ends the in-memory session.

The frontend does not create a second authentication mechanism.

## API ownership

The UI consumes only the public management HTTP boundary under `/_admin/v1`.

The Python management API owns configuration revision checks, redaction, restart-required reporting, provider-command validation, and management authorization. The frontend represents those contracts; it does not duplicate their authority.

## Generated assets

Production builds emit a validated, content-hashed asset set and manifest under the core-management app's `dist` directory.

`npm run build:package` copies those assets to:

```text
llm_proxy/static/management
```

Both locations are generated state. Do not commit generated frontend output merely to make a local Python build serve the UI.

