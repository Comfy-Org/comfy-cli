# ADR-DEPENDENCIES-0019: Load comfy-sdk Only Inside the Router Generate Path

Date: 2026-11-27

## Status

Proposed

Relation: Builds on
[ADR-DEPENDENCIES-0001](DEPENDENCIES-0001-delegate-unified-resolution-to-comfyui-manager.md).
comfy-cli still delegates unified dependency resolution to ComfyUI-Manager;
this record only governs when the CLI's own runtime dependency on `comfy-sdk`
may be loaded.

## Context

`comfy generate` is moving onto Comfy's Router service, whose first-party
Python client is `comfy-sdk`. The SDK depends on pydantic, which loads the
compiled `pydantic_core` extension. A `comfy` process that has loaded that
extension keeps the `.pyd`/`.so` open for its lifetime, and on Windows an open
DLL cannot be replaced. `comfy install` runs `uv pip install` of ComfyUI's
requirements into the same environment, so if the extension were loaded on that
path, uv could not upgrade `pydantic_core` when ComfyUI resolves a different
version (os error 5). The existing `mixpanel<5` pin exists for the same reason.

The SDK also resolves the Router host only from its own environment variable
(`ROUTER_BASE_URL_ENV_VAR`, currently `COMFY_ROUTER_BASE_URL`) at client
construction, with no base-URL argument, while comfy-cli users point the CLI at
a non-production deployment with `COMFY_API_BASE_URL`.

## Decision

- `comfy-sdk>=0.6.0` is a regular runtime dependency of comfy-cli.
- `comfy_sdk` is imported only inside a function, never at module scope in any
  module reachable from `comfy_cli.cmdline`. The single entry point is
  `router_models()` in `comfy_cli/command/generate/router_client.py`; the
  `generate` command is a `LazyCommand`, so the import is reachable only from
  the Router `generate` path.
- A missing SDK, or an installed SDK that fails to import, becomes an
  `ApiError` that names the cause and the recovery command, not a bare
  `ImportError`.
- Compatibility contract: when the SDK's Router variable is unset and
  `COMFY_API_BASE_URL` is set, `router_models()` copies the normalised
  `COMFY_API_BASE_URL` into the variable the SDK publishes, only while the
  client is constructed and under a module lock, and restores the environment
  afterwards.

## Alternatives considered

- Import the SDK at module scope like other dependencies. Rejected: it loads
  `pydantic_core` on every `comfy` invocation, including `comfy install`, and
  reintroduces the Windows locked-DLL failure.
- Make the SDK an optional extra. Rejected: `comfy generate` would fail on a
  default install, and the lazy import already removes the install-time cost.
- Vendor or reimplement the Router client. Rejected: it creates a second client
  that drifts from the first-party SDK as Router changes.
- Set the Router variable for the whole process. Rejected: later calls and
  child processes (including ComfyUI started by the CLI) would inherit a value
  the user never set.

## What we reject

- A module-scope `import comfy_sdk` (or any module that imports it at module
  scope) anywhere reachable from `comfy_cli.cmdline`.
- Hardcoding the SDK's Router variable name instead of reading the one the SDK
  publishes, which would let an SDK rename silently send traffic to production.

## Consequences

### Positive

- Installs and non-generate commands never load pydantic, so the Windows
  `comfy install` path keeps working.
- Router calls use the first-party client, so Router changes arrive with SDK
  upgrades.

### Negative

- Every new Router caller must go through the lazy accessor; a careless import
  elsewhere breaks the invariant (the guard test catches it).
- The environment bridge is a workaround tied to undocumented SDK behaviour.

### Known limits

- The SDK sends the key as `Authorization: Bearer`; Router accepting a
  `comfyui-` key that way must be confirmed end to end when the first caller
  lands.
- `router_models()` returns the SDK's `models` API, not the owning client, so
  callers cannot close its connection pool deterministically. Revisit the
  return shape when the first caller lands.

## Validation

- `tests/comfy_cli/test_tracking_lazy_import.py` fails if importing
  `comfy_cli.command.generate.app` or `router_client` loads `comfy_sdk`,
  `comfy_low` or `pydantic_core`, and checks the missing- and broken-SDK errors.
- `tests/comfy_cli/command/generate/test_router_client.py` covers the bridge's
  scoping, restore, normalisation and overlapping calls, and pins that the
  installed SDK still names `COMFY_ROUTER_BASE_URL`.

## Revisit triggers

- `comfy-sdk` exposes a base-URL argument: remove the environment bridge.
- `comfy-sdk` ships a new major version, or drops pydantic.
- Router changes how it accepts `comfyui-` keys.
- A Windows `comfy install` regression is traced to a loaded `pydantic_core`.
