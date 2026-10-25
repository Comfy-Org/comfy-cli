# ADR-DEPENDENCIES-0001: Delegate Unified Dependency Resolution to ComfyUI-Manager

Date: 2026-03-27

## Status

Accepted

## Context

ComfyUI-Manager already owns `UnifiedDepResolver` and its `uv pip compile`
behavior. comfy-cli needed to expose unified dependency resolution without
creating a second implementation that could drift from the manager.

This decision was originally recorded in
[`DESIGN-uv-compile.md`](../DESIGN-uv-compile.md) and shipped in commit
`0cb2cb8a` through PR #363. The current command implementation and tests still
enforce that boundary.

## Decision

comfy-cli owns command-line selection, configuration, conflict validation, and
process invocation. It delegates unified dependency resolution by passing
`--uv-compile` or invoking `uv-sync` in `cm_cli`; it does not import or
reimplement `UnifiedDepResolver`.

The existing comfy-cli `--fast-deps` path remains a separate mechanism.

## Alternatives considered

- Import `UnifiedDepResolver` into comfy-cli. Rejected because it couples the
  CLI to manager internals and creates shared lifecycle and compatibility work.
- Reimplement unified resolution in comfy-cli. Rejected because it creates two
  authorities for dependency semantics.

## Consequences

### Positive

- Resolver behavior and upgrades remain owned by ComfyUI-Manager.
- comfy-cli remains a thin, testable command and policy layer.

### Negative

- `--uv-compile` depends on a compatible ComfyUI-Manager version.
- Manager errors and compatibility behavior cross the subprocess boundary.

## Validation

`comfy_cli/command/custom_nodes/cm_cli_util.py` contains the pass-through, and
the custom-node command tests assert forwarding, precedence, and mutual
exclusion behavior.
