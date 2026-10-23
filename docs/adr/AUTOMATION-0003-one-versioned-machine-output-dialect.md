# ADR-AUTOMATION-0003: Use One Versioned Machine-Output Dialect

Date: 2026-05-19

## Status

Accepted

## Context

Agents and automation need stable output that does not depend on Rich
presentation, terminal behavior, or command-specific error shapes. Streaming
commands additionally need progress events without losing one unambiguous
terminal result.

The contract first shipped in commit `dffa5c17` through PR #455 and is
maintained in [`json-output.md`](../json-output.md). The repository now uses
the same renderer and schemas beyond the original `run` surface.

## Decision

Machine-readable commands use the CLI-wide versioned renderer dialect:

- JSON mode writes one terminal `envelope/1` object to stdout.
- NDJSON mode writes versioned `event/1` objects followed by exactly one
  terminal `envelope/1` object.
- `type` is the discriminator, and registered `error.code` values carry
  machine-readable failure categories.
- Registered error codes are append-only and are never repurposed.
- Human diagnostics and output that cannot satisfy the contract stay off the
  machine-readable stdout channel.
- Consumers tolerate unknown fields and event or error variants within the
  documented compatibility rules.

## Alternatives considered

- Maintain a separate `run --json` dialect. Rejected because every consumer
  would need command-specific parsing and error handling.
- Communicate fine-grained failures through exit codes. Rejected because exit
  codes cannot carry structured evidence and are constrained by process
  conventions.
- Stream events without a final envelope. Rejected because interruption and
  terminal failure would be ambiguous to consumers.

## Consequences

### Positive

- Agents can use one parser and one terminal-result model across commands.
- Streaming progress and final success or failure remain distinguishable.
- Schema and error registries provide discoverable compatibility boundaries.

### Negative

- Commands must route incidental human output away from stdout in machine
  modes.
- Breaking dialect changes require explicit versioning and migration guidance.

## Validation

`comfy_cli/output/renderer.py` owns JSON and NDJSON emission,
`comfy_cli/schemas/envelope.json` defines the terminal envelope, and
`comfy_cli/schemas/error_codes.md` documents the error-code contract.
`tests/comfy_cli/command/test_run_json.py` pins the run-stream contract, while
error-code registry tests prove that emitters and registrations stay aligned.
