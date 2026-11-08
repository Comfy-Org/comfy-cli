# comfy-cli error codes

The accepted compatibility decision for envelopes and registered error codes is
recorded in
[`ADR-AUTOMATION-0003`](../../docs/adr/AUTOMATION-0003-one-versioned-machine-output-dialect.md).

Stable machine identifiers for `error.code` in the JSON envelope. The
canonical list lives in `comfy_cli/error_codes.py` — this file is for humans.
**Append-only** — never repurpose an existing code.

For programmatic access, agents call:

```bash
comfy --json discover | jq '.data.error_codes[]'
```

## Conventions

- `code` is the contract; agents branch on it.
- `message` is human-readable; phrased so it can be shown to users verbatim.
- `hint` is one actionable next step. Always set a hint when there is one.
- `details` carries structured context (`status`, `path`, `host`,
  `close_matches`, `valid_inputs`, etc.) so agents can self-correct.

## Stopped builds

`build_stopped` means `comfy build release create --watch` observed the terminal
release status `stopped`. It exits 1, including when other targets are ready or
failed. Its details contain `releaseId`, `status: "stopped"` and the received
`artifactCounts`. Read the release and its target logs before deciding what to do
next. Successfully reading logs, including following a stopped release, exits 0.

## Tests

Three tests pin the contract:

- Every `renderer.error(code="…")` site is registered.
- Every registered code is raised somewhere or explicitly retired in the
  registry test. Retired codes stay discoverable for compatibility and must
  no longer be emitted. `asset_upload_timeout` is retired because model
  uploads now wait until storing ends instead of stopping after fifteen minutes.
- A code emitted only as a warning (an object in a `warnings` list of a
  successful result) is not registered. Registered codes are failure
  categories; a warning keeps its `code` string in `warnings[]`. Warning-only
  codes registered before this rule stay registered, since codes are
  append-only.

A code goes into `error_codes.py` *before* it lands in any call site.
