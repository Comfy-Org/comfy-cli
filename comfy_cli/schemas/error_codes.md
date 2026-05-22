# comfy-cli error codes

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

## Tests

Two tests pin the contract:

- Every `renderer.error(code="…")` site is registered.
- Every registered code is raised somewhere (no dead codes).

A code goes into `error_codes.py` *before* it lands in any call site.
