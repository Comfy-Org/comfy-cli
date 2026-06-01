---
name: comfy-debug
description: Debugging skill for the comfy CLI — failed workflows, stuck jobs, error envelopes, and the fastest path from "it broke" to "fixed".
---

When something fails on the comfy CLI, **read the JSON envelope first**, act on the `hint`, and only go deeper if needed. This skill is the bottom-up playbook.

## The envelope is the source of truth

Every command emits the standard JSON envelope (see core skill). When `error` is present, map `error.code` to a fix below.

Three rules:

1. **Map `error.code` to a fix below.** Most codes have a one-line resolution.
2. **`hint` is authoritative.** If it says "run: comfy launch", that's the fix. Don't paraphrase, don't guess.
3. **`details` carries the why.** Look here before scrolling logs.

## Decision tree for common failures

### `server_not_running` (local)
```
comfy launch              # start the local server in the foreground
# or, for background mode:
comfy launch --background
```
If `comfy launch` itself fails: check `comfy --json env` for the resolved interpreter and workspace, and inspect whatever stdout `comfy launch` was attached to.

### `cloud_not_configured` / `cloud_unauthorized` (cloud)
```
comfy cloud login                   # opens browser, OAuth + PKCE
# or, if you already have an API key:
comfy cloud set-key --key sk-…      # alternative path; API key wins over OAuth
comfy --json cloud whoami           # confirm signed_in: true (auth_method is "oauth" or "api_key")
```
If you're on a custom env (PR preview), set the base URL first:
```
comfy cloud set-base-url https://fe-pr-NNNNN.testenvs.comfy.org
```

### `workflow_not_api_format`
The CLI received a UI-exported workflow (the one with `nodes:[…]`, `links:[…]`) instead of the API format. Fix from ComfyUI: **File → Export (API)**. Or, if a local server is running, the CLI will try `/workflow/convert` automatically — if that 404s, you need ComfyUI's API export.

### `workflow_invalid_json`
The file isn't valid JSON. Inspect the first/last 100 bytes — often it's an HTML error page that was saved with `.json`.

### `prompt_rejected` (server returned `node_errors`)
The server validated the workflow and rejected nodes. The full per-node error map is in `details.node_errors`. The two most common causes:

- **Unknown class_type** → the cloud or local server lacks the custom node. Run:
  ```
  comfy --json nodes search MissingNodeName
  ```
  If `data.count` is 0, the node is genuinely absent. Then either install via `comfy node install <pkg>` (local) or pick a different workflow (cloud).

- **Missing model file** (`ckpt_name`, `lora_name`, `vae_name` not found) → confirm the loader node exists and see its choices:
  ```
  comfy --json nodes show CheckpointLoaderSimple   # inputs[].choices lists available checkpoints
  comfy --json env                                  # see workspace/models path
  ```

### `cloud_http_error` (HTTP 4xx/5xx from cloud)
`details.status` is the HTTP code. `details.body` is the truncated server response. Most common:

- `401 invalid auth token` → token audience mismatch. Decode the token at jwt.io or via `python3 -c 'import base64,json,sys; ...'` and check `aud`. The `aud` field must match what `/api/prompt` expects.
- `404` returning XML `<AuthenticationRequired>` → wrong path. Real ComfyUI endpoints on cloud live under `/api/*`; everything else hits the CDN catch-all.
- `503` → check the Comfy Cloud dashboard or server logs for deployment health.

### `cloud_timeout`
The workflow took longer than `--timeout` (default 600s). Either raise it (`--timeout 1800`) or watch instead of polling:
```
RES=$(comfy --json run --workflow X.json --where cloud)   # async by default
PID=$(echo "$RES" | jq -r .data.prompt_id)
comfy --json-stream jobs watch "$PID" --where cloud --max-wait 1800
```

### `ws_timeout` / `ws_disconnected` (local)
The WebSocket dropped mid-stream. Reconnect:
```
comfy jobs status <prompt_id>      # one-shot snapshot
comfy --json-stream jobs watch <prompt_id>   # re-attach
```
The local server keeps the job; the CLI just lost its tail.

### `cql_query_invalid`
The query syntax is wrong. CQL uses pipe-separated filter directives — *not* SQL.
For "is this node here?" use `comfy nodes search <name>` instead.
For the full grammar, see the core `comfy` skill or `comfy nodes ls --help`.

### `cql_no_graph`
The CLI needs an `object_info.json` to query against. Two options:
```
comfy launch                                        # then re-run the nodes command
comfy --json nodes ls --produces IMAGE --input /path/to/object_info.json
```

## Job-stuck triage

If a prompt seems to hang:

```
# 1. Is it still tracked?
comfy --json jobs ls

# 2. What does the server think?
comfy --json jobs status <prompt_id>

# 3. Watch live events (NDJSON, one per line)
comfy --json-stream jobs watch <prompt_id>
```

Three possible answers:

| Symptom | Likely cause | Fix |
|---|---|---|
| `status: pending`, `queue_position > 0` | Earlier jobs ahead of you | Wait or cancel them |
| `status: running` but no progress events | A long sampler step, or a hung custom node | Wait one sampler-step's worth; if still stuck, interrupt |
| `status: error` with no `error_message` | Server crashed mid-execution | Check the Comfy Cloud dashboard or server logs (cloud), or the stdout of `comfy launch` (local) |

## Interrupting / cancelling

```
comfy --json run --workflow X.json    # async by default, returns a prompt_id
# later, if it's running too long:
comfy jobs cancel <prompt_id>         # works on both local and cloud
```

## When `--json` makes diagnosis worse, not better

Pretty mode hides the envelope behind a rendered panel. If the rendered output is opaque, **always re-run with `--json`** — that's the contract. Conversely, if `--json` floods you with NDJSON, drop `-stream` to get a single envelope.

## What to NOT do

- Don't `chmod` files in the model dirs to "fix" load failures — the path is wrong, not the permissions.
- Don't delete `.installed.json` or the OAuth session to "reset" — that loses state. Use `comfy cloud logout` and `comfy skills uninstall` instead.
- Don't paste raw error messages into prompts without the `details` block — the details are usually what disambiguates the fix.

## Where to look when nothing in the envelope helps

1. `comfy --json env`               → interpreter, workspace, server reachability, conda env
2. `comfy --json which`              → resolved workspace path + how it was picked
3. `comfy --json cloud whoami`        → cloud session state + base URL
4. Server logs:
   - Local: whatever stdout `comfy launch` was attached to (foreground), or the journal/console for `comfy launch --background`
   - Cloud: check the Comfy Cloud dashboard or server logs
5. The bundled `comfy` skill — re-read it; the right command for what you're trying to do is almost certainly already documented there.
