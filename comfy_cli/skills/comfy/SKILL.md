---
name: comfy
description: Generate images and videos, manage ComfyUI workflows, install models, query the node graph — via a local CLI. No MCP server required.
---

You have access to `comfy`, a local CLI that drives ComfyUI (local install or Comfy Cloud).

The surface splits cleanly in two:

- **Discovery** — read-only commands that answer "what's here?" (nodes,
  schemas, workflow slots, auth state, env). Safe to call freely.
- **Execution** — state-changing commands that submit work, edit files,
  sign in, or install software.

Read the **Ground rules** first; they cross both halves. Then the two
halves are independent — you can scan only what's relevant to the task.

---

# Ground rules

## Output contract (the envelope)

Every command emits the same JSON shape:

```json
{
  "ok": true,
  "command": "...",
  "version": "0.0.0",
  "where": "local" | "cloud" | null,
  "data": { ... },
  "error": null | { "code": "...", "message": "...", "hint": "...", "details": {...} }
}
```

When `error` is present, **read the `hint` and act on it**. Don't guess.

## Routing

The CLI auto-detects: `cloud` if credentials are configured (API key or
OAuth session), else `local`. Override with `--where`, `COMFY_WHERE` env,
or `comfy set-default --where cloud`. Check routing:

```bash
comfy --json cloud whoami   # signed_in, auth_method, base_url
```

If the user is signed in, commands auto-route to cloud — just run them
without `--where`. Mention routing only when the user asks to switch.

## Error codes — react, don't guess

| Code | Do this |
|---|---|
| `cloud_not_configured` | Ask the user to run `comfy cloud login` (opens browser, OAuth + PKCE) |
| `cloud_unauthorized`   | Session expired or token rejected. Run `comfy cloud login` again. |
| `cloud_http_error`     | HTTP failure from cloud — `details.status` + `details.body` carry the why |
| `auth_not_signed_in`   | Run `comfy cloud login`; the user must complete sign-in in their browser |
| `oauth_*` (any)        | Surface the message + hint verbatim; the user needs to act in their browser |
| `server_not_running`   | `comfy launch` to start the local server, or switch to `--where cloud` |
| `cql_no_graph`         | Pass `--input <path>` to a saved `object_info.json`, or run `comfy launch` |
| `cql_query_invalid`    | The CQL grammar is pipe-separated directives (e.g. `produces IMAGE \| NOT deprecated`), not SQL. See `comfy nodes ls --help` for examples; for "is X here?" use `nodes search` instead |
| `node_not_found`       | Read `details.close_matches` — pick the closest match and re-run |
| `workflow_not_frontend_format` | This command requires the UI export, not the API export. Ask the user to save via `File > Save`. |
| `auth_not_found`       | Re-list with `comfy --json auth list` to see what's actually stored |
| `not_in_workspace`     | `comfy install` or pass `--workspace /path/to/ComfyUI` |

---

# Discovery — what's here?

Read-only. None of these mutate state, charge quota beyond a cheap read,
or require sign-in unless you target `--where cloud` against a node graph
the user doesn't have locally. Run them freely.

**Always start a non-trivial task with:**

```bash
comfy --json discover
```

Returns the full command tree, JSON Schemas for every output, error
codes, and capabilities. Everything below flows from it.

## Workspace + auth state

```bash
comfy --json env             # what's installed locally
comfy --json which           # workspace path
comfy --json cloud whoami     # signed_in, auth_method (oauth/api_key), base_url, api_key_source
comfy --json auth list       # all credentials (redacted)
```

## Nodes — introspect the graph

For most cases use the flag-based verbs:

```bash
comfy --json nodes search "checkpoint"           # fuzzy by name/desc
comfy --json nodes show KSampler                 # full schema
comfy --json nodes ls --produces MODEL --limit 5 # filter by output type
comfy --json nodes ls --accepts CONDITIONING     # nodes that take this input
comfy --json nodes ls --category "loaders*"      # glob on category path
comfy --json nodes upstream KSampler             # what feeds in
comfy --json nodes downstream CheckpointLoaderSimple  # what follows
comfy --json nodes path MODEL IMAGE              # routed paths between types
comfy --json nodes types                         # all connection types
comfy --json nodes categories                    # full category tree
```

For pipeline/boolean grammar, use `--query` (the **CQL** typed query
language — pipelines, boolean predicates, sorting):

```bash
comfy --json nodes ls --query "produces IMAGE | NOT deprecated | sort connections | limit 10"
comfy --json nodes ls --query "(produces VIDEO OR produces LATENT) AND NOT api"
```

CQL grammar reference: [github.com/Comfy-Org/cql](https://github.com/Comfy-Org/cql).

If no local server is running and you're not signed into cloud, pass
`--input <object_info.json>` to query against a saved dump.

## The ecosystem is vast — explore before building

ComfyUI is not just image generation. The node graph spans **image, video,
audio, 3D, and text** — with hundreds of models and 30+ partner API
providers (BFL, Kling, Runway, ElevenLabs, Meshy, Gemini, Grok, …).

| What | Count | Discover |
|---|---|---|
| Total nodes | 3,400+ | `comfy --json nodes ls --limit 1` → `total` |
| IMAGE producers | 900+ | `comfy --json nodes ls --produces IMAGE` |
| VIDEO producers | 80+ | `comfy --json nodes ls --produces VIDEO` |
| AUDIO producers | 70+ | `comfy --json nodes ls --produces AUDIO` |
| Partner API nodes | 200+ | `comfy --json nodes ls --category "api node*"` |
| API providers | 30+ | `comfy --json nodes categories --prefix "api node"` |
| Checkpoints | 60+ | `comfy --json nodes show CheckpointLoaderSimple` → `choices` |
| LoRAs | 580+ | `comfy --json nodes show LoraLoader` → `choices` |
| Connection types | 88 | `comfy --json nodes types` |

The `total` field in `nodes ls` and `nodes search` gives the full count
even when `--limit` caps the returned rows.

## Workflows — what can I tweak?

```bash
comfy --json workflow slots path.json   # every addressable slot, by address
```

Slot addresses are `<instance_id>.<input_name>`. Feed them to
`workflow set-slot` / `workflow vary` in the Execution half.

---

# Execution — make it happen

State-changing. Each of these submits work, edits files, charges cloud
quota, or talks to an authenticated backend.

## Submit a workflow

`comfy run` is **async by default** — returns a `prompt_id` and
`state_file` path in milliseconds. A detached watcher polls in the
background and writes the state file as the job progresses through
`queued → allocated → executing → terminal`.

**Do NOT poll `jobs status` in a loop.** That wastes your turns AND the
cloud's quota. Use one of the three patterns below.

```bash
# 1. Submit. Returns immediately.
RES=$(comfy --json run --workflow path.json)
PROMPT_ID=$(echo "$RES" | jq -r .data.prompt_id)
STATE_FILE=$(echo "$RES" | jq -r .data.state_file)
```

Pick one to wait — never poll:

```bash
# (a) Block in the current turn until the prompt is terminal. Best when
#     you need the outputs to do the next step.
comfy --json jobs watch "$PROMPT_ID"
# → returns when status ∈ {completed, error, cancelled}, with outputs in
#   the final envelope.

# (b) Read the state file directly once you reason the job should be done.
#     Cheapest — no extra process. Status is source of truth.
cat "$STATE_FILE" | jq '{status, outputs, error}'

# (c) `--wait` on submit: foreground blocks from start to end.
comfy --json run --workflow path.json --wait
```

Pass `--notify` on `comfy run` to fire a desktop notification when the
job is terminal (handy for human-driven sessions; off by default so
agent pipelines don't spam).

## Pre-flight — validate before you submit

Before `comfy run`, verify the workflow will succeed:

```bash
# Check every class_type exists on the target
comfy --json nodes show <ClassName>
# If error.code == "node_not_found", check details.close_matches

# Check model/checkpoint names against available choices
comfy --json nodes show CheckpointLoaderSimple | jq '.data.inputs[] | select(.name=="ckpt_name") | .choices'
```

This catches the two most common failures — unknown nodes and missing
models — before burning cloud compute.

## Inspect / track jobs

```bash
comfy --json jobs ls                # merged: local state files + server queue
comfy --json jobs status <prompt_id>
comfy --json jobs watch <prompt_id> # blocks until terminal; emits NDJSON with --json-stream
```

## Edit workflows in place

Get slot addresses first via `workflow slots` (Discovery), then:

```bash
comfy workflow set-slot path.json 6.text="a cat"

comfy workflow vary path.json \
    --slot positive_prompt.text='["a cat","a dog","a fox"]' \
    --slot sampler.seed='[1,2,3]' \
    --out-dir ./variants
# → 3 workflow JSONs in ./variants; slot lists are zipped.
```

## Auth

```bash
comfy --json cloud login                    # browser OAuth + PKCE
comfy --json cloud logout
comfy --json auth set huggingface --key hf-…    # third-party provider key
comfy --json cloud set-key --key sk-…      # API-key path for cloud
```

## File transfer — upload and download

**Never extract API keys manually.** The CLI handles auth internally.

```bash
# Upload local files to the server's input directory
comfy --json upload photo.png video.mp4 --where cloud
# → {"uploads": [{"local_path": "...", "cloud_name": "abc123.png", ...}]}

# Download outputs from a completed job
comfy --json download <prompt_id> --where cloud
# → saves to ./outputs/<prompt_id[:8]>_000.png, etc.

# Pipe pattern — the idiomatic way to generate + collect:
comfy --json run --workflow flux.json --where cloud --wait | comfy download --where cloud
# → submits, waits, downloads outputs to ./outputs/ in one pipeline
```

`comfy download` reads prompt_id + output URLs from piped stdin
automatically — no manual extraction, no `jq`, no API key exposure.

Default output directory: `./outputs/` (configurable via `--out-dir`).

## Project layout convention

```
my-project/
├── workflows/     # API-format workflow JSON files
├── inputs/        # source images, videos, audio for upload
├── outputs/       # generated outputs (comfy download writes here)
└── variants/      # sweep outputs from comfy workflow vary
```

All `comfy` commands respect this layout by default.

## Lifecycle (local installs + persistent config)

```bash
comfy install                              # set up a local ComfyUI workspace
comfy launch                               # start the local server
comfy set-default --where cloud            # persist the routing mode
comfy set-default --clear-where
```

---

# Async + parallel — cross-cuts both halves

Image generation: ~5-30s. Video generation: **2-5 minutes**. Upscale
chains and multi-stage pipelines: variable.

Don't block your turn on a long job — do other useful work while the
watcher updates the state file, then check when you need the result.
The three wait patterns are in **Submit a workflow** above (`jobs watch`,
state file read, `--wait`).

For parallel fan-out, batch sweeps, and multi-stage pipelines, see the
**comfy-pipeline** skill.
