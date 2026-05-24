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

The four most common error codes and what to do:

| Code | Do this |
|---|---|
| `server_not_running` | `comfy launch` to start the local server, or switch to `--where cloud` |
| `cloud_not_configured` | Ask the user to run `comfy cloud login` (opens browser, OAuth + PKCE) |
| `cloud_unauthorized` | Session expired or token rejected. Run `comfy cloud login` again. |
| `node_not_found` | Read `details.close_matches` — pick the closest match and re-run |

For the full error code list and resolution steps, run `comfy --json discover`.

## Routing the request — match the path to the intent

Most creative requests fall into one of three paths. Pick by what
matches the user's intent best — partner-API providers are often the
highest-quality option, not the fallback.

| If the user… | Use |
|---|---|
| names a partner provider (Flux Pro, Kling, Nano Banana, Veo, Grok, Ideogram, …) | `comfy generate <slug>` — direct dispatch against the provider's API |
| asks for a shape the **gallery already covers** ("text-to-video", "remove background", "upscale image", "img-to-3D") | `comfy templates ls` → `comfy templates fetch <name>` → slot-edit → `comfy run` |
| needs LoRAs, ControlNets, multi-step pipelines, or an OSS model the gallery doesn't cover | `comfy models search` to find the right files → build the workflow → `comfy run` |

The middle row is the workhorse — `Comfy-Org/workflow_templates` has
hundreds of curated workflows that are higher-quality than anything an
agent would build from raw nodes. **Start there.**

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
comfy --json cloud whoami    # signed_in, auth_method (oauth/api_key), base_url, api_key_source
comfy --json auth list       # all credentials (redacted)
```

## Nodes — introspect the graph

Use flag-based filters on `nodes ls` to find nodes by capability:

```bash
comfy --json nodes search "checkpoint"           # fuzzy by name/desc
comfy --json nodes show KSampler                 # full schema
comfy --json nodes ls --produces MODEL --limit 5 # filter by output type
comfy --json nodes ls --accepts CONDITIONING     # nodes that take this input
comfy --json nodes ls --category "loaders*"      # glob on category path
comfy --json nodes ls --pack comfyui-impact-pack # nodes from a specific pack
comfy --json nodes ls --api-only                 # only partner-API nodes
comfy --json nodes ls --output-only              # terminal output nodes (SaveImage, etc.)
comfy --json nodes ls --exclude-deprecated       # skip deprecated nodes
comfy --json nodes ls --cloud-disabled           # what cloud refuses to run
comfy --json nodes upstream KSampler             # what feeds in
comfy --json nodes downstream CheckpointLoaderSimple  # what follows
comfy --json nodes path MODEL IMAGE              # routed paths between types
comfy --json nodes types                         # all connection types
comfy --json nodes categories                    # full category tree
```

Combine flags to narrow results:

```bash
comfy --json nodes ls --produces VIDEO --exclude-deprecated --limit 10
comfy --json nodes ls --pack core --produces MASK --limit 5
```

If no local server is running and you're not signed into cloud, pass
`--input <object_info.json>` to query against a saved dump.

## Models — find what's installed, with metadata

On **cloud**, `comfy models search` hits the live asset catalog
(`/api/assets`) and returns enriched rows: `name`, `type`, `tags`,
`base_model`, `source_url`, `preview_url`, `size`. On **local**, the same
command falls back to `/models/<folder>` listings (filenames only).

```bash
comfy --json models list-folders                 # every model folder (loras, checkpoints, vae, …)
comfy --json models list-folder loras            # files in a folder, with pathIndex
comfy --json models search --text "wan2.2" --type lora --limit 10
comfy --json models search --text "flux"         # text search across the catalog
comfy --json models show wan2.2_vae.safetensors  # full Asset + projected row
```

`models search --type <X>` accepts the conventional folder names
(`lora`/`loras`, `checkpoint`/`checkpoints`, `vae`, `controlnet`,
`upscale`, `clip`, `clip_vision`, `unet`/`diffusion_models`, …). Use
`models list-folders` first if you're unsure what types the backend
exposes.

## Templates — start from a known-good workflow

The curated `Comfy-Org/workflow_templates` gallery is the **canonical
entry point** for any "build me a workflow that does X" request. Don't
reinvent.

```bash
comfy --json templates ls --type video --tag "Image to Video" --limit 10
comfy --json templates show <name>               # full metadata: models, tags, providers
comfy --json templates fetch <name> --out my.json # pulls the workflow JSON itself
```

`templates fetch` validates the name against the gallery index first, so
typos surface as `template_not_found` with `details.close_matches` — not
as a raw 404. The downloaded JSON is frontend-format; `comfy run --where
cloud` auto-converts it to API format on submit.

## Saved workflows on cloud

`comfy workflow {list,save,get,delete}` manages workflows persisted to
your cloud account via `/api/workflows`. Cloud-only — on local, manage
JSON files on disk via `workflow slots/set-slot/vary` instead.

```bash
comfy --json workflow list                             # paginated, sorted by create_time
comfy --json workflow list --name "wan" --limit 5      # case-insensitive name filter
comfy --json workflow get <id> --out my.json           # writes workflow JSON
comfy --json workflow save my.json --name "X" --description "Y"
comfy --json workflow delete <id>
```

## Cancel a running job

```bash
comfy --json jobs cancel <prompt_id>            # auto-routes via --where
comfy --json jobs cancel <prompt_id> --where cloud
```

Idempotent on cloud — calling on an already-terminal job returns ok.
Local cancels both the pending-queue entry and any in-flight execution.

## The ecosystem is vast — explore before building

ComfyUI spans **image, video, audio, 3D, and text** — with hundreds of
models and many partner API providers (BFL, Kling, Runway, ElevenLabs,
Meshy, Gemini, Grok, …). Don't guess at counts — discover them:

```bash
comfy --json nodes ls --limit 1                  # check data.total for node count
comfy --json nodes ls --produces IMAGE --limit 1 # IMAGE producer count
comfy --json nodes ls --produces VIDEO --limit 1 # VIDEO producer count
comfy --json nodes ls --produces AUDIO --limit 1 # AUDIO producer count
comfy --json nodes ls --api-only --limit 1       # partner API node count
comfy --json nodes categories --prefix "api node"# API provider categories
comfy --json nodes types                         # all connection types
comfy --json models list-folders                 # all model folders
comfy --json templates ls --limit 1              # template count
```

The `total` field in `nodes ls`, `nodes search`, and `models search`
gives the full count even when `--limit` caps the returned rows.

## Workflows — what can I tweak?

```bash
comfy --json workflow slots path.json   # every addressable slot, by address
```

Slot addresses are `<instance_id>.<input_name>`. Feed them to
`workflow set-slot` / `workflow vary` in the Execution half. Works on
any frontend-format workflow JSON — templates, saved workflows, or
hand-built files.

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
jq '{status, outputs, error}' "$STATE_FILE"

# (c) `--wait` on submit: foreground blocks from start to end.
comfy --json run --workflow path.json --wait
```

Pass `--notify` on `comfy run` to fire a desktop notification when the
job is terminal (handy for human-driven sessions; off by default so
agent pipelines don't spam).

## Pre-flight — validate before you submit

Before `comfy run`, verify the workflow will succeed:

```bash
# Full pre-flight: checks class_types, input shapes, enum values, edge wiring
comfy --json validate --workflow api.json

# Spot-check a single node class exists on the target
comfy --json nodes show <ClassName>
# If error.code == "node_not_found", check details.close_matches

# Confirm a model filename is actually available on the resolved backend
comfy --json models show <filename>
# If error.code == "model_not_found", check details.close_matches and pick one
```

This catches the most common failures — unknown nodes, missing models,
bad wiring — before burning cloud compute.

## Inspect / track jobs

```bash
comfy --json jobs ls                # merged: local state files + server queue
comfy --json jobs status <prompt_id>
comfy --json jobs watch <prompt_id> # blocks until terminal; emits NDJSON with --json-stream
```

## Edit workflows in place

`workflow slots`, `set-slot`, and `vary` work on any frontend-format
workflow JSON — not just templates. Get slot addresses first:

```bash
# 1. Discover addressable slots
comfy --json workflow slots path.json
# → lists every slot as <instance_id>.<input_name> with current values

# 2. Set a single slot
comfy workflow set-slot path.json 6.text="a cat"

# 3. Generate variations (slot lists are zipped — same length required)
comfy workflow vary path.json \
    --slot positive_prompt.text='["a cat","a dog","a fox"]' \
    --slot sampler.seed='[1,2,3]' \
    --out-dir ./variants
# → 3 workflow JSONs in ./variants
```

## Auth

```bash
comfy --json cloud login                         # browser OAuth + PKCE
comfy --json cloud logout
comfy --json auth set huggingface --key hf-…     # third-party provider key
comfy --json cloud set-key --key sk-…            # API-key path for cloud
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
