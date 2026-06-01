---
name: comfy
description: Generate images, videos, audio, and 3D via ComfyUI — CLI surface, workflow creation hierarchy (template → fragment → raw JSON), domain gotchas, cloud auth, multi-stage orchestration.
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
| needs LoRAs, ControlNets, multi-step pipelines, or an OSS model the gallery doesn't cover | fragments + blueprint → `comfy workflow compose` → `comfy run` (see decision tree below) |

The middle row is the workhorse — `Comfy-Org/workflow_templates` has
hundreds of curated workflows that are higher-quality than anything an
agent would build from raw nodes. **Start there.**

## Workflow creation — the decision tree (ALWAYS follow this)

For **every** creative request, follow this hierarchy strictly:

1. **Template first** — `comfy templates ls --type <image|video|audio>`
   If a match exists → fetch → slot-edit → run. Don't reinvent.

2. **Fragment + blueprint** — if no template fits, this is the **default path**:

   **a. Discover nodes** for each step of the pipeline:
   ```bash
   comfy --json nodes show GrokImageNode         # check inputs/outputs
   comfy --json nodes show KlingImage2VideoNode   # check inputs/outputs
   ```

   **b. Create one fragment per logical step** — write to `fragments/<name>.json`:
   Each fragment wraps 1-15 nodes with a `_fragment` header declaring
   typed inputs, outputs, and params. The interior nodes are standard
   API-format ComfyUI JSON. Mark caller-supplied values as `"PLACEHOLDER"`.

   **c. Validate each fragment:**
   ```bash
   comfy --json workflow fragment validate <name>
   ```

   **d. Write a YAML blueprint** in `blueprints/<name>.yaml` that wires
   the fragments together — cross-step refs use `$alias.output_name`:
   ```yaml
   output_prefix: outputs/my_project
   pipeline:
     - fragment: generate_image
       alias: hero
       params:
         prompt: "a zen garden at dawn"
         seed: 42
     - fragment: animate_i2v
       alias: video
       inputs:
         image: $hero.image       # ← wires step 1 output to step 2 input
       params:
         motion: "slow camera pan left"
   ```

   **e. Compose + run:**
   ```bash
   comfy workflow compose blueprints/<name>.yaml -o workflows/<name>.json
   RES=$(comfy --json run --workflow workflows/<name>.json)
   PROMPT_ID=$(echo "$RES" | jq -r .data.prompt_id)
   comfy --json jobs watch "$PROMPT_ID"
   ```

   For the full fragment format and blueprint syntax, load the
   `comfy-fragments` skill.

3. **Raw JSON** — ONLY for throwaway one-shot workflows under ~30 nodes
   that will never be reused. Write to `workflows/` and run directly.

**Hard rule: never build raw workflow JSON with >30 nodes. Use fragments.**
Even for smaller workflows, prefer fragments if any part will be reused.

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

**Requires a local server** (`comfy launch`) or `--input object_info.json`.
If you're cloud-only with no local server, skip `workflow slots` — instead
read the workflow JSON directly to find node inputs, or use
`comfy --json nodes show <ClassName>` to inspect individual node schemas.

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

**Prefer async-first: submit, then watch separately.** Never poll
`jobs status` in a loop. There are three ways to wait — pick one:

```bash
# Step 1: Submit (returns immediately with prompt_id)
RES=$(comfy --json run --workflow path.json)
PROMPT_ID=$(echo "$RES" | jq -r .data.prompt_id)
STATE_FILE=$(echo "$RES" | jq -r .data.state_file)

# (a) Watch — blocks until terminal, returns outputs. The default.
comfy --json jobs watch "$PROMPT_ID"
# → returns when status ∈ {completed, error, cancelled}, with outputs

# (b) Read the state file for a quick non-blocking check
jq '{status, outputs, error}' "$STATE_FILE"

# (c) --wait on submit — foreground blocks start-to-finish. Fine for
#     one-shot synchronous runs (e.g. the download pipe below).
comfy --json run --workflow path.json --wait
```

**Why prefer async:** submit returns in milliseconds so you can report
the prompt_id to the user immediately, then watch in a separate step.
Reach for `--wait` when you want a single blocking call and don't need
the prompt_id mid-flight (it's hidden until the job finishes).

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
├── fragments/     # reusable workflow pieces (_fragment JSON files)
├── blueprints/       # YAML files that compose fragments into workflows
├── workflows/     # compiled workflow JSON (compose output) or one-shots
├── inputs/        # source images, videos, audio for upload
├── outputs/       # generated outputs (comfy download writes here)
└── variants/      # sweep outputs from comfy workflow vary
```

All `comfy` commands respect this layout by default.

**Never write workflows, fragments, or outputs to `/tmp`.** Always use
the project directory. If no project directory exists, create one with
this layout before building anything.

## Lifecycle (local installs + persistent config)

```bash
comfy install                              # set up a local ComfyUI workspace
comfy launch                               # start the local server
comfy set-default --where cloud            # persist the routing mode
comfy set-default --clear-where
```

## Relaying user feedback

This channel carries the **user's own words** to the maintainers — never your
opinion of the tool. You are a relay, not the author.

```bash
comfy --json feedback "run is great, but jobs watch should show an ETA"
```

One-shot and non-interactive. The message ships as a `feedback_submitted`
event; the JSON envelope reports `{"sent": true|false}`. Feedback is
consent-independent (user-initiated), but the hard opt-out (`DO_NOT_TRACK` /
`COMFY_NO_TELEMETRY`) still suppresses it → `sent: false`.

**When to ask.** Occasionally — *not* every session — invite feedback after a
clearly good moment (a job just completed successfully, a multi-step pipeline
landed). Roughly one session in five; pick randomly so it stays rare. Ask at
most once per session, and never twice if they ignored or declined the first
time. One short question, e.g.: *"Quick one — anything about the comfy CLI you'd
want the maintainers to know? (totally optional)"*

**What to send.** Only what the user actually says, relayed close to verbatim.
Never invent, embellish, or send your own assessment. If they say nothing or
brush it off, send nothing. Strip secrets, file contents, and PII. For long or
sensitive reports, point them to
`https://github.com/Comfy-Org/comfy-cli/issues/new/choose` instead.

## Agent session review (separate, consent-gated)

Distinct from user feedback above: this is **your** short summary of how the
session went — what the user tried, what worked, where they got stuck. Send it
at most once, at the end of a substantive session:

```bash
comfy --json agent-review "User generated a text-to-video clip; hit a missing-model error on first try, succeeded after switching templates."
```

**Honors every opt-out.** Unlike user feedback, this is agent-authored, so it's
treated as passive telemetry: it sends ONLY if the user has telemetry enabled.
If they opted out by any means (`DO_NOT_TRACK`, `COMFY_NO_TELEMETRY`, or no
consent), the envelope returns `{"sent": false}` and nothing is transmitted —
that's expected, don't retry or work around it. Keep it short and factual; no
secrets, no PII, no user verbatim (that's what `comfy feedback` is for).

---

# Domain gotchas by media type

Hard-won lessons per domain. Not a tutorial — a reference card.

## Image

- Template-first: `comfy templates ls --type image --tag "Text to Image"`
- Batch sweeps: `comfy workflow vary` for multi-prompt/seed generation
- Text rendering: use Ideogram (IdeogramV3), NOT Flux — Flux garbles text
- Partner API shortcut: `comfy generate bfl/flux-pro-1.1-ultra --prompt "..."`
- Never hardcode checkpoint/LoRA names — discover via `models search`

## Video

- **SaveVideo is REQUIRED** — video API nodes produce VIDEO but are NOT output nodes
- **Never hardcode fps** — wire from GetVideoComponents output index 2
- Motion prompts: describe HOW the scene moves, not WHAT is in it
- Assembly: GetVideoComponents → ImageBatch → CreateVideo → SaveVideo
- I2V pattern: LoadImage → I2VNode → SaveVideo (check `nodes show` for the I2V node)
- Audio sync: match durations — short audio = silent ending, long audio = truncated ending
- Template-first: `comfy templates ls --type video`

## Audio

- ACE-Step: timesignature is `"4"`, NOT `"4/4"`
- Duration on TextEncode AND EmptyLatentAudio MUST match
- Output format is FLAC, not MP3
- For instrumental: set lyrics to empty string `""`
- Wire both positive AND negative to same TextEncode output when cfg=1.0
- Template-first: `comfy templates ls --type audio`

## Editing (upscale, inpaint, style transfer)

- FluxProFillNode is REPLACE-ONLY — no denoise/strength param
- For refinement: use KSampler with denoise=0.15–0.25, not FluxProFill
- MagnificImageUpscalerCreativeNode: creativity 0–10, resemblance -10–10 (NOT 0–100)
- MagnificImageRelightNode style="smooth" drains color — use "brighter" or "clean"
- Local upscale: LoadImage → UpscaleModelLoader → ImageUpscaleWithModel → SaveImage
- API upscale: discover via `comfy nodes search "upscale"`

## Conditioning (ControlNet, masks, references)

- Preprocessor output ≠ ControlNet model (two separate things)
- Don't feed raw photos into ControlNet without preprocessing first
- ImageCompositeMasked: mask MUST match SOURCE size, not destination
- COMFY_DYNAMICCOMBO_V3: use flat dotted keys (`"model.max_tokens": 800`), not nested
- First/last frame transitions: wire start_frame + end_frame → I2V node fills in between
- Wiring: ControlNetApplyAdvanced takes CONDITIONING + IMAGE + CONTROL_NET → modified CONDITIONING

## Cloud

- Auth: `comfy cloud login` (OAuth) or `comfy cloud set-key --key sk-…`
- Check: `comfy --json cloud whoami`
- Custom env: `comfy cloud set-base-url <url>` before login
- CLI auto-injects API keys for partner nodes — never extract manually
- Session tokens are short-lived (~1h); CLI auto-refreshes on 401
- HTTP 401 with XML body = CDN catch-all, not ComfyUI — endpoints are under `/api/*`
- Cloud uses HTTP polling (no WebSocket); `jobs watch` polls `/api/job/<id>/status`

---

# Multi-stage orchestration

Build one large workflow graph when possible — ComfyUI parallelizes
independent branches automatically. Only split into separate workflows when:
- An intermediate result needs human review before continuing
- Different stages need different routing (local vs cloud)
- The workflow would exceed server memory constraints

For parallel fan-out:

```bash
PIDS=()
for f in ./workflows/phase*.json; do
    PID=$(comfy --json run --workflow "$f" | jq -r .data.prompt_id)
    PIDS+=("$PID")
done
for p in "${PIDS[@]}"; do
    comfy --json jobs watch "$p" | comfy download --where cloud
done
```

Stage handoff (download → upload → re-reference):

```bash
comfy --json jobs watch "$PID" | comfy download --where cloud
CLOUD=$(comfy --json upload ./outputs/abc_000.png --where cloud \
    | jq -r '.data.uploads[0].cloud_name')
# Use $CLOUD in the next workflow's LoadImage input
```

Pipeline failure recovery: re-submit only the failed workflow. Use
`comfy --json jobs status <id>` to identify which failed.

---

# Async + parallel — cross-cuts both halves

Image generation: ~5-30s. Video generation: **2-5 minutes**. Upscale
chains and multi-stage pipelines: variable.

Don't block your turn on a long job — do other useful work while the
watcher updates the state file, then check when you need the result.
The three wait patterns are in **Submit a workflow** above (`jobs watch`,
state file read, `--wait`). The parallel fan-out pattern is in
**Multi-stage orchestration** above.
