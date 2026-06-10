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

## Routing the request — survey first, then choose

Don't commit to the first approach that fits. The ecosystem spans gallery
templates, partner-API providers, and thousands of OSS nodes/models. Survey
the option space before deciding (see "The ecosystem is vast" below for the
commands). The default is **workflow-first**: even when a partner provider is
the right model, reach for its *node* inside a fragment/blueprint so the result
is a reusable, inspectable Job on the graph. What to build — which model,
provider, and approach — is your judgment to make from what discovery returns;
this skill teaches you how to look, not what to pick.

Once you know what you want, there are three *mechanisms* to build it. Pick
by structure, not by habit — this is a mechanism map, not a quality ranking:

| Mechanism | When it fits |
|---|---|
| `comfy templates ls/fetch` → slot-edit → `comfy run` | A curated gallery workflow already matches the shape you need |
| fragments + blueprint → one composed workflow → `comfy run` | **Default** — workflows you may extend, fan out, reuse, vary, or explain later; wrap a partner provider's *node* here too |
| `comfy generate <slug>` | **Escape hatch** — a throwaway one-shot against a single partner provider (a proxy call, not a graph Job) |

Prefer **one larger Comfy workflow** over many separate submissions when the
steps can run in the same graph. Comfy can parallelize independent branches,
so use fan-out branches, batch nodes, and shared loaders/references inside one
workflow before splitting into separate jobs. Split only when a stage needs
human review, different routing/auth, server memory isolation, or failure
recovery that is worth losing graph-level parallelism.

**Escape hatch — `comfy generate <slug>`:** for a throwaway one-shot
against a single partner provider, `comfy generate` skips the graph
entirely. It dispatches to Comfy's partner-API **proxy**
(`api.comfy.org/proxy/...`), which calls the provider on your behalf and
bills to your Comfy account (it is **not** a workflow Job — nothing lands
on the graph and nothing is reusable). Reach for it only when you want a
quick disposable result; anything you'll reiterate on belongs in a
workflow.

## Workflow creation — choosing how to build

Once discovery has told you *what* to build, choose the construction
mechanism by complexity and reuse — not as a quality ranking:

1. **Template** — `comfy templates ls --type <image|video|audio>`
   If a curated workflow matches the shape, fetch it. For one-off smoke tests,
   slot-edit and run it directly. For anything that may become a longer piece,
   multiple variations, or a reusable pattern, wrap the useful template region
   as a fragment and drive it from a blueprint.

2. **Fragment + blueprint** — this is the **default construction path** once
   the workflow is more than a throwaway. Use it even for simple workflows if
   the next likely step is "make it longer", "add another shot", "vary seeds",
   "reuse this with a different prompt", or "chain another model".

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

   Prefer a single composed workflow with repeated fragment instances over a
   loop of separate `comfy run` calls. For example, a music video should be a
   manifest/blueprint that composes one fan-out graph with N video branches and
   shared character references, then a separate assembly step if final editing
   needs exact audio sync.

3. **Raw JSON** — ONLY for truly throwaway one-shot workflows under ~10-15
   nodes where extension is not expected. Write to `workflows/` and run
   directly.

**Hard rule: never build raw workflow JSON with >30 nodes. Use fragments and a
blueprint.** Even for smaller workflows, prefer fragments if any part could be
extended, repeated, or reused.

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

## Templates — one starting point among several

The curated `Comfy-Org/workflow_templates` gallery is a strong starting
point *when a template matches your intent* — but it sits beside partner-API
providers and hand-composed fragments, not above them. Survey all three
(see "The ecosystem is vast") before committing.

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

**Scope:** the async-first / `jobs watch` / state-file pattern above is the
**`comfy run`** workflow path only. `comfy generate` (partner-API one-call)
has its own waiting model — see the next section.

## Partner-API one-call generation (`comfy generate`)

`comfy generate` dispatches straight to a partner provider (BFL Flux, Kling,
Gemini, Veo, Ideogram, …) and is often the highest-quality route for a single
image/video/edit — not a fallback. It is a **separate sub-surface** from
`comfy run`, with its own commands and conventions:

```bash
comfy generate list                       # enumerate provider models (+ their sync/async mode)
comfy generate schema <model>             # params for one model (e.g. flux-2, kling-i2v)
comfy generate <model> --prompt "…" [--<param> v]… --download outputs/x.png
comfy generate upload <file>              # host a local file → signed URL (for I2V image inputs)
comfy generate <model> … --async          # submit, returns a job id
comfy generate resume <model> <job_id> --download outputs/x.mp4
```

Mechanical contracts that bite agents — encode them, don't rediscover:

- **`comfy generate` does NOT honor the global `--json` envelope.** Unlike the
  rest of the CLI, the `generate` subtree prints human/Rich output (including an
  ANSI image preview on stdout) and gives you **no machine-readable envelope**.
  Do not parse its stdout. The reliable result is the **file** written by
  `--download` — submit with `--download <path>`, then verify the file exists.
- **`generate upload` prints `Uploaded: <url>` as pretty text**, and the signed
  URL soft-wraps across terminal lines. Sanitize before reuse
  (`sed 's/^Uploaded:[[:space:]]*//' | tr -d '[:space:]'`).
- **Prefer sync** (plain `--download`, no `--async`): the CLI polls internally
  and waits for you (that's the tool blocking, not you sleep-polling), so an
  expensive video gen can't be orphaned. Reach for `--async` + `resume` only
  when you deliberately want to detach. (`generate resume` for BFL is currently
  unreliable — if it errors, just re-run sync.)
- **I2V pattern:** `generate <i2v-model>` needs an image **URL**, so the flow is
  `generate <image-model> --download still.png` → `generate upload still.png` →
  `generate <i2v-model> --image <url> --download clip.mp4`.

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
# 1. Discover addressable slots — addresses are <node_id>.<input>, never titles
comfy --json workflow slots path.json
# → lists every slot as <node_id>.<input_name> with current values
#   subgraph interiors: <instance_id>/<inner_id>.<input>
#   copy addresses verbatim from this output

# 2. Set a single slot
comfy workflow set-slot path.json 6.text="a cat"

# 3. Generate variations (slot lists are zipped — same length required)
comfy --json workflow slots wf.json          # discover addresses first
comfy workflow vary wf.json \
    --slot '6.text=["a cat","a dog","a fox"]' \
    --slot '3.seed=[1,2,3]' \
    --out-dir ./variants
# → 3 workflow JSONs in ./variants
# NOTE: slot addresses use node ids (numeric or UUID), never titles.
#       Always run `slots` first and copy addresses verbatim.
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

- Survey first: `comfy nodes ls --produces IMAGE --api-only` (partner APIs), `comfy templates ls --type image`, `comfy models search --type checkpoint` — then choose
- Batch sweeps: `comfy workflow vary` for multi-prompt/seed generation
- Text rendering: use Ideogram (IdeogramV3), NOT Flux — Flux garbles text
- Partner API escape hatch (one-shots only, via the proxy — not a workflow Job): `comfy generate bfl/flux-pro-1.1-ultra --prompt "..."`
- Never hardcode checkpoint/LoRA names — discover via `models search`

## Video

- **SaveVideo is REQUIRED** — video API nodes produce VIDEO but are NOT output nodes
- **Never hardcode fps** — wire from GetVideoComponents output index 2
- Motion prompts: describe HOW the scene moves, not WHAT is in it
- Assembly: GetVideoComponents → ImageBatch → CreateVideo → SaveVideo
- I2V pattern: LoadImage → I2VNode → SaveVideo (check `nodes show` for the I2V node)
- Audio sync: match durations — short audio = silent ending, long audio = truncated ending
- Survey first: `comfy nodes ls --produces VIDEO --exclude-deprecated`, `comfy nodes ls --category "api node/video*"`, `comfy templates ls --type video` — compare OSS, partner-API, and gallery before choosing

## Audio

- ACE-Step: timesignature is `"4"`, NOT `"4/4"`
- Duration on TextEncode AND EmptyLatentAudio MUST match
- Output format is FLAC, not MP3
- For instrumental: set lyrics to empty string `""`
- Wire both positive AND negative to same TextEncode output when cfg=1.0
- Survey first: `comfy nodes ls --produces AUDIO`, `comfy nodes ls --category "api node/audio*"`, `comfy templates ls --type audio` — compare before choosing

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

Some steps don't belong in a Comfy graph at all — final assembly, format
conversion, timing/structure analysis of generated media. Comfy outputs are
just files; when the graph can't express what the task needs, you're free to
orchestrate your own tools around those files. How is your call — this skill
defines what `comfy` does, not the limits of what you can do with its output.

**For parallel generation, don't hand-roll fan-out with shell loops.** The
engine already parallelizes independent branches, so author **ONE** graph and
let it run them concurrently. Independent fragment steps in a blueprint
`pipeline` (those that don't wire into each other) become parallel branches.
For the same pipeline across many inputs, use `foreach` — it instantiates the
pipeline once per item into a single graph:

```yaml
# blueprints/fan_out.yaml — one graph, N parallel branches via foreach
output_prefix: outputs/sweep
foreach:
  - {id: a, prompt: "a zen garden"}
  - {id: b, prompt: "a neon city"}
  - {id: c, prompt: "a desert at dusk"}
pipeline:
  - fragment: generate_image
    alias: shot
    params:
      prompt: $item.prompt
```

```bash
comfy workflow compose blueprints/fan_out.yaml -o workflows/fan_out.json
RES=$(comfy --json run --workflow workflows/fan_out.json)
comfy --json jobs watch "$(echo "$RES" | jq -r .data.prompt_id)"
```

This submits a single Job; the engine runs the independent branches
concurrently. Avoid the old `PIDS=()` shell-loop pattern — it duplicates
scheduling the engine already does and gives you N jobs to babysit instead
of one. (For a pure prompt/seed sweep over the *same* graph, `comfy
workflow vary` is the right tool; see the `comfy-fragments` skill for the
full blueprint syntax.)

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
state file read, `--wait`). For parallelism, author one graph with
independent (parallel) branches in a single blueprint rather than fanning
out across jobs — see **Multi-stage orchestration** above.
