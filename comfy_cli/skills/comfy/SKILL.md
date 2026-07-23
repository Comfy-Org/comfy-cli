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

**This is one of a skill family — skim the siblings before a big task so you
know what exists, and reach for the right one rather than improvising its job:**
`comfy-fragments` (compose large graphs from reusable, validated pieces),
`comfy-director` (multi-shot narrative video — story, continuity, conform),
`comfy-debug` (any failed job: error code → fix), `comfy-relay` (surface a
workflow/result in chat, never leave it in /tmp). When a task spans several,
load them up front instead of discovering the gap mid-render.

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
OAuth session), else `local`. Precedence: `--where` flag → `COMFY_WHERE`
env → the governing project's `defaults.where` (`comfy.yaml`, see
Projects) → `comfy set-default --where …` config → auto-detect. A local
`--where` always beats the project default. Check routing:

```bash
comfy --json cloud whoami   # signed_in, auth_method, base_url
```

If the user is signed in, commands auto-route to cloud — just run them
without `--where`. Mention routing only when the user asks to switch.

## Error codes — react, don't guess

The most common error codes and what to do:

| Code | Do this |
|---|---|
| `server_not_running` | `comfy launch` to start the local server, or switch to `--where cloud` |
| `cloud_not_configured` | Ask the user to run `comfy cloud login` (opens browser, OAuth + PKCE) |
| `cloud_unauthorized` | Your CLI session expired or token rejected *before submission*. Run `comfy cloud login` again. |
| `transient_auth` | A cloud job died mid-run with "Unauthorized: Please login first to use this node" — server-side token expiry, NOT your login. Resubmit the same workflow; do NOT re-login. |
| `node_not_found` | Read `details.close_matches` — pick the closest match and re-run |

For the full error code list and resolution steps, run `comfy --json discover`.
When any *job* fails (an `execution_error`-family envelope), invoke the
`comfy-debug` skill before improvising — it maps every failure code to a fix.

## Presenting your work — show, don't tell

This is **visual, iterative** work, not code — the user steers by *seeing* the
result, so the image is the message, never a path or a sentence about it. The
moment a generation lands, **`Read` it into chat** (for a clip, run `comfy
preview clip.mp4` → a contact-sheet PNG you `Read`, plus duration/fps/audio).
Lead with the
visual, then show the *source* that made it (the blueprint/prompt) — never the
compiled JSON. Recommend with taste; iterate in fast show→react loops rather
than long upfront questionnaires. You can see frames but **cannot hear audio** —
for music/SFX say "give it a listen" and defer to the user's ear. The full
playbook is the **`comfy-relay`** skill — load it whenever you generate, review,
or iterate on media.

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
   multiple variations, or a reusable pattern, **project it into source** with
   `comfy workflow decompose` (below) and drive it from a blueprint — don't
   hand-edit the fetched JSON.

2. **Fragment + blueprint** — this is the **default construction path** once
   the workflow is more than a throwaway. Use it even for simple workflows if
   the next likely step is "make it longer", "add another shot", "vary seeds",
   "reuse this with a different prompt", or "chain another model".

   There is no shipped fragment library. A fragment is what YOU write
   *after* deriving the wiring from live sources — distilled knowledge,
   kept in the project's `./fragments/`. The loop below is the reusable
   part; the artifacts it produces depend on what YOUR backend has today.

   **a. Survey the space** — compare OSS, partner, and gallery options
   before choosing (swap `video`/`VIDEO` for the media type at hand):
   ```bash
   comfy --json templates ls --type video --limit 10        # working exemplar graphs
   comfy --json nodes ls --produces VIDEO --exclude-deprecated
   comfy --json nodes ls --category "partner/video*"       # partner providers
   ```

   **b. Learn the wiring from a real graph.** Templates are
   upstream-curated, *working* workflows — THE reference for how nodes
   actually wire (positive vs negative conditioning, lora'd CLIP feeding
   both text encoders, VAE-from-checkpoint, denoise semantics):
   ```bash
   comfy templates fetch <name-from-YOUR-survey> --out ref.json
   comfy --json workflow slots ref.json          # its addressable surface
   ```
   Or derive from the type graph directly:
   ```bash
   comfy --json nodes show <NodeClass-from-YOUR-survey> --where cloud  # exact schema + enum choices
   comfy nodes path IMAGE VIDEO                  # what CAN connect these types
   ```

   **c. Distill into a local fragment.**

   **If a working workflow already exists — a template you fetched, a
   slot-edited file, anything that runs — DECOMPOSE it. Do not re-author the
   fragment by hand.**
   ```bash
   comfy workflow decompose ref.json --name <name>   # → ./fragments/<name>.json
   ```
   `decompose` strips loaders → typed **inputs** (bound to the consumer),
   strips the terminal save → a typed **output**, and surfaces every scalar
   widget as a **named param** with its current default — all derived from the
   graph, nothing hardcoded. The buried prompt that used to need
   `jq '…widgets_values[0]'` becomes a named param you set in the blueprint.
   (Frontend/subgraph templates are flattened first, so this needs a running or
   cloud server, or `--input object_info.json`.)

   Why decompose instead of hand-authoring: the projection **inherits the exact
   working values** — correct widget *types* (the `4` int vs `"4"` string a
   model's enum actually accepts), real enum choices, wiring that already ran.
   Re-typing a fragment from a node schema re-introduces those as transcription
   bugs you only discover when the cloud rejects the job. Start from what works.

   Only author `./fragments/<name>.json` by hand when there is **no** working
   graph to project from (you're building a shape that doesn't exist yet): 1-15
   API-format nodes wrapped with a `_fragment` header declaring typed inputs,
   outputs, and params (caller-supplied values marked `"PLACEHOLDER"`). Make
   EVERY asset/model name a required param with no default. Load the
   `comfy-fragments` skill for the format, then check your work:
   ```bash
   comfy --json workflow fragment validate <name>
   ```

   **d. Compose + run** — a YAML blueprint in `blueprints/<name>.yaml`
   wires your fragments together; cross-step refs use `$alias.output_name`,
   project assets use `$asset.<relative/path>`, project constants use
   `$var.<name>` (the full `$`-reference algebra is in the Projects section):
   ```bash
   comfy workflow compose blueprints/<name>.yaml   # → blueprints/<name>.compiled.json
   RES=$(comfy --json run --workflow blueprints/<name>.compiled.json)
   PROMPT_ID=$(echo "$RES" | jq -r .data.prompt_id)
   comfy --json jobs watch "$PROMPT_ID"
   ```

   **Trace (video, partner node) — a record of one run of the loop, NOT a
   recommendation.** On this backend, today, the survey returned what's
   sketched below; yours WILL differ — pick from YOUR rows:
   ```bash
   comfy --json nodes ls --category "partner/video*" --limit 10
   # → today's rows included an image-to-video partner node; call it <I2VNode>
   comfy --json nodes show <I2VNode> --where cloud
   # → schema said: start image + prompt + a duration enum in, VIDEO out
   comfy nodes path IMAGE VIDEO   # confirmed the route; SaveVideo still required at the end
   ```
   The agent then AUTHORED `./fragments/i2v.json` (input `start_frame:
   IMAGE`; params `prompt`, `duration` — all required) and wired it behind
   a t2i fragment derived the same way from an image survey:
   ```yaml
   # blueprints/video.yaml — both fragments written by the agent, not shipped
   pipeline:
     - fragment: t2i             # ./fragments/t2i.json — from YOUR image survey
       alias: hero
       params: {prompt: "a fennec fox astronaut, golden hour"}
     - fragment: i2v             # ./fragments/i2v.json — derived above
       alias: vid
       inputs:
         start_frame: $hero.image   # ← t2i output "image" → i2v input "start_frame"
       params: {duration: "<a value from the enum nodes show returned>"}
   ```

   `comfy workflow fragment ls` lists `./fragments` — it errors with
   `fragment_lib_not_found` until that directory exists, so create it when
   you author your first fragment.

   Prefer a single composed workflow with repeated fragment instances over a
   loop of separate `comfy run` calls. For example, a music video should be a
   blueprint that composes one fan-out graph with N video branches and
   shared character references, then a separate assembly step if final editing
   needs exact audio sync.

3. **Raw JSON** — ONLY for truly throwaway one-shot workflows under ~10-15
   nodes where extension is not expected. Write the file and run it
   directly.

**Hard rule: never build raw workflow JSON with >30 nodes. Use fragments and a
blueprint.** Even for smaller workflows, prefer fragments if any part could be
extended, repeated, or reused.

## The compile model — edit source, never the artifact (REQUIRED)

The folders are **source**; the workflow JSON is a **build artifact**.
`fragments/` + `blueprints/` are what you edit; `compose` is the compiler;
`blueprints/<name>.compiled.json` is the artifact `run` executes.
(`templates fetch` pulls a vendored dependency *into* source; `decompose`
turns any workflow into a fragment; `compose` builds it back.)

This is a hard contract, not a style preference:

- **NEVER hand-edit a fetched template or a compiled/exported workflow JSON.**
  Do not open it in an editor, and do not run `jq`/`sed`/`python` to change a
  value inside a node's `inputs`/`widgets_values`, and do not append/rewire
  nodes by hand.
- **The moment you need to change anything inside a fetched/compiled workflow,
  STOP and `comfy workflow decompose <file>` it.** Then set the value as a
  named param in a blueprint and `compose`. The decomposed fragment is
  self-documenting — its `_fragment.description`/`source` say where it came
  from, and `comfy workflow fragment show <name>` lists every param with its
  `binds` + default, so you edit by name, never by node id.
- **Only exception** — a *throwaway* run of a template you will not reuse:
  `comfy workflow slots` → `set-slot`/`vary` to tweak top-level values, then
  `run`. The instant the work will be extended, reused, varied, or chained — or
  the value lives inside a subgraph `slots` can't address — decompose instead.

**Red flag — STOP:** you typed `jq`/`sed`/`Edit` against a workflow's
`widgets_values` or `inputs`, or you're hunting for a node by numeric id
(`select(.id==128)`). That means the source representation failed. `decompose`
it and set a named param. (This rule exists because that exact jq-on-`id==128`
hand-edit is the anti-pattern `decompose` was built to kill.)

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
comfy --json models show <rows[0].name>          # full Asset + projected row (cloud-only)
```

`models search --type <X>` accepts the conventional folder names
(`lora`/`loras`, `checkpoint`/`checkpoints`, `vae`, `controlnet`,
`upscale`, `clip`, `clip_vision`, `unet`/`diffusion_models`, …). Use
`models list-folders` first if you're unsure what types the backend
exposes.

**Discover → wire loop — every asset type, never hardcoded names:**

Every asset name (checkpoint, lora, controlnet, vae, upscaler, embedding,
clip-vision model, …) must be discovered at runtime — never hardcoded. Do
not default to any model family you've seen in examples, either — the
survey IS the decision input; backends differ and the ecosystem moves. The
pattern is the same regardless of type:

```bash
# 1. Discover available assets for any type
comfy --json models search --type lora --where cloud --text "detail" --limit 5
comfy --json models search --type controlnet --where cloud --limit 5
comfy --json models search --type checkpoint --where cloud --limit 5
comfy --json models search --type vae --where cloud --limit 5
comfy --json models search --type upscale --where cloud --limit 5
comfy --json models search --type embeddings --where cloud --limit 5

# 2. Take rows[0].name verbatim — paste it into your fragment's required param

# 3. Precision check — what will the server actually accept?
comfy --json nodes show LoraLoader --where cloud
# → the lora_name input's "choices" array is the exact list the server accepts
# Same pattern for any loader: ControlNetLoader, VAELoader, UpscaleModelLoader, etc.
```

**Trace (image, OSS checkpoint + lora) — one run of the loop, NOT a
recommendation.** On this backend, today, the survey returned the rows
sketched below; yours will differ — pick from YOUR rows:

```bash
comfy --json models search --type checkpoint --where cloud --limit 5  # → picked <ckpt> from rows
comfy --json models search --type lora --where cloud --limit 5        # → picked <lora> from rows
# Learn the lora wiring from a real graph, not memory — fetch a matching template:
comfy --json templates ls --type image --model "<family of <ckpt>, from its row>"
comfy templates fetch <name-from-those-rows> --out ref.json   # read how it wires
# …or derive it from the type graph:
comfy --json nodes show LoraLoader --where cloud  # MODEL+CLIP in, MODEL+CLIP out —
#   the lora'd CLIP must feed BOTH text encoders, not just positive
comfy nodes path MODEL IMAGE                      # sampler → decode → save spine
```

The agent then authored `./fragments/<your_name>.json` with `ckpt_name`
and `lora_name` as required params (no defaults) and drove it from a
blueprint:

```yaml
pipeline:
  - fragment: <your_name>        # the fragment YOU just wrote
    alias: out
    params:
      ckpt_name: "<rows[0].name from checkpoint search>"
      lora_name: "<rows[0].name from lora search>"
      prompt: "a detailed portrait"
```

The `choices` array from `nodes show` is the universal precision check: it
reflects exactly what `<server>/object_info` reports — authoritative for
any loader node on that target.

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
comfy --json nodes categories --prefix "partner"# API provider categories
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

`workflow slots`/`set-slot`/`vary` and all `nodes` commands resolve
object_info through the routing chain with a cached fallback — cloud-signed-in
works with no local server. If the live fetch fails, the command still succeeds
from cache and the envelope carries `data.stale: true` +
`warnings[] {code: "object_info_stale"}` — treat results as possibly outdated
and run `comfy nodes refresh --where cloud` to refresh.

Slot addresses are `<instance_id>.<input_name>`. Feed them to
`workflow set-slot` / `workflow vary` in the Execution half. Works on
any frontend-format workflow JSON — templates, saved workflows, or
hand-built files.

---

# Execution — make it happen

State-changing. Each of these submits work, edits files, charges cloud
quota, or talks to an authenticated backend.

## Projects (project/1) — the working convention

Anything beyond a one-shot lives in a project: a directory with a
`comfy.yaml` marker (`schema: project/1`, plus `defaults.where`) and five
conventional dirs. The convention is the contract — like the envelope.

```
my-project/
├── comfy.yaml     # marker: schema project/1 + defaults (e.g. defaults.where)
├── assets/        # source files — reference as $asset.<relative/path>
├── fragments/     # fragments YOU author (_fragment JSON)
├── blueprints/    # blueprint YAML; compose writes <name>.compiled.json beside it
├── outputs/       # downloads land here by default
└── .comfy/        # machine-owned: assets.lock.json + runs.jsonl journal
```

The loop:

```bash
comfy project init                     # marker + the five dirs; --where sets the default
cp ~/ref.png assets/s1_first.png       # drop source files under assets/ (subdirs fine)
comfy --json assets push               # upload new/changed files, record them in the lock
# blueprints reference assets by path relative to assets/ (inputs or params):
#   inputs: {start_frame: $asset.s1_first.png}
comfy workflow compose blueprints/<name>.yaml          # → blueprints/<name>.compiled.json
comfy --json run --workflow blueprints/<name>.compiled.json
comfy --json jobs watch <prompt_id>    # terminal envelope: outputs_by_item / outputs_by_node
comfy --json download <prompt_id>      # → outputs/, item-named files
comfy --json project status            # THE state query — root, defaults, blueprints,
                                       #   assets {pushed, stale}, recent_runs, warnings
```

What the convention buys you:

- **The `$`-reference algebra.** Four reference kinds in blueprints, each
  with ONE resolution source, all resolved at compose time:

  | Reference | Resolves from | Where it works |
  |---|---|---|
  | `$alias.output` | a prior step's graph output (a wire) | inputs |
  | `$item.field` | the current `foreach` item | inputs + params |
  | `$asset.<relative/path>` | the push lock → server-side filename | inputs + params + item field values |
  | `$var.<name>` | the `vars:` block in `comfy.yaml` | inputs + params + item field values |

  **Whole-value only**: a `$`-ref must be the ENTIRE string. `"a $asset.x b"`
  is plain text — there is no interpolation. `$var` returns the raw scalar
  (int stays int); `$asset` resolves through the lock with staleness checks.
- **`$asset` kills upload-then-paste.** The lock (`.comfy/assets.lock.json`)
  records sha256 + server name + push target per file, so local and cloud
  both work; `assets push` skips files whose content AND target are
  unchanged (`--force` re-pushes everything), `--where` picks the target.
- **`$var` kills copy-pasted constants.** Declare a top-level `vars:`
  mapping (str/int/float/bool scalars) in `comfy.yaml`; blueprints reference
  `$var.<name>`. Compose snapshots the referenced names + values into the
  compiled JSON's `_meta.vars` — provenance for what this compilation used.
- **Errors are instructions.** Compose fails closed with `asset_not_pushed`
  (no lock entry / file gone), `asset_stale` (file changed since its last
  push) — both hint exactly `run: comfy assets push` — or `var_not_defined`
  (add the name under `vars:`). Run the hint, re-compose. `$asset` / `$var`
  outside a project hint `comfy project init` first.
- **Provenance is queryable state, not prose.** compose and run append to
  the `.comfy/runs.jsonl` journal automatically (best-effort, never fails a
  run); `comfy --json project status` joins assets against the lock and
  returns `recent_runs`. Never hand-write a manifest.md.
- **Routing**: `defaults.where` in `comfy.yaml` routes every command run
  inside the project tree (discovery walks up from cwd). The precedence
  chain in Ground rules applies — flag and env always win over the project.
- `project status` warns on unknown top-level dirs (warnings only, nothing
  enforces). `comfy project init` inside an existing project errors with
  `project_already_exists`; `project status` / `assets push` outside one
  error with `project_not_found`.

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
# → exit 0 / ok:true only on completed; failed job exits 1 / ok:false /
#   error.code=execution_error (payload under error.details); cancelled exits 130.
# `&& next-step` therefore only proceeds on success.

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

`--notify` fires a desktop notification when the job is terminal.
It defaults **on** in pretty/human async mode, and **off** in JSON/agent
contexts and with `--wait`. Override explicitly with `--notify` or `--no-notify`.

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

- **Spend gate — generation spends the user's Comfy credits and requires
  consent.** Interactive TTY runs confirm before spending; `--json` / non-TTY
  runs **fail closed** with error code `spend_consent_required` (exit 1,
  nothing spent) unless consent is supplied. Pass `--yes` only when the human
  has actually approved the spend — do not reflexively add it to make the
  error go away. A human can persist always-proceed with
  `comfy generate consent always` (revert: `consent ask`; inspect:
  `consent show`). `list` / `schema` / `refresh` / `upload` / `resume` /
  `--emit-workflow` spend nothing and are not gated.
- **Machine-readable output:** `generate <model> --json` prints the raw API
  response as JSON; `generate upload <file> --json` emits structured
  `{url, expires_at, …}`; `generate <model> --emit-workflow out.json` goes
  through the full renderer envelope (`command: "generate emit-workflow"`,
  error code `emit_workflow_failed`) — the output is a runnable partner-node
  workflow you can compose with (fragments+`run` route, no extra API key).
  The default pretty path (no flags) is still human-only — do not parse it.
- **`--emit-workflow` resolves the escape-hatch vs. quality tradeoff:**
  fragments+`run` is the default for graph work; `generate` is the
  highest-quality single-shot for partner models. With
  `--emit-workflow out.json` you get a runnable workflow you can extend,
  compose with other fragments, or iterate on — it's not a dead end.
- **`generate upload --json`** returns structured JSON; without `--json` the
  signed URL may soft-wrap across terminal lines — use `--json` in scripts.
- **Prefer sync** (plain `--download`, no `--async`): the CLI polls internally
  and waits for you (that's the tool blocking, not you sleep-polling), so an
  expensive video gen can't be orphaned. Reach for `--async` + `resume` only
  when you deliberately want to detach.
- **I2V pattern:** `generate <i2v-model>` needs an image **URL**, so the flow is
  `generate <image-model> --download still.png` → `generate upload still.png --json`
  → `generate <i2v-model> --image <url> --download clip.mp4`.

## Pre-flight — validate before you submit

Before `comfy run`, verify the workflow will succeed:

```bash
# Free dry-run: prints the exact API-format graph that WOULD be submitted,
# exits without POSTing — works on both local and cloud routes.
comfy run --workflow wf.json --print-prompt

# Full pre-flight: checks class_types, input shapes, enum values, edge wiring
comfy --json validate --workflow api.json

# Spot-check a single node class exists on the target
comfy --json nodes show <ClassName>
# If error.code == "node_not_found", check details.close_matches

# Confirm a model filename is actually available on the resolved backend (cloud-only)
# On local: use `comfy models list-folder <type>` instead
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

Terminal envelopes (`run --wait`, `jobs status`, `jobs watch`) carry the
flat `outputs` list plus grouped views of the same artifacts:
`outputs_by_node: {node_id: [url]}` always, and `outputs_by_item:
{item: [url]}` when the workflow came from a `foreach` blueprint (compose
embeds the item map — see Multi-stage orchestration). **Never identify
fan-out outputs by array order** — read `outputs_by_item`.

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

**In a project, prefer `comfy assets push`** — it uploads new/changed files
under `assets/`, records server names in the lock, and blueprints just say
`$asset.<relative/path>` (see Projects). The raw commands remain as the
non-project fallback:

```bash
# Non-project upload fallback: take data.uploads[0].cloud_name from the envelope
comfy --json upload photo.png --where cloud

# Download outputs from a completed job (works in or out of a project)
comfy --json download <prompt_id>

# Pipe pattern — the idiomatic way to generate + collect:
comfy --json run --workflow blueprints/<name>.compiled.json --wait | comfy download
# → submits, waits, downloads outputs in one pipeline
```

`comfy download` reads prompt_id + output URLs from piped stdin
automatically — no manual extraction, no `jq`, no API key exposure.

Output directory: the governing project's `outputs/`, else `./outputs/`
(override with `--out-dir`). Naming: outputs from a `foreach` blueprint
are named by originating item — `<item>_<nnn>.<ext>` with a per-item
counter — and `files[]` entries carry `node_id`/`item` when known;
everything else keeps `<prompt8>_<idx>.<ext>`.

## Project layout convention

The project/1 convention above IS the layout: `comfy.yaml` +
`assets/ fragments/ blueprints/ outputs/ .comfy/`. Lay it down with
`comfy project init`; `comfy --json project status` reports unknown
top-level dirs as warnings (warnings only, nothing enforces).

**Never write workflows, fragments, or outputs to `/tmp`.** Always work
inside a project. If none governs the directory, run `comfy project init`
before building anything.

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
want the maintainers to know? (totally optional)"* It's especially worth a nudge
after a session with real **friction** — a papercut you hit repeatedly, a
workaround you had to invent, a missing flag (a multi-job wait, a cost readout).
That's the highest-signal feedback; surface it in the user's words rather than
let it evaporate.

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
- Partner API escape hatch (one-shots only, via the proxy — not a workflow Job): `comfy generate flux-ultra --prompt "..."`
- Never hardcode checkpoint/LoRA names — discover via `models search`

## Video

- **SaveVideo is REQUIRED** — video API nodes produce VIDEO but are NOT output nodes
- **Never hardcode fps** — wire from GetVideoComponents output index 2
- Motion prompts: describe HOW the scene moves, not WHAT is in it
- Assembly: GetVideoComponents → BatchImagesNode → CreateVideo → SaveVideo (ImageBatch is deprecated)
- Autogrow inputs (type COMFY_AUTOGROW_*, e.g. BatchImagesNode `images`): wire ONE slot key per
  connection — `"images.image0": [..], "images.image1": [..]` — never a single `images` link.
  `nodes show` prints the `wire_as` form; `comfy validate` rejects the bare form before submit.
- I2V pattern: LoadImage → I2VNode → SaveVideo (check `nodes show` for the I2V node)
- **Model enums mix t2v and i2v variants** — a node's `model` choices may include
  image-to-video-only models (e.g. `grok-imagine-video-1.5`) that fail at runtime
  without an `image` input. Capability isn't in the metadata: if the model name
  hints at i2v/image, wire an image or pick another model before burning a cloud run
- Provider clips come back off-spec (e.g. 5.042s @ 1924x1076) — ffprobe and
  normalize (crop/trim) every clip before concat/conform
- Audio sync: match durations — short audio = silent ending, long audio = truncated ending
- **Talking heads (KlingAvatarNode):** lip-sync is by construction — it animates
  the mouth FROM the `sound_file` you pass, so feed it the EXACT audio that plays
  under the shot (it carries non-speech like laughter fine, too). BUT it pads the
  **video** past the **audio** (trailing still frames), so clip video-duration >
  speech. Concatenating such clips raw drifts the voice progressively out of sync
  — trim each clip to its own audio length (+ a small freeze-held breath beat)
  before concat; never butt-join the raw clips.
- Survey first: `comfy nodes ls --produces VIDEO --exclude-deprecated`, `comfy nodes ls --category "partner/video*"`, `comfy templates ls --type video` — compare OSS, partner-API, and gallery before choosing

## Audio

- ACE-Step: timesignature is `"4"`, NOT `"4/4"`
- Duration on TextEncode AND EmptyLatentAudio MUST match
- Output format is FLAC, not MP3
- For instrumental: set lyrics to empty string `""`
- Wire both positive AND negative to same TextEncode output when cfg=1.0
- **Cloud can't load uploaded audio.** `LoadAudio` AND `VHS_LoadAudioUpload`
  enums are blind to uploads (only ever list `bedroom.mp4`), even though
  `LoadImage` sees uploaded images. Don't upload→LoadAudio — it wall-fails with
  no signpost. Generate audio **in-graph** (a TTS / music node) and wire it by
  connection into the consumer (e.g. KlingAvatar `sound_file`).
- **Emotional TTS:** `eleven_multilingual_v2` is a flat reader. For real emotion
  use `eleven_v3` + inline performance tags in the text (`[whispers]`,
  `[voice breaking]`, `[long pause]`) + lower `stability` (~0.30) for swing.
  `model.style` caps at **0.2** (>0.2 fails validation).
- **Accents the preset voices don't cover** (Irish, Italian, …): coaxing a preset
  with an accent tag is unreliable — `FB_Qwen3TTSVoiceDesign` (an `instruct`
  voice description + fixed `seed`) builds a genuine one. Reuse the SAME
  `instruct`+`seed` across all of a character's lines to keep the voice consistent.
- Survey first: `comfy nodes ls --produces AUDIO`, `comfy nodes ls --category "partner/audio*"`, `comfy templates ls --type audio` — compare before choosing

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
- Cloud uses HTTP polling (no WebSocket); `jobs watch` polls `/api/jobs/<id>`

---

# Multi-stage orchestration

Build one large workflow graph when possible — ComfyUI parallelizes
independent branches automatically. Only split into separate workflows when:
- An intermediate result needs human review before continuing
- Different stages need different routing (local vs cloud)
- The workflow would exceed server memory constraints

**Video productions: prefer ONE graph from keyframes to finished film.**
Generated VIDEO flows between nodes as wires — no save/upload round-trip —
so clips + music + assembly belong in a single graph:

```
LoadImage($asset.s1_first.png) ×N  →  N× first/last-frame i2v nodes
  ├→ per-scene SaveVideo            (side-taps: each clip saved for review)
  └→ N× GetVideoComponents → BatchImagesNode(images.image0…N, scene order)
       → CreateVideo(fps ← GetVideoComponents, audio ← music node) → SaveVideo
```

The side-taps mean ONE job emits both the reviewable clips AND the
assembled film; if review fails a scene, fix and re-run the graph (the
extra assembly cost is small next to the video generations). The
job-boundary alternative (save clips → download → `assets push` →
LoadVideo in a second graph) is the fallback for a genuine review gate —
and note its current limit: pushed *images* appear in LoadImage's choices,
pushed *videos* are not yet catalogued for LoadVideo on cloud, so the
cross-job video handoff requires local assembly today.

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
  - fragment: t2i               # ./fragments/t2i.json — a fragment YOU authored via the derivation loop
    alias: shot
    params:
      prompt: $item.prompt
```

```bash
comfy workflow compose blueprints/fan_out.yaml   # → blueprints/fan_out.compiled.json
RES=$(comfy --json run --workflow blueprints/fan_out.compiled.json)
comfy --json jobs watch "$(echo "$RES" | jq -r .data.prompt_id)"
```

This submits a single Job; the engine runs the independent branches
concurrently. Compose embeds `_meta` (`schema: compose/1`) provenance in
the compiled workflow — blueprint path plus which nodes belong to which
`foreach` item. `comfy run` strips it before submit (old servers are
unaffected) and stashes the map on the job state, so the terminal
envelopes report `outputs_by_item: {item: [url]}` and `comfy download`
names files `<item>_<nnn>.<ext>`. Read outputs by item, never by array
order. Avoid the old `PIDS=()` shell-loop pattern — it duplicates
scheduling the engine already does and gives you N jobs to babysit instead
of one. (For a pure prompt/seed sweep over the *same* graph, `comfy
workflow vary` is the right tool; see the `comfy-fragments` skill for the
full blueprint syntax.)

**The exception — when fan-out across separate jobs IS right.** A multi-shot
film built on **partner-API video/avatar nodes** (KlingAvatar, Kling i2v, Sora,
…) is the case the one-graph rule doesn't fit: each shot must fail, retry
(`transient_auth`), and QC **independently** — one mega-graph sinks every shot
on a single mid-run auth blip — and the final edit (cuts, ducking, loudnorm) is
**ffmpeg**, which can't live in a Comfy graph. There you DO submit N independent
jobs. The loop is: **submit N → wait on all at once → download → conform**:

```bash
# submit each shot, collect prompt_ids
for wf in workflows/s*.json; do
  comfy --json run --workflow "$wf" | jq -r .data.prompt_id
done > ids.txt
# block on the whole batch with one call (NOT a hand-rolled poll loop):
comfy --json jobs wait $(cat ids.txt)   # summary envelope; exit≠0 if any failed
#   --all waits on every tracked non-terminal job; --timeout bounds it;
#   a `settled` NDJSON event fires per job as it finishes (--json-stream)
xargs -a ids.txt -n1 comfy --json download   # then conform with ffmpeg (in shell)
```

The ffmpeg conform stage legitimately stays in shell — `comfy` defines what the
graph does, not the limit of what you do with its outputs. This is the
*deliberate* per-shot fan-out the `foreach` advice rules out for **in-engine**
work; it is not the accidental `PIDS=()` babysitting loop (`jobs wait` replaces
that). Keep the per-shot `job_ids.txt` / `LOG.md` discipline from `comfy-director`.

With `chunk: N` in a `foreach` blueprint, compose splits items into
N-item batches and writes one numbered file per batch. The envelope then
reports `out: null` plus `written[]` (all paths) — script against
`data.written`, not `data.out`.

**On cloud, prefer ONE graph over a stage handoff.** If a generated image
feeds the next step (e.g. keyframe → image-to-video), wire the producer's
**IMAGE output straight into the next node's image input in the same
workflow.** The image stays an in-graph tensor on the server — **no download,
no re-upload, no files.** A cross-job handoff instead does cloud → local →
cloud and **re-transmits the full file bytes** (`comfy upload` always re-sends
the body; the cloud has no exists-by-hash skip), so it's pure waste when the
graph could have expressed the edge.

Split into separate jobs (and pay the handoff) only when you genuinely need to
**review the intermediate before spending** on the next stage — e.g. QC a
keyframe before burning a 5-minute video run. That trade is often worth it;
just don't split out of habit.

Stage handoff, when you do need it (download → promote into assets/ → push → `$asset`):

```bash
# `jobs watch` exits 0 only on completion — use && to chain safely:
comfy --json jobs watch "$PID" && comfy --json download "$PID"
# downloads land in outputs/ (item-named) — promote the keepers:
cp outputs/s1_000.png assets/s1_first.png
comfy --json assets push
# the next stage's blueprint references it directly in its inputs:
#   inputs: {start_frame: $asset.s1_first.png}
```

Outside a project, the legacy handoff still works: `comfy --json upload
<file> --where cloud`, take `data.uploads[0].cloud_name`, paste it into
the next workflow's LoadImage input — but it re-sends the bytes; prefer one
graph, or the project flow.

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
