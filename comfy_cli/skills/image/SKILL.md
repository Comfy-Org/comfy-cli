---
name: comfy-image
description: Image generation patterns — text-to-image, variations, upscaling, ControlNet, style transfer. Teaches the agent how to explore models, build workflows, and batch-generate via comfy CLI.
---

You have the `comfy` CLI installed. This skill teaches image generation
patterns — from single shots to batch sweeps — using whatever models and
nodes the user's environment has available.

## Golden rule: start from a template, then slot-edit

The `Comfy-Org/workflow_templates` gallery has hundreds of pre-built
image and use-case templates. **Use one as the starting point; don't
build from raw nodes unless none of them fit.** Discover what's
available: `comfy --json templates ls --type image --limit 5`

```bash
# 1. Find a template that matches the user's intent.
comfy --json templates ls --type image --tag "Text to Image" --limit 10
comfy --json templates show <name>             # see what models + providers it uses

# 2. Pull the workflow JSON.
comfy --json templates fetch <name> --out ./workflows/start.json

# 3. See what's tweakable.
comfy --json workflow slots ./workflows/start.json

# 4. Find a model to plug into a slot.
comfy --json models search --type checkpoint --text "flux"     # cloud: enriched
comfy --json models search --type lora --text "wan2.2"

# 5. Edit slots and submit. Cloud auto-converts UI→API on submit.
comfy workflow set-slot ./workflows/start.json positive.text="a cat in space"
comfy --json run --workflow ./workflows/start.json --where cloud --wait
```

**Never hardcode a checkpoint or LoRA name** — the catalog changes
monthly. Always discover via `models search` (cloud: enriched metadata;
local: filenames). Skip the template step only when the user explicitly
wants something the gallery doesn't cover.

## Building from scratch (rare path)

If no template fits and you must build the graph yourself, use the node
introspection commands to find the wiring:

```bash
comfy --json nodes path MODEL IMAGE --max-paths 3
comfy --json nodes upstream KSampler
comfy --json nodes show KSampler   # see all inputs + defaults
```

**Choosing a model:** Discover available checkpoints via
`comfy --json models search --type checkpoint`. For fastest iteration,
use partner-API image nodes
(`comfy --json nodes ls --category "api node/image*"`) — or skip the
workflow entirely with `comfy generate <model> --prompt "…"`
(see `comfy generate list` for available models).

### Sampler defaults

Sampler defaults vary by model. Discover valid choices:
`comfy --json nodes show KSampler | jq '.data.inputs[] | select(.name=="sampler_name") | .choices'`.
The node's `choices` field is always the source of truth.

## Variations: the batch/sweep pattern

For "generate 10 images with different prompts/seeds":

```bash
# 1. What slots can I tweak?
comfy --json workflow slots my_workflow.json

# 2. Produce N variations (lists are zipped — same length required)
comfy workflow vary my_workflow.json \
    --slot positive_prompt.text='["a cat in space","a dog on mars","a fox underwater"]' \
    --slot sampler.seed='[42,77,123]' \
    --out-dir ./variants

# 3. Submit all in parallel, capture prompt_ids
PIDS=()
for f in ./variants/*.json; do
    PID=$(comfy --json run --workflow "$f" --where cloud | jq -r .data.prompt_id)
    PIDS+=("$PID")
done

# 4. Download all outputs
for p in "${PIDS[@]}"; do
    comfy --json jobs watch "$p" --where cloud | comfy download --where cloud
done
```

All jobs submit async by default and run in parallel on cloud.

## Upscaling

Find what's available:

```bash
comfy --json nodes search "upscale" --limit 10
comfy --json nodes ls --category "upscaling*"
comfy --json nodes show UpscaleModelLoader  # what upscale models are installed
```

Common pattern: generate at lower res, then upscale. Chain two workflows
or build one with the upscale nodes inline. For detailed upscale patterns
and API upscale options, see `comfy-edit`.

## ControlNet / guided generation

For ControlNet, depth, canny, pose, and other structural guides, see the
`comfy-condition` skill.

## Partner API nodes (cloud-only)

```bash
comfy --json nodes ls --category "api node/image*"
```

Auth is handled automatically via the user's cloud session.

## Multi-stage pipelines

For chaining image → video → assembly, see the **comfy-pipeline** skill.
It covers upload/download composition, parallel fan-out, and the pipe operator.

## What NOT to do

- **Don't hardcode model filenames.** They vary per installation. Always
  discover via `nodes show`.
- **Don't assume VRAM.** Let the user choose resolution and batch size.
  Cloud has more headroom than local.
- **Don't build from scratch when a template exists.** Ask the user if they
  have an existing workflow to start from.
- **Don't poll `jobs status` in a loop.** Use `jobs watch` (blocks until
  done) or pipe to `comfy download`.
- **Don't extract API keys manually.** Use `comfy upload` and `comfy download`
  instead of raw curl with credentials.
