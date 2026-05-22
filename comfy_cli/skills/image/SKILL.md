---
name: comfy-image
description: Image generation patterns — text-to-image, variations, upscaling, ControlNet, style transfer. Teaches the agent how to explore models, build workflows, and batch-generate via comfy CLI.
---

You have the `comfy` CLI installed. This skill teaches image generation
patterns — from single shots to batch sweeps — using whatever models and
nodes the user's environment has available.

## Golden rule: discover, don't prescribe

Models change monthly. **Never hardcode a checkpoint or LoRA name.** Instead,
discover what's installed at runtime:

```bash
# What checkpoints are available?
comfy --json nodes show CheckpointLoaderSimple | jq '.data.inputs[] | select(.name=="ckpt_name") | .choices'

# What LoRAs?
comfy --json nodes show LoraLoader | jq '.data.inputs[] | select(.name=="lora_name") | .choices | length'

# What samplers / schedulers?
comfy --json nodes show KSampler | jq '.data.inputs[] | select(.name=="sampler_name") | .choices'
```

Pick models by asking the user what style they want, then filter the
available list. If unsure, list a few options and let the user choose.

## Text-to-image: the basic pattern

To find the wiring for a t2i pipeline:

```bash
comfy --json nodes path MODEL IMAGE --max-paths 3
comfy --json nodes upstream KSampler
comfy --json nodes show KSampler   # see all inputs + defaults
```

**Choosing a model:** For photorealism or general purpose, prefer Flux
(cfg=1.0). For stylized/anime, check available SDXL checkpoints. For
fastest iteration, use API image nodes
(`comfy --json nodes ls --category "api node/image*"`).

### Sampler defaults

Sampler defaults vary by model. Discover valid choices:
`comfy --json nodes show KSampler | jq '.data.inputs[] | select(.name=="sampler_name") | .choices'`.
The node's `choices` field is always the source of truth.

## Variations: the batch/sweep pattern

This is the most common power-user request: "generate 10 images with
different prompts/seeds."

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
