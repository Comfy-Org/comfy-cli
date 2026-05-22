---
name: comfy-condition
description: Guide generation with structure — ControlNet, masks, reference images, motion control. The skill for telling models WHERE and HOW to generate, not just WHAT.
---

Generation without conditioning is a coin flip. This skill teaches how to
constrain output — pose, depth, edges, motion trajectories, reference
consistency. Composable with `comfy-image` (conditioned generation),
`comfy-video` (motion control), and `comfy-edit` (masked inpainting).

## Discover conditioning nodes

```bash
# ControlNet nodes
comfy --json nodes search "controlnet" --limit 10
comfy --json nodes ls --category "conditioning/controlnet*"

# Preprocessors (image → condition map)
comfy --json nodes search "canny" --limit 5
comfy --json nodes search "depth" --limit 5
comfy --json nodes search "pose" --limit 5

# Mask operations
comfy --json nodes ls --produces MASK --limit 10
comfy --json nodes ls --accepts MASK --limit 10
```

## ControlNet: the core conditioning primitive

Discover available preprocessors and models:

```bash
comfy --json nodes search "canny" --limit 5
comfy --json nodes search "depth" --limit 5
comfy --json nodes show ControlNetLoader
comfy --json nodes show ControlNetApplyAdvanced
```

Wiring: ControlNet apply node takes CONDITIONING (from CLIPTextEncode)
+ IMAGE (the condition map) + CONTROL_NET (from ControlNetLoader) and
outputs modified CONDITIONING. Feed that into KSampler instead of the
raw text conditioning.

## Masks: spatial selection

MASK type = grayscale image where white = selected region.

Common sources:
```bash
comfy --json nodes search "mask" --limit 15
# → SolidMask, ImageToMask, MaskComposite, InvertMask, etc.
```

Masks compose: `MaskComposite` combines masks. `InvertMask` flips them.
Feed masks into inpainting nodes or `SetLatentNoiseMask` for
region-specific generation.

## Reference images: consistency control

Some nodes accept a reference image to maintain subject/style consistency:

```bash
comfy --json nodes search "reference" --limit 10
comfy --json nodes search "IP-Adapter" --limit 5
# API nodes with reference support:
comfy --json nodes show FluxKontextProImageNode
comfy --json nodes show GrokImageEditNodeV2
```

## First-frame / last-frame (video transitions)

The most powerful video conditioning technique. Instead of prompting motion,
you provide the exact start and end images — the model fills in between.

```bash
comfy --json nodes show KlingStartEndFrameNode
# Inputs: start_frame (IMAGE), end_frame (IMAGE), prompt, mode, cfg_scale
# Mode choices: "pro mode / 5s duration / kling-v2-5-turbo", etc.
```

Use `comfy --json nodes show KlingStartEndFrameNode` to discover inputs.
Key pattern: wire start_frame and end_frame from two LoadImage nodes.
Chain N key frames into N-1 transitions. Concatenate via `comfy-video`
assembly pattern. Each transition is independently parallelizable.

## Motion control (video conditioning)

For video, conditioning means controlling camera or subject motion:

```bash
comfy --json nodes show KlingCameraControlI2VNode
comfy --json nodes show KlingCameraControls
comfy --json nodes show KlingMotionControl
```

Motion control is to video what ControlNet is to images — structural
guidance over the output. See `comfy-video` for the animation patterns
this feeds into.

## Composing with other skills

Conditioning is an **input modifier** — it doesn't produce final output,
it shapes how other skills generate:

- `comfy-condition` + `comfy-image` → structurally guided image generation
- `comfy-condition` + `comfy-video` → motion-controlled animation
- `comfy-condition` + `comfy-edit` → masked inpainting with region control
- All three + `comfy-pipeline` → multi-stage conditioned production

## What NOT to do

- Don't apply ControlNet without checking which models are installed
- Don't confuse the condition map (preprocessor output) with the ControlNet model (separate)
- Don't skip the preprocessor — feeding a raw photo into ControlNet won't work
- Don't hardcode ControlNet strength — it varies by model and use case
