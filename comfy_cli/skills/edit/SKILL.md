---
name: comfy-edit
description: Transform existing assets — edit images, upscale, inpaint, style transfer. The skill for modifying what already exists rather than generating from scratch.
---

Transform, don't generate. This skill covers every operation that takes an
existing image or video and produces a modified version. Composable with
`comfy-image` (generate first, then edit) and `comfy-pipeline` (chain stages).

## Discover edit nodes

```bash
# Image editing (Grok, Gemini, Qwen, BFL Kontext, etc.)
comfy --json nodes search "edit" --limit 15
comfy --json nodes ls --category "api node/image*" --limit 30

# Upscaling (local + API)
comfy --json nodes search "upscale" --limit 15

# Inpainting
comfy --json nodes search "inpaint" --limit 10
```

## Upscale: the most common transform

Two paths — local model or partner API:

```bash
# Local: what upscale models are available?
comfy --json nodes show UpscaleModelLoader
comfy --json nodes show ImageUpscaleWithModel
```

Local upscale pipeline (LoadImage → UpscaleModelLoader → ImageUpscaleWithModel → SaveImage):
```json
{
  "1": {"class_type": "LoadImage", "inputs": {"image": "input.png"}},
  "2": {"class_type": "UpscaleModelLoader", "inputs": {"model_name": "..."}},
  "3": {"class_type": "ImageUpscaleWithModel", "inputs": {"upscale_model": ["2", 0], "image": ["1", 0]}},
  "4": {"class_type": "SaveImage", "inputs": {"images": ["3", 0], "filename_prefix": "upscaled"}}
}
```

API upscale (cloud, no local GPU needed):
```bash
comfy --json nodes show MagnificImageUpscalerCreativeNode
comfy --json nodes show RecraftCrispUpscaleNode
comfy --json nodes show StabilityUpscaleFastNode
```

Always check `nodes show` for the available model names and parameters.

## Image editing via API nodes

Partner API nodes take an image + text prompt and return a modified image:

```bash
# What image edit nodes exist?
comfy --json nodes search "image edit" --limit 10
# → GrokImageEditNode, BriaImageEditNode, FluxKontextProImageNode, etc.
```

Pattern: LoadImage → EditNode → SaveImage. The edit node takes IMAGE +
prompt. Check each node's inputs — they vary by provider.

## Inpainting: mask + fill

Inpainting flow: Load image → generate/load MASK → feed both into inpainting.
For KSampler-based inpainting: `VAEEncode → SetLatentNoiseMask` (with the
mask) → `KSampler` with denoise=0.3-0.6. For API inpainting: nodes like
`FluxProFillNode` take IMAGE + MASK directly. Discover mask generators:
`comfy --json nodes ls --produces MASK --limit 10`

## Style transfer

For style transfer, use API edit nodes with a style-descriptive prompt,
or see `comfy-condition` for ControlNet-based structural style guidance.

## Composing with other skills

Edit is the middle of a pipeline. Common compositions:

- **generate → edit**: `comfy-image` generates, `comfy-edit` upscales or refines
- **edit → animate**: `comfy-edit` prepares a frame, `comfy-video` animates it
- **generate → edit → animate → compose**: the full pipeline (`comfy-pipeline`)

Use `comfy upload` / `comfy download` between stages (see `comfy-pipeline`).

When showing generated/edited images to the user, follow the `comfy-relay` conventions.

## What NOT to do

- Don't regenerate when you can edit — editing preserves the parts you like
- Don't skip `nodes show` — edit nodes vary wildly in their parameters
- Don't assume a local upscale model exists — check choices first, fall back to API
