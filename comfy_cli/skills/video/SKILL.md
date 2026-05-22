---
name: comfy-video
description: Video generation patterns — image-to-video, text-to-video, video assembly, audio sync. Teaches the agent how to animate images, concatenate clips, and produce final videos via comfy CLI.
---

This skill is composable — it assumes the `comfy` (core) skill is already
loaded. It does NOT repeat the output contract, routing, or async patterns.
See the comfy skill for the async submit pattern.

## Golden rule: discover video nodes first

Video is not just Kling. The graph has 80+ video nodes across Kling,
Wan/HappyHorse, LTX, Grok, ByteDance/Seedance, Runway, and more.

Always discover what's available:

```bash
comfy --json nodes ls --produces VIDEO --limit 10
comfy --json nodes ls --category "api node/video*" --limit 20
comfy --json nodes categories --prefix "api node/video"
```

Check the specific node's inputs before building a workflow:

```bash
comfy --json nodes show KlingImage2VideoNode
comfy --json nodes show HappyHorseImageToVideoApi
```

## Image-to-Video (I2V): the core pattern

Most I2V nodes follow: `LoadImage → I2VNode → SaveVideo`

**Critical:** I2V API nodes produce VIDEO but are NOT output nodes. You
MUST add SaveVideo or the workflow produces no file.

Discover available I2V nodes and their inputs:
`comfy --json nodes show KlingImage2VideoNode`

**Choosing a video provider:** Kling (highest quality, 2-5min), Wan/HappyHorse
(good quality, faster), Veo (text-to-video with audio). Discover all:
`comfy --json nodes ls --category "api node/video*"`

Always verify model names, duration choices, and mode options via
`comfy --json nodes show <NodeName>` — they vary by provider and update
frequently.

## Text-to-Video (T2V)

Some nodes support direct text-to-video without a start frame.
Examples: `KlingTextToVideoNode`, `GrokVideoNode`,
`HappyHorseTextToVideoApi`.

Same pattern: `T2VNode → SaveVideo`

## Video assembly: concatenating clips

The key pipeline for multi-clip videos:

```
LoadVideo → GetVideoComponents → (IMAGE frames, AUDIO, FLOAT fps)
                                        ↓
                                   ImageBatch (chain multiple)
                                        ↓
                              CreateVideo (frames + audio) → SaveVideo
```

Critical nodes:

- **`GetVideoComponents`**: VIDEO → [0] IMAGE (frame batch), [1] AUDIO,
  [2] FLOAT (fps)
- **`ImageBatch`**: IMAGE + IMAGE → IMAGE (concatenates frame batches).
  Deprecated but reliable in API format.
- **`CreateVideo`**: IMAGE (frames) + optional AUDIO → VIDEO. fps input
  is FLOAT.
- **`SaveVideo`**: VIDEO → (output node, saves to disk)
- **`Video Slice`**: VIDEO → VIDEO (trim by start_time + duration)

### THE FPS LESSON (critical, learned the hard way)

- Kling outputs 24fps video. Other models may differ.
- When assembling, wire the fps FROM the source:
  `"fps": ["4", 2]` (GetVideoComponents output index 2)
- **NEVER hardcode `"fps": 30.0`** — it will speed up or slow down the
  video silently.
- 3 × 10s clips at 24fps = 723 frames. At 30fps that plays as 24.1s.
  At 24fps it plays as 30.1s. The math matters.

## Audio sync

- **`LoadAudio`**: loads a `.flac`/`.wav` from the input directory
- **`CreateVideo`** accepts optional `audio` input at index 2
- For 30s video: generate 30s audio (ACE-Step with `duration: 30.0`),
  then wire into CreateVideo
- Audio and video durations should match — if video is 30.1s and audio
  is 30.0s, the last 0.1s will be silent (acceptable). If reversed (24s
  video, 30s audio), the audio gets truncated and the ending is lost.

## Motion prompts: what works

I2V motion prompts describe **HOW the scene moves**, not WHAT is in it
(the image provides that).

- **Good**: "gentle camera pan left, soft wind blowing through hair,
  clouds drifting slowly"
- **Bad**: "a beautiful woman standing in a field" (duplicates the image
  content)

Keep motion prompts short and specific. Describe: camera movement,
subject action, environmental motion.

Always include negative: "blurry, distorted, low quality, watermark,
text overlay"

## What NOT to do

- Don't forget SaveVideo — API video nodes are not output nodes
- Don't hardcode fps — wire it from GetVideoComponents
- Don't use `"format": "video/mp4"` — the choice is `"mp4"` not a MIME
  type
- Don't assume all video models output the same fps
- Don't chain `comfy jobs watch` calls sequentially for independent
  videos — submit all in parallel, then collect (see comfy-pipeline skill)
- For first-frame/last-frame transitions between clips, see `comfy-condition`.
