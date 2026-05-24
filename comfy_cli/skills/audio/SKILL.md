---
name: comfy-audio
description: Audio and music generation patterns — ACE-Step music, text-to-speech, audio loading. Teaches the agent how to generate soundtracks and sync audio with video via comfy CLI.
---

Audio generation in ComfyUI. For async/routing patterns see [comfy],
for video+audio assembly see [comfy-video], for multi-stage pipelines
see [comfy-pipeline].

## Discover audio nodes

```bash
comfy --json nodes ls --produces AUDIO --limit 20
comfy --json nodes ls --category "api node/audio*" --limit 10
comfy --json nodes categories --prefix "api node/audio"
```

## ACE-Step music generation: the standard pattern

The standard ACE-Step pattern:
`CheckpointLoaderSimple → TextEncodeAceStepAudio1.5 → EmptyAceStep1.5LatentAudio → KSampler → VAEDecodeAudio → SaveAudio`.

Discover available checkpoints: `comfy --json nodes show CheckpointLoaderSimple`.
Discover TextEncode inputs: `comfy --json nodes show TextEncodeAceStepAudio1.5`.

Key details:

- Wire both positive AND negative to the same TextEncode output `["2", 0]` when using cfg=1.0 (check defaults via `comfy --json nodes show KSampler`)
- `duration` on TextEncode AND EmptyLatentAudio must match
- `timesignature`: use `"4"` not `"4/4"`
- `keyscale` choices: verify via `comfy --json nodes show TextEncodeAceStepAudio1.5`
- Output format is FLAC. SaveAudio produces `.flac` files.
- For instrumental: set `lyrics` to empty string `""`
- `tags` is a comma/space-separated list of style descriptors (genre, mood, instruments)
- BPM range: 60–180 typical. Slower = more atmospheric, faster = more energetic

## Loading audio for video assembly

- `LoadAudio`: takes a COMBO (filename from input directory)
- Upload first: `comfy upload music.flac --where cloud`
- Use the cloud name in the workflow:
  `{"class_type": "LoadAudio", "inputs": {"audio": "hash.flac"}}`
- Wire LoadAudio output `[0]` (AUDIO) into CreateVideo's optional audio
  input

## Duration matching

- Generate audio to match your target video length
- Match your total video length — e.g. 30s video → generate 30s audio
- If audio is shorter than video, the end of the video is silent
- If audio is longer than video, the audio is truncated (you lose the
  ending)
- Plan duration before generating — changing it later means regenerating

## Partner API audio nodes

- **ElevenLabs**: text-to-speech, voice cloning
- **Sonilo**: video-to-music (generates music that matches a video's mood)
- Discover: `comfy --json nodes ls --category "api node/audio*"`
- These handle auth automatically via cloud session

## What NOT to do

- Don't use `timesignature: "4/4"` — it's `"4"`
- Don't forget to match duration between TextEncode and EmptyLatentAudio
- Don't expect MP3 output — SaveAudio produces FLAC
- Don't hardcode keyscale without checking choices via `nodes show`
