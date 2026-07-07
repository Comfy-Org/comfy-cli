# Keyframe Relay — the "Impossible Oner"

**Date:** 2026-06-14
**Author:** Kishore (with Claude)
**Status:** Built — v2 (proper) shipped as `keyframe-relay/outputs/UNMADE_final.mp4`
**Routing:** Comfy Cloud (signed in, OAuth)

## 1. Problem & goal

Create a **novel video-generation technique** whose results are obviously good.
Two constraints set by the user:

- **Novelty axis = "both"** — the *method* must be new *and* it must produce a
  look that method directly creates (not a stock pipeline with a new prompt).
- **Win condition = "impossible-to-fake coherence"** — the result must clearly
  not be one lucky single generation. It must read as authored, continuous, and
  identity-stable.

This is explicitly **a workflow** (a multi-node graph orchestrated across
stages), not a single prompt to a video model. Beating "one lucky generation"
is the whole point.

## 2. The technique: Keyframe Relay

A single creature is **always recognizably itself** while it travels through a
chain of states that **cannot physically coexist** (different materials,
different worlds), in **one unbroken continuous take**.

Coherence is unfakeable two ways at once:

1. **Identity lock.** Each anchor is *edited from the previous anchor* (image to
   image via an edit model), so the same creature is provably carried forward —
   never re-rolled from scratch. Identity is additionally **reinforced inside
   the relay sampler** via a reference mechanism, so the creature holds *through*
   each morph, not only at the endpoints.
2. **Controlled transitions.** Every morph is interpolated between two fixed,
   hand-authored endpoints (`WanFirstLastFrameToVideo`), so it is authored,
   reproducible, and continuous — not a dice roll.

### Subject

A small creature. **Identity thread held constant the entire film:** the same
spiral eye-markings, the same two curled antennae, the same head-tilt and
silhouette. Everything else transforms.

### Morph arc (5 anchors → 4 relay segments → ~20s oner, slow push-in)

| # | Material            | Setting                          |
|---|---------------------|----------------------------------|
| 0 | folded wet **paper**| rain-soaked windowsill           |
| 1 | molten **blown glass** | dark furnace                  |
| 2 | **coral & barnacle**| sunken shipwreck                 |
| 3 | cracked **circuitry & LED** | derelict server room     |
| 4 | living **moss & lichen** | reclaimed forest ruin, fireflies |

Surprise compounds across **material + setting** every beat while the
**subject** stays fixed. (Per the "triadic surprise" principle: weirdness must
compound across axes, not sit on one.) All states stay inside AI's strengths —
faces/creatures, atmosphere, slow camera; no hands, no physics, no text.

## 3. Pipeline (4 stages)

```
Stage 1  seed_anchor      t2i  ───────────────► anchor[0].png
Stage 2  identity_chain   edit(anchor[k-1], "same creature, now <state k>")
                          ×4  ───────────────► anchor[1..4].png   (identity lock)
Stage 3  relay_segment    WanFirstLastFrameToVideo(start=anchor[k],
                          end=anchor[k+1], reference=anchor[0])
                          + KSampler + VAEDecode
                          ×4  ───────────────► seg[0..3].mp4      (the look)
Stage 3b RIFE finishing   RIFE VFI on each seg ► seg[k]_smooth.mp4
Stage 4  conform + concat ffprobe-normalize + concat ► impossible_oner.mp4
```

### Stage details

- **Stage 1 — seed anchor.** Text-to-image of the creature in state 0 (paper).
  Establishes the identity thread (eyes, antennae, silhouette) in the prompt.
- **Stage 2 — identity-lock chain.** For k=1..4, run an **edit model** on
  `anchor[k-1]`: *"keep this exact creature — same spiral eye-markings, same
  curled antennae, same head-tilt and silhouette — but now made of <material k>,
  in <setting k>."* Candidate edit nodes (selected at plan time by quality on a
  test edit): `GeminiNanoBanana2`, `FluxKontextProImageNode`, `GrokImageEditNode`,
  `ByteDanceSeedreamNode`, Qwen-edit. The creature is literally carried
  image-to-image down the chain.
- **Stage 3 — relay interpolation.** For each consecutive pair (k, k+1):
  `WanFirstLastFrameToVideo(start=anchor[k], end=anchor[k+1])` → latents →
  `KSampler` → `VAEDecode` → frames. The Wan model fills the impossible
  in-between. Slow push-in camera prompt per segment.
  - **Identity reinforcement (in-relay lock).** Inject `anchor[0]` (or a clean
    reference crop) as a reference signal into the Wan sampling so identity holds
    mid-morph. **This is the riskiest wiring** — see §6 spike.
- **Stage 3b — RIFE finishing pass.** `RIFE VFI` (or `FrameInterpolate`) on each
  segment to raise fps and smooth the morph for an obviously-premium oner.
- **Stage 4 — conform + concat.** `ffprobe` every segment (provider/OSS clips
  come back off-spec), normalize fps/resolution/duration, concat end-to-end into
  one continuous take. Optional slow continuous audio bed under the whole thing.

## 4. Build mechanics (comfy CLI)

Template-first, then fragments + blueprint (per the comfy decision tree). Start
by checking for an existing **Wan first/last-frame template**
(`comfy templates ls --type video`) and fetch it as the Stage-3 base rather than
hand-building the Wan sampler graph.

Project layout (created under `comfy-cli/` working dir, **not** /tmp):

```
keyframe-relay/
├── fragments/
│   ├── seed_anchor.json          # t2i → IMAGE
│   ├── anchor_edit.json          # edit(image, instruction) → IMAGE   (reused ×4)
│   ├── relay_segment.json        # Wan first/last + sampler + decode → frames/VIDEO (reused ×4)
│   └── rife_smooth.json          # RIFE VFI → VIDEO                    (reused ×4)
├── blueprints/
│   └── keyframe_relay.yaml        # wires the chain; $alias.output refs
├── workflows/                     # compose output
├── inputs/
├── outputs/                       # anchors, segments, final oner
└── variants/
```

Orchestration is **multi-submission** (each stage's output feeds the next),
because the anchor chain is inherently sequential (anchor k needs anchor k-1).
Relay segments, once anchors exist, can fan out in parallel.

## 5. Success criteria

- One continuous video where the creature is recognizably the same throughout
  all five states (identity holds — verifiable by eye across the whole clip).
- Transitions are smooth, continuous morphs between materials/settings — no hard
  cuts, no popping.
- The result is self-evidently *not* a single generation: it spans incompatible
  realities while staying coherent.
- Reproducible: re-running with the same anchors + seeds yields the same oner.

## 6. Risks & spikes

- **[SPIKE — highest risk] In-relay identity reinforcement on Wan.** IPAdapter is
  SD/SDXL-era; Wan 2.2 uses its own reference conditioning (VACE-style / reference
  latent), not classic IPAdapter. Plan must spike *how* to inject `anchor[0]` as a
  reference into `WanFirstLastFrameToVideo` sampling. **Fallback if too fiddly:**
  drop to anchors-only identity (the edit-chain alone is already strong), and lean
  on a tighter edit-chain + the RIFE pass. Decision gate: build the glass→coral
  segment both ways, keep the one with less mid-morph drift.
- **Off-spec clips.** OSS Wan output may vary in fps/res/length; Stage 4 ffprobe
  normalization is mandatory before concat.
- **Edit-model drift.** Over 4 chained edits, identity can erode. Mitigate by
  always re-referencing the *original* anchor[0] in edit prompts, and by
  re-injecting anchor[0] as a reference image into the edit node where supported.
- **Cloud spend.** ~5 image edits + 4 Wan video gens + 4 RIFE passes. A few
  minutes per video gen. Acceptable for a demo; note it before submitting.
- **Transition violence.** Very different materials (glass→coral) may morph
  wildly. That is the intended effect, but the slow push-in + reference lock keep
  it readable.

## 7. Out of scope (YAGNI)

- Dialogue, lip-sync, characters beyond the one creature.
- Hard cuts / multi-shot editing (this is deliberately a single oner).
- Partner-API relay engines (Kling/Runway/Veo) — OSS Wan chosen; partner nodes
  remain a documented fallback only.
- Generalizing into a reusable parameterized tool — that is a possible *next*
  project once the one-off demo proves the technique.

---

## v2 — the "proper" technique (research-grounded, as built)

Web research into the continuous-transformation genre (esp. Seungho Yeo /
@seungho__yeo "Cosmic Dust", and the documented WAN-VACE first/last workflow)
established that the pro method is **WAN-VACE first/last frame, CHAINED on the
real rendered last frame** — not concatenated clean anchors.

**What changed from v1:**
- **True last-frame relay (spine, not reserve):** segment k's *actual final
  rendered frame* becomes segment k+1's start frame (uploaded back to cloud),
  so joins are frame-exact. Eliminates the seam-pop of anchor concatenation.
  Pipeline is therefore serial.
- **Reference pin confirmed mandatory:** the no-reference test (5abcdfe4)
  dissolved mid-morph into a flower field; `WanVaceToVideo.reference_image` =
  the clean start anchor of each morph holds the creature through the middle.
- **Shorter segments (49f / ~3s):** less unconstrained middle to hallucinate.
- **Story layer:** "Unmade" — one creature survives the end of its world over
  and over by remaking its body from the wreckage (paper→glass→coral→circuit→
  moss = birth→fire→flood→cold death→rebirth). The relay *is* the narrative.
- **Audio:** Stable Audio 2.5 score (one arc) + 5 per-world ambience beds,
  crossfaded to each world's time window, mixed and loudness-normalized to
  -14 LUFS.
- **Finish:** 1.5s/2.0s intro/outro holds + ffmpeg `minterpolate` 16→32fps
  (RIFE-equivalent, QC'd clean).

**Final:** `keyframe-relay/outputs/UNMADE_final.mp4` — 832x480, 32fps, 15.7s,
H.264 + AAC. Full job ledger in `keyframe-relay/job_ids.txt`, event log in
`keyframe-relay/LOG.md`.

**Known limits / next:** 832x480 is demo-res (add a Topaz/ESRGAN upscale for
finals); a real RIFE-VFI cloud pass would beat `minterpolate` on the brightest
transition flashes; segment length is fixed (Stage-B clip-extend could give any
world more screen time).
