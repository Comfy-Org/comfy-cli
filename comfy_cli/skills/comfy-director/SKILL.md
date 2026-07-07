---
name: comfy-director
description: Use when asked to make a narrative video — an ad, brand film, short, trailer, music video, or any multi-shot clip that tells a story (not a single shot or loop). Also use when a video was rejected for "no story", "doesn't flow", "feels like a montage", or characters that change between shots.
---

# Comfy Director

Story first, pictures second. Multi-shot AI films fail in the edit, not the
render: beats that don't cause each other read as a montage wearing a story's
clothes, no matter how good each clip looks.

**REQUIRED BACKGROUND:** the `comfy` skill (CLI mechanics, workflow hierarchy)
and `comfy-fragments` (composing multi-stage graphs). On any failed job,
`comfy-debug`.

## Order of operations

1. **Concept** — one idea a stranger can retell.
2. **Screenplay** — before touching the pipeline. Deliver it to the user first.
3. **Shot list** — each scene mapped to a continuity strategy.
4. **Audio design** — against the screenplay's emotional beats, not after picture.
5. **Produce** → **conform** → **QC**.

Skipping 2 and jumping to renders is the #1 cause of killed cuts.

## Screenplay rules

- **One protagonist** the camera can follow. Recognizable in every shot.
- **Causality test:** every cut answers "and BECAUSE of that…" — never
  "and then, somewhere else…". If a beat can be reordered without breaking
  anything, it isn't a story yet.
- **Stranger test:** someone with zero context must retell the plot in two
  sentences ("X wanted A, B stopped them, they did C, so D").
- Beats carry timings that sum to the target runtime.

## Concept rules

- The brand/product is **felt, not name-checked**: argue its philosophy;
  in-jokes and community lore quoted on screen read as fan-service.
- A technique the client mentions ("style transfer", a model name) is a
  **spark, not a mandate** — use it only if the idea needs it. Building the
  film around a technique is the tell of a vendor, not a director.
- Weirdness must compound: surprising subject AND object AND setting.
  One weird element in a normal frame is decoration.

## Prompting shots

- **Photoreal shots:** live-action cinematographer / film-stock language ONLY.
  One illustrator/anime/comics name in a photoreal prompt poisons it.
  Illustrated references belong only in deliberately stylized shots.
- Motion prompts describe HOW the scene moves, not what's in it (see `comfy`
  skill video gotchas — t2v/i2v model traps, SaveVideo, fps wiring).

## Continuity toolkit

Pick per shot, in rough order of strength:

| Strategy | Use for |
|---|---|
| Character/location reference image carried into each shot's i2v graph | protagonist identity |
| Same identity, new framings via reference-image edit (one hero → N angles) | recognizable face across cuts |
| Keyframe of shot N seeds shot N+1 (image-referenced restyle or end-frame) | contiguous action |
| Talking head fed the EXACT audio that plays (KlingAvatar), cut between angles of the *same* lip-synced head — never B-roll over the VO | dialogue that must feel real |
| Repeated wardrobe/light/lens lines verbatim in every prompt | cheap baseline glue |

If a shot can't hold continuity, **rewrite the screenplay** so it doesn't
need that shot — don't ship the break.

## Audio

- Score = **short designed cues joined at story pivots**; one long generation
  will not follow a dynamic arc (it inverts exactly where you need the peak).
- Place VO by math: stem duration vs beat window, before mixing.
- Duck score under VO; loudness-normalize the final mix (~-14 LUFS).
- **Dialogue must lip-sync by construction.** Drive the mouth from the exact
  audio that plays (KlingAvatar `sound_file`) and cut between *angles of the same
  lip-synced head* — B-roll over a continuous VO reads as a montage, and
  i2v-invented mouths drift. "No lip-sync / doesn't feel real" is the classic
  rejection. (Generate audio in-graph — cloud `LoadAudio` can't see uploads; see
  `comfy` audio gotchas.)
- **Lock voice & casting by EAR before the expensive renders.** Voice/accent is
  subjective and unreadable from a waveform — render a few cheap candidate TTS
  takes of one line, have the user pick, THEN bake the winner into the costly,
  `transient_auth`-prone avatar/lip-sync shots. Re-rendering all of them on a
  rejected voice is the avoidable burn. Same logic for the protagonist's look:
  approve the hero still before generating every shot from it.
- **Verify the mix by measurement, not faith.** You can't hear the render —
  `ffmpeg volumedetect` each section: a "dead air" ending or a buried line shows
  up as a window near -91 dB. Confirm every beat is present at level before
  declaring it done.

## Production ops

- Append EVERY submit — shots, keyframes, audio stems, resubmits — to
  `job_ids.txt` (`<job_name> <prompt_id>`) **at submit time**. Append-only:
  a resubmit gets a new line (`shot_s3_resubmit <id>`), never an overwrite,
  so dead prompt_ids keep their shot identity.
- Keep `LOG.md` at the project root: an append-only event log, written **at
  event time, not session end** — one line per submit/resubmit (+ failure
  code), per QC verdict (+ re-roll reason), and per decision that invalidates
  earlier work: recasts, prompt changes, which reference image or stem is now
  canonical. Frame-by-frame findings live in `qc/QC_LOG.md`; LOG.md indexes them.

  ```
  17:36 ERROR shot_s3 transient_auth -> RESUBMIT 27dd45b4 (same workflow, no re-login)
  17:44 QC shot_s2 FAIL background-crawl -> re-roll (same kf, new seed); recut beat if 2nd fails
  17:50 DECISION recast protagonist -> ref_ray.png canonical; vo2, vo6 stale, re-gen
  ```
- **Recovery contract:** a successor agent with zero memory must resume from
  disk alone: `LOG.md` (what happened, why) → `job_ids.txt` (which jobs) →
  `comfy jobs status` (their states). A decision not in LOG.md dies with the
  session — completed-but-stale jobs (an old casting's VO stems) otherwise
  look identical to good ones.
- ffprobe every downloaded clip; normalize off-spec duration/dimensions
  before conform.
- QC one frame from every clip against the screenplay before cutting.
- Exact runtime: trim/retime per-beat in ffmpeg, don't hope.

## Common mistakes (each killed a real cut)

| Mistake | Result |
|---|---|
| Renders before screenplay | "Eight handsome beats that don't cause each other" |
| Community lore on screen | "Too direct, fan-service" |
| Client's mentioned technique as the concept | "I said it to spark ideas" |
| No continuity plan | Protagonist changes face every cut |
| One 40s music gen | Arc inverts at the climax |
| B-roll over a continuous VO | "Feels like a montage / audio doesn't match" |
| Baked the voice into N shots before the user heard it | Re-render everything when they reject it |
| Butt-joined talking-head clips | Voice drifts out of sync (avatar pads video past audio) |
