---
name: comfy-relay
description: How to present and interact during comfy creative work (images, video, audio) — show visual previews in chat, not text about files. Creative media work is iterative and visual; it is NOT like code. Use whenever generating, reviewing, or iterating on media with the comfy CLI.
---

Comfy work is **visual and iterative**. Code work communicates with diffs, logs,
and passing tests. Creative work is different: the user steers by **seeing the
work**. Your whole job in chat is to make them see it and react. **The image is
the message — text about the image is not.**

This skill is the presentation/interaction layer. It is paired with `comfy`
(the surface) and `comfy-fragments` (the compile model).

---

## Rule 1 — Show, don't tell

The moment a generation lands, **show it**: `Read` the image file so it renders
inline in the conversation. Never say "I generated an image at `outputs/x.png`"
— a path is not something the user can react to. Show:

- the base / source frame,
- **every** variant or take (not just the one you like),
- the final result.

Each preview is a decision point. The user redirects the work by looking at it.
A generation the user never sees might as well not exist.

## Rule 2 — Video: you can't inline it, so show frames

You can't play a clip in chat. Surface it visually instead — **one command does
it:**

```bash
comfy --json preview clip.mp4          # → clip.preview.png (a contact sheet) + duration/fps/has_audio
# then Read clip.preview.png
```

`comfy preview` handles all three modalities: **video → contact sheet** (a grid
across the whole timeline, best for judging pacing/arc), **image → thumbnail**,
**audio → waveform** (so you can *see* the dynamics you can't hear). It also
reports the facts frames can't show — **duration, fps, and whether there's
audio** — in the envelope.

Prefer it over hand-rolling ffmpeg. (If you need a custom grid: `--grid 6x4`;
custom width: `--width 720`.) For a key-moments read instead of a grid, extract
the open / turn / close frames and `Read` them in sequence.

## Rule 3 — Decisions are visual, not verbal

For a creative fork (which style, which take, which ending), **show the
candidates** — frames side by side — and recommend one. Don't describe options
in prose and ask the user to imagine them. Use `AskUserQuestion` with previews
for a clean pick, or just `Read` the contenders and say which you'd ship and
why. "Here are the three, I'd run #2" beats three paragraphs.

## Rule 4 — Cadence for long renders

Cloud video is minutes per clip. Don't go silent, and don't block on trivial
confirms. The rhythm is: **submit → one line on what's cooking (the prompt) →
run it in the background → show the result when it lands.** For a multi-shot
piece, surface pieces *as they arrive* so the user can course-correct early
instead of discovering a problem only at final assembly.

## Rule 4.5 — Async-first: make the wait work for you (and use subagents)

The CLI is **async-first** by design: `comfy run` returns a `prompt_id` in
milliseconds and the job runs server-side; you watch/collect separately. Don't
fight it — exploit it. Image gen is seconds, video is **2–5 minutes** — that's a
lot of wall-clock you should never spend blocked.

1. **Never block on a render.** Submit, then do real work while it cooks — prep
   the next stage, write the assembly/ffmpeg script, generate the music. Collect
   when you need the result (`jobs watch`, the state file, or background it).
   Don't poll in a tight loop.

2. **Fan out in parallel.** A 4-shot piece is 4 submits, then watch all four. The
   cloud parallelizes; your wall-clock is the *slowest single shot*, not the sum.
   Same for seed/variant sweeps (`workflow vary`).

3. **Dispatch a subagent per long, self-contained job.** When you have subagents
   available, a whole shot/clip/pipeline — survey → compose → run → wait →
   assemble — is minutes of work and many steps. Hand each to a **background
   subagent**: it drives the CLI end-to-end and reports back the artifact path +
   a short build log, while the main session stays responsive and other
   subagents run in parallel. The natural unit is **one subagent per independent
   piece** — a shot, the music, the title card — then the orchestrator assembles
   the returned clips.
   - Brief it like a creative director: the concept/shot, the technique, what to
     **reuse** (existing fragments/clips), and to report the path + any friction.
   - It returns the *conclusion* (the clip + log), not its 100-step transcript —
     so the orchestrator's context stays clean and it can run many in parallel.
   - Keep one piece = one subagent so a re-roll re-runs only that piece.

4. **Let the harness track the work.** When you submit and move on, completion
   re-invokes you — you don't babysit. Reserve a scheduled check only for
   external state the harness can't see (a remote queue), not for jobs it tracks.

This is the whole reason the surface is async: the agent that treats renders as
*fire-and-collect* (and fans the work across subagents) finishes a multi-shot
piece in the time of its slowest shot — not the sum of all of them.

## Rule 5 — Taste-forward, fast loops

Bring a strong creative POV and **recommend** — don't survey. The loop is
**show a version → user reacts → adjust**: short, visual, cheap. A rejected take
is useful information you got in one round; a ten-question upfront brief is
friction the user wades through before seeing anything. Lead with work, not
questions.

## Rule 6 — Lead with the visual, then show the source

After the preview, show the **source that made it** — the blueprint YAML, the
prompt, the params — so the user can tweak the *inputs*, not hunt through
compiled JSON. The compiled workflow is a build artifact; keep it out of chat.

| Moment | Lead with | Then show |
|---|---|---|
| Generated an image | the image (`Read` it) | the prompt / blueprint that made it |
| Rendered a clip | a contact sheet + duration/audio | the blueprint |
| A creative fork | the candidate frames | your recommendation |
| Iterating one value | the new result | `param: old → new` (one line) |
| Composing a workflow | the blueprint YAML (10–30 lines) | the `compose` summary — never the 100-node JSON |
| Editing one slot | the re-rendered result | `addr: old → new` (one line) |

## Rule 7 — Be honest about what you can't perceive

You can **see** images and video frames (you `Read` them). You **cannot hear
audio.** For music, SFX, or a mix, say so plainly — *"give it a listen"* — and
defer to the user's ear. Never claim a track "sounds great"; you don't know.
Report only what you can actually verify (loudness levels, duration, where the
beats/cuts land, no dead-air) and let the user judge the rest. Honesty here
keeps their trust on every other claim.

## Rule 8 — Never let the work vanish into /tmp

Outputs live in the **project dir** (see the layout in `comfy`), previewable and
re-openable. Surface the path, keep backups when you iterate destructively
(`x__v2_backup.mp4`), and `open` the final on the user's machine when it's done.
The artifact is the deliverable — keep it in reach, not buried in scratch.

---

**The one-line version:** in code you ship a diff; in comfy you ship a picture.
Put the picture in front of the user at every step, recommend with taste, iterate
in fast visual loops, and be honest about the audio you can't hear.
