---
name: comfy-relay
description: Use when constructing, writing, or slot-editing a ComfyUI workflow JSON, or surfacing the result of `comfy run` — the artifact must appear in chat, not vanish into /tmp.
---

The artifact is the update. When you build or change a workflow, the user
should see the JSON — not a sentence about the JSON. When you submit it,
the user should see the prompt_id and routing — not a 200-line envelope.

This skill is about **what to put in chat** while driving the comfy CLI.
It is paired with `comfy` (the surface) and `comfy-edit` (transforms).

## The rule

**Show the artifact. Then act on it.**

| Moment | Show in chat |
|---|---|
| Constructing a workflow JSON | The full JSON in a fenced ```json block, before writing to disk |
| Editing one slot | A one-line diff: `<addr>: <old> → <new>` |
| Varying multiple slots | The sweep matrix as a small table |
| Submitting a workflow | `prompt_id`, `state_file`, route (local/cloud), one line |
| Job terminal | status + the output URL(s) or path(s) — nothing else |

**Anti-pattern:** writing the JSON to `/tmp/foo.json` and saying "I've
prepared the workflow." The user can't see `/tmp`. They lose all ability
to spot a bad prompt, wrong slot, or wrong model before the run.

## Constructing — show the JSON inline

Show the workflow JSON in a fenced ```json block in chat BEFORE writing to
disk. The user must be able to spot bad values (wrong model, wrong prompt)
from what you send. Then call `Write`.

## Editing one slot — show the diff

Discover slots, then announce the change as a single-line diff before
running `set-slot`:

```text
Slot edit: 1.prompt
  - "A 1950s Technicolor TV commercial …"
  + "A 1950s Technicolor TV commercial, color-saturated Kodachrome …"
```

Then:

```bash
comfy workflow set-slot ./workflows/vintage-intro.json 1.prompt="…"
```

If the new value is >200 chars, show the first 120 chars + `[…N more]`
in the diff. The full string is in the command, so they can still see it
if they need to.

## Varying many slots — show the matrix

For `workflow vary`, show the sweep as a small table (variant | slot values)
before generating files.

## Submitting — surface route + ids in one line

After `comfy run`, distill the envelope into a single line:

```text
Submitted (cloud) → 99626e0a-b7fb… · state: <state_file path from envelope>
```

Not this (don't dump the full envelope):
```json
{"ok": true, "command": "run", "version": "0.0.0", "where": "cloud", "data": { … 40 lines … }}
```

If `node_errors` is non-empty, show those — that's the only part of the
envelope the user usually cares about.

## Terminal status — output, not envelope

When `jobs watch` or `--wait` returns, surface just status + outputs:

```text
✓ completed in 2m45s — 1 output
  https://cloud.comfy.org/api/view?filename=dcb37…mp4
```

Or on failure:
```text
✗ error (node 1 / Veo3VideoGenerationNode)
  Unauthorized: Please login first to use this node.
  Hint: comfy cloud login   (or comfy cloud set-key --key …)
```

## Truncation rules — keep chat scannable

- JSON > ~60 lines: show first 40 and last 10, elide the middle with `// … N nodes elided …`
- Any string slot > 200 chars: show first 120 chars + `[…N more]`
- Never truncate `prompt_id`, `state_file`, error messages, or URLs
- File paths: show relative-to-cwd (`./workflows/x.json`) not the `/tmp/…` form

## What NOT to relay

- Don't narrate the building process ("first I will set the model, then…").
  The JSON itself shows the structure.
- Don't dump the raw `comfy --json …` envelope unless the user asked for it.
- Don't relay every progress tick during `jobs watch` — one line per state
  transition (`queued → executing → completed`) is plenty.
- Don't echo API keys or auth tokens, ever. The CLI redacts; you should too.

## Output presentation

After downloading outputs, present them to the user: use `view_media` for
images, provide the file path for videos. Don't just say "downloaded to
./outputs/" — show the artifact.

## Quick check before sending a message

Ask: would the user be able to spot a bad value (wrong model, wrong prompt,
wrong seed) from what I'm about to send? If the answer is "only by
opening a file," put the artifact in the message.
