# ADR-TRANSFER-0016: Share Model Download Claims Across Workspaces

Date: 2026-11-18

## Status

Proposed

## Context

`comfy model download` refuses a second download into a destination that
another download holds. The proof is an exclusive OS lock on a claim file
keyed by the destination's real path. That file lived in
`<workspace>/.comfy-downloads/claims/`, so two invocations that resolved
different workspaces used different claim files for one destination. Both
won and both wrote the file.

Before #1094 this needed a `--relative-path` outside the workspace. Since
#1094, a typed download saves into the folder the running ComfyUI scans. On
Comfy Desktop that is one shared model library for every install, so two
workspaces reach the same destination by default (comfy-cli#1095).

## Decision

Claims live in one owner-only directory per user and host:
`download-claims/` in comfy-cli's config directory, or the directory named
by `COMFY_DOWNLOAD_CLAIMS_DIR`. The submitter and its background worker use
one resolver for the claim path, and neither passes a workspace.

The claim payload also names the holder's state file (`state`). A reader in
another workspace follows that pointer to judge whether the holder is still
in its spawn gap, the time between a submit releasing the lock and its
worker taking it. A payload with no pointer is resolved in the reader's own
state directory, as before. A pointer that fails validation is judged on the
payload's own `created_at`: it refuses for the startup grace and is stale
after that.

Without a home directory the claims directory is unavailable, and downloads
degrade to the advisory in-workspace scan with a warning, as they already
did for an unusable claims directory. `prune` still sweeps the old
per-workspace claims directory so it drains.

## Alternatives considered

- A claim next to the destination. It is shared by construction, but it puts
  a dot-file in the user's models folder, which may not be writable.
- Share claims only for destinations that came from the server lookup. It
  needs a new `DownloadState` field and leaves `--relative-path` open.
- Read an unresolvable holder as held with no time bound. A crashed submitter
  would then block its destination until someone deletes the claim by hand.

## Consequences

Downloads from any workspace on one host refuse each other for every
destination. The tests redirect the directory through the environment
(`tests/conftest.py`), and a spawned worker inherits it. Claims on a network
filesystem are still best effort, and two hosts sharing a model library can
still both win.

## Revisit triggers

- Some platform cannot create or lock files in the config directory.
- comfy-cli gains a second process that writes claims without a state file.
