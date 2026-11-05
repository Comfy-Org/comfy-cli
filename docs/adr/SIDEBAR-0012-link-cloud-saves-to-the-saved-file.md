# ADR-SIDEBAR-0012: A Cloud Save Links to the Saved File

Date: 2026-11-05

## Status

Proposed

Relation: Refines [ADR-SIDEBAR-0009](SIDEBAR-0009-save-into-a-new-sidebar-folder-and-delete-only-the-record.md).
Its feature-gated save, new-folder, no-overwrite and record-only delete rules
remain unchanged.

## Context

A person saves a workflow to Cloud and follows the printed folder link. The
editor opens every workflow directly in that folder. Moving the saved file out
of the folder loses it from the link. A saved-file UUID can identify that one
file across a rename or move, but an older Cloud page cannot read that link.

## Decision

- Read `private_saved_file_links_enabled` alongside `cli_cloud_save_enabled`
  from the save's single `GET /api/features` reply. Only a JSON `true` enables
  either setting. An unreadable reply keeps the record-only save.
- For a sidebar save, use the userdata POST reply's `id` only when it is a UUID
  string and saved-file links are enabled. Normalize it to canonical UUID text
  and return it as `data.saved_file_id`. The URL uses `savedFile=<id>` instead
  of `workflows=<folder>`, retaining the workspace and deployment values.
- Otherwise keep the folder URL and return `saved_file_id: null`. Record-only,
  API-format and local saves keep their output unchanged.
- `workflow_id` keeps naming the mirrored workflow record. The new receipt
  field never changes which id `workflow get` or `workflow delete` takes.

## Alternatives considered

- Always print an exact-file link: older Cloud replies or pages cannot support
  it. Both the feature setting and the reply UUID are required.
- Keep only the folder link: it cannot follow the file after a move and can
  open other files added to that folder.

## What we reject

- Guessing the file UUID from a record id, filename or a later list request.
- Changing overwrite or delete behavior to support a link.

## Consequences

Cloud controls whether installed CLIs use exact-file links without a CLI
release. Turning the link setting off restores folder links. The CLI cannot
prove that the browser will open the file; Cloud must enable the setting only
when its page supports `savedFile`. Missing workspace information still means
the browser's current workspace is used, with the existing warning.

## Validation

`tests/comfy_cli/command/test_workflow_saved.py` covers exact-file output,
folder fallback, strict boolean flags, malformed replies, one features read,
workspace/deployment values, pretty/JSON agreement and unchanged record saves.
Existing local-save and delete tests protect the unchanged paths.

## Revisit triggers

Revisit if Cloud changes the saved-file UUID or URL contract, or removes the
feature settings. A link that opens the wrong file requires disabling
saved-file links until the contract matches again.
