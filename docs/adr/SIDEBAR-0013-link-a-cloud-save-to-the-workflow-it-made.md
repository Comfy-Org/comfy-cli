# ADR-SIDEBAR-0013: Link a Cloud Save to the Workflow It Made

Date: 2026-11-09

## Status

Proposed

Relation: Refines [ADR-SIDEBAR-0009](SIDEBAR-0009-save-into-a-new-sidebar-folder-and-delete-only-the-record.md).
It changes one consequence of that record: the printed link opens its folder.
With a second Cloud flag on, the link opens the one saved workflow instead.
The feature-gated save, the new folder, the `overwrite=false` write, the
free-folder rule and the record-only delete stay as ADR-SIDEBAR-0009 says.

## Context

A person runs `comfy workflow save flux.json --name flux --where cloud`. The
CLI writes `workflows/flux/flux.json` and prints a link with `workflows=flux`.
Cloud's page opens every workflow file directly in that folder. If the person
later adds a second file there, the link opens both. If they move the file out,
the link opens nothing.

Cloud copies each sidebar file into a workflow record named after its path:
`workflows/flux/flux.json` becomes the record `flux/flux` (ADR-SIDEBAR-0009
lists this). The copy is best-effort, so a save can end with no record. The CLI
already searches for that record after the write to report `workflow_id`.

Cloud's page can open a workflow by its record id. It reads the record, maps
the record's name back to its one sidebar path, checks that the file there
holds the record's newest version, and opens that file. Save in the browser
then writes back to the same file. When any check fails, the page shows
"Couldn't open this workflow" and opens nothing.

## Decision

- Read `private_saved_file_links_enabled` from the same `GET /api/features`
  reply as `cli_cloud_save_enabled`. Only a JSON `true` turns it on. It changes
  only the link of a sidebar save.
- With it on, the link uses the record that the search after the write finds:
  the live records named exactly `<folder>/<name>`, every page read, with the
  same page cap as the free-folder check. With exactly one such record, the
  link is `?workspace=<id>&workflowId=<record id>&deployment=<id or standard>`.
  The workspace and deployment follow the folder link's rules: both go in only
  when Cloud names the workspace.
- Otherwise the link stays the folder link. `data.url_opens` is `workflow` or
  `folder`, or null with no link. For a folder link, `data.folder_link_reason`
  is `workflow_links_off`, `record_not_found`, `record_ambiguous` or
  `record_lookup_failed`. The last three also carry the existing
  `workflow_record_unknown` warning.
- `workflow_id` keeps naming the record that `workflow get` and
  `workflow delete` take. Record-only, API-format and local saves do not
  change.

## Alternatives considered

- An id for the sidebar file itself, returned in Cloud's save reply. Rejected:
  it needed a new Cloud route that reads files by id and a new field in the
  save reply. Cloud already serves workflow records by id
  (`GET /api/workflows/{workflow_id}`).
- Open the record's own content instead of the sidebar file. Rejected: Save in
  the browser would not write back to the sidebar file, and a record that fell
  behind its file would open old content.
- Keep only the folder link. Rejected: it opens other files added to the
  folder, and it loses the file once the file moves.

## What we reject

- Picking one record when several live records share the name, or matching on
  part of a name.
- Printing a `workflowId` link when the record search failed or ran past its
  page cap.
- Changing the file write, the free-folder rule or delete to make the link
  work.

## Consequences

### Positive

- The link opens one workflow. A file added to the folder later does not open
  with it.
- Cloud turns workflow links on or off for every installed CLI without a
  release. Off restores the folder link.
- The save makes no new request. The record search it uses already ran to
  report `workflow_id`.

### Negative

- When the file and its record drift apart, the link opens nothing: a record
  that missed a rename, or a version added to the record and not the file. The
  person sees "Couldn't open this workflow" and opens the sidebar instead.
- The CLI cannot tell whether Cloud's page reads `workflowId`. The flag must
  go on only once the page does.

## Validation

`tests/comfy_cli/command/test_workflow_saved.py` covers it in
`TestSaveLinksTheWorkflow`: one record gives a `workflowId` link; a flag that
is not a JSON `true` keeps the folder link; zero records, two records, records
split across pages, a failed or unreadable search and a search past the page
cap each keep the folder link with their reason; deployment and an unknown
workspace follow the folder link's rules. `TestSaveFlagOffKeepsTheRecordSave`
keeps the record-only save when sidebar saving is off and workflow links are
on.

## Revisit triggers

- Cloud's page stops reading `workflowId`, or Cloud changes how a file becomes
  a record. The flag goes off until the CLI matches.
- Cloud's save reply starts naming the record it made. Then the search after
  the write can go.
