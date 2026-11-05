# ADR-SIDEBAR-0009: Save Into a New Sidebar Folder and Delete Only the Record

Date: 2026-11-05

## Status

Accepted

Relation: Builds on ADR-AUTOMATION-0003. Save and delete keep the `envelope/1`
output and add only fields and registered warning codes.

## Context

A person saves a workflow from a terminal with `comfy workflow save flux.json
--name flux --where cloud`. Before this decision the CLI posted a workflow
record to `POST /api/workflows`. The Comfy Cloud Workflows sidebar lists
`/userdata` files, not records, so the saved workflow never showed there.

Writing a userdata file puts it in the sidebar. The CLI then depends on three
things Cloud does but does not publish as a contract:

- Cloud copies `workflows/<name>.json` into a workflow record named `<name>`.
- A file write whose `<name>` matches a live record adds a version to that
  record, even a record the person made some other way.
- Cloud deletes a copied record together with its file only when that record is
  the one live record of its name and its latest version equals the file.

Cloud's userdata `DELETE` takes no condition, so the CLI cannot delete a file
only if it still holds what the CLI last read.

## Decision

- The server flag `cli_cloud_save_enabled` from `GET /api/features` decides the
  path. Only a JSON `true` turns on the folder save. Anything else, including a
  failed read, keeps the record save. Turning the flag off moves every
  installed CLI back to the record save without a release.
- With the flag on, an editor-format workflow (one with `nodes[]`) is written as
  `workflows/<folder>/<name>.json` with `POST /api/userdata/...?overwrite=false`.
  No existing file is ever replaced. An API-format workflow keeps the record
  save, since the sidebar opens a file as an editor graph.
- A folder is used only when it holds no file at any depth and no live record
  is named `<folder>/<name>`. Otherwise the CLI moves on to `<name> (2)`, up to
  20 folders. A record search with no clear answer counts the folder as taken.
- `comfy workflow delete` deletes only the record, with the flag on or off.
  The CLI never deletes a sidebar file. With the flag on it adds the
  `sidebar_file_kept` warning only under Cloud's own rule above: this record is
  the one live record of the file's name and the file's JSON equals the
  record's latest version. Any other case, or any failed read, gives the plain
  delete.

## Alternatives considered

- Keep the record-only save. Rejected because the saved workflow never shows in
  the sidebar and the CLI has no link to give.
- Overwrite a file of the same name. Rejected because it replaces an edit the
  person made in the browser.
- Delete the sidebar file on `workflow delete`. Rejected because an
  unconditional file delete can remove an edit made in the browser after the
  CLI read the file.

## What we reject

- Any CLI call that deletes or overwrites a userdata file under `workflows/`.
- A file write whose `<folder>/<name>` matches a live record.
- Naming a file in `sidebar_file_kept` on a name match alone.

## Consequences

### Positive

- A workflow saved from a terminal shows in the sidebar, and the CLI prints a
  link that opens its folder.
- Neither save nor delete can change or remove an existing workflow.

### Negative

- After `workflow delete`, the sidebar file stays until the person removes it
  in Comfy Cloud.
- The folder save rests on Cloud behavior that only this record and code
  comments describe.

## Validation

`tests/comfy_cli/command/test_workflow_saved.py` pins each rule:
`TestSaveFlagOffKeepsTheRecordSave` the flag-off and failed-read paths,
`TestSaveToSidebarFolder` the `overwrite=false` write and the free-folder rule,
and `TestDeleteKeepsTheSidebarFile` the record-only delete and the warning rule.
The fake Cloud in `TestDeleteKeepsTheSidebarFile` fails any userdata `DELETE`.

## Revisit triggers

- Cloud publishes how a file becomes a record, or changes it. The flag goes off
  until the CLI matches.
- Cloud's userdata `DELETE` gains a condition on the file's content or version.
  Then `workflow delete` can remove the file safely.
- Cloud adds a way to write a file without adding a version to an existing
  record. Then the record search before each write can go.
