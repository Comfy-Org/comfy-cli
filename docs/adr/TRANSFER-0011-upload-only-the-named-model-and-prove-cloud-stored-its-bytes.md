# ADR-TRANSFER-0011: Upload Only the Named Model and Prove Comfy Cloud Stored Its Bytes

Date: 2026-11-05

## Status

Proposed

Relation: Builds on ADR-AUTOMATION-0003. `comfy model upload` and
`comfy assets library upload` keep the `envelope/1` output and add only fields,
a schema and registered error codes.

## Context

A person has a ComfyUI workflow that runs on their machine and wants it to run
on Comfy Cloud. The workflow loads `my_style.safetensors` from `loras`, and
Comfy Cloud has no file by that name. An agent helping them finds the gap and
asks: "Upload my_style.safetensors (151 MB) to loras on Comfy Cloud?" The person
says yes. The agent then runs `comfy model upload` for that one file.

Several things can go wrong between that yes and a model Comfy Cloud loads:

- The agent uploads a file the person did not name, such as a whole folder or a
  file found by a search.
- The file changes between the question and the upload, so Comfy Cloud gets
  bytes the person never approved.
- The sign-in moves to another workspace, so the file lands in a library the
  person did not pick.
- The bytes are cut or altered on the way, and Comfy Cloud stores something
  other than the local file.
- The folder already holds a different file of the same name. A Comfy Cloud
  run loads a model by folder and name, so a second file would change what
  that name loads for every workflow in the workspace.

Draft PR #1008 added an upload command with an approval step. This decision
keeps its approval idea and builds on the presigned upload already in
`comfy assets library upload`.

## Decision

- The command uploads exactly one file, the path given as its argument. It never
  expands a folder, a glob or a search into more files.
- The CLI reads the file once before anything is sent and computes its blake3
  (Comfy Cloud's hash), its sha256 (what `shasum -a 256` prints on any
  machine) and the blake3 of each 1 MiB block. A dry run reports the two file
  hashes and uploads nothing.
- An approval is pinned to bytes and a workspace. `--expected-sha256` uploads
  nothing unless the file still has that sha256. `--workspace` uploads nothing
  unless `GET /api/workspaces/current` says this sign-in uploads into that
  workspace; no answer counts as a mismatch.
- `comfy model upload` asks before it starts. A machine caller must pass
  `--yes`, through the shared `confirm` helper, or it is refused with
  `model_upload_needs_confirm`.
- Before sending, the CLI lists the models with that exact name in that folder
  of the workspace's model library (`workspace_library=true`): every member's
  models, and no public ones. A run on Comfy Cloud loads from that same set.
  The same blake3 there means nothing is sent, and the output says
  `already_present`. For `comfy model upload`, any other hash, or no hash,
  refuses with `name_in_use`. `comfy assets library upload` uploads the file
  anyway, as before this decision, so a script that replaces a model by
  uploading a new version under its name keeps working. A listing that may be
  cut short refuses with `name_unverified`.
- The upload sends only bytes that were hashed. It reads the file again one
  block at a time and checks each block against its blake3 from the first read
  before any of that block is sent. A changed block stops the upload, so none
  of the changed bytes reach storage, and complete is never called, so Comfy
  Cloud never files the earlier blocks. Reading stops at the hashed size.
- The upload succeeds only when the hash Comfy Cloud reports for the stored
  asset equals the local blake3. The task result is read first, then the asset.
  A different hash, or none, ends as `asset_upload_unverified`.
- The output's `loader_path` is the value a loader node takes to load the file
  on Comfy Cloud. The CLI reports only Comfy Cloud's own value: from the
  matching row when the bytes are already present, and from the asset's record,
  read after the hash check, when they were uploaded. It is null when Comfy
  Cloud gives none, when that read fails, and for a dry run that would upload.
  Comfy Cloud gives null when the name does not form a valid model path, and
  its own model list then leaves the model out, so the CLI does not put the
  file name in its place. A failed read does not fail an upload whose hash
  already matched.

## Alternatives considered

- Upload with a multipart `POST` through the API server, as #1008 does.
  Rejected because the presigned upload sends bytes straight to storage and
  already has retries, progress and token refresh.
- Copy the file to a temporary snapshot before uploading, as #1008 does.
  Rejected because a model can be several GB. The block check sends only
  hashed bytes with no second copy.
- Hash the bytes as they stream and compare once the PUT ends. Rejected
  because the changed bytes are already in storage by then.
- Take `--name` to upload under another name. Left out because a workflow
  names the file it loads, and a different name would not match it.
- Trust the upload once storage accepts it. Rejected because only Comfy
  Cloud's hash of the stored asset shows the right bytes landed.

## What we reject

- Any upload of a file the caller did not name on the command line.
- A second file under a folder and name already in use in the library, from
  `comfy model upload`.
- Reporting success before Comfy Cloud's stored hash equals the local hash.

## Consequences

### Positive

- An agent can ask a person about one file by name, show its sha256, and
  upload exactly those bytes into the workspace they approved.
- Running the upload again for a file Comfy Cloud already has sends nothing.

### Negative

- Every upload reads the file twice: once to hash it and once to send it. The
  second read hashes each block again.
- A Comfy Cloud server without `/api/workspaces/current` refuses every
  `--workspace` upload.

### Known limits

- Two uploads of different bytes under one name can both pass the name check
  if each lists the folder before the other completes. Only a Comfy Cloud write
  that refuses a name already in use can stop that.

## Validation

`tests/comfy_cli/command/test_assets_library.py` covers each refusal above, the
lookup's workspace-library query, the skip for bytes already present, a file
changed after it was hashed sending none of the changed bytes and never
completing, a stored hash that differs or is missing, the `--yes` refusal for an
agent caller, and a `loader_path` that is null when Comfy Cloud gives none or
its record cannot be read after the upload.

## Revisit triggers

- Comfy Cloud loads models by hash instead of by folder and name.
- Comfy Cloud stops sending `loader_path` on asset records, so every upload
  reports null.
