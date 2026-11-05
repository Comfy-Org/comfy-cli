# ADR-UPLOAD-0010: Upload Only the Named Model and Prove Comfy Cloud Stored Its Bytes

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
  (Comfy Cloud's hash) and its sha256 (what `shasum -a 256` prints on any
  machine). A dry run reports both and uploads nothing.
- An approval is pinned to bytes and a workspace. `--expected-sha256` uploads
  nothing unless the file still has that sha256. `--workspace` uploads nothing
  unless `GET /api/workspaces/current` says this sign-in uploads into that
  workspace; no answer counts as a mismatch.
- `comfy model upload` asks before it starts. A machine caller must pass
  `--yes`, through the shared `confirm` helper, or it is refused with
  `model_upload_needs_confirm`.
- Before sending, the CLI lists the folder's models with that exact name. The
  same blake3 there means nothing is sent, and the output says
  `already_present`. Any other hash, or no hash, refuses with `name_in_use`. A
  listing that may be cut short refuses with `name_unverified`.
- The CLI hashes the bytes as it streams them. If they differ from the hash it
  read first, it never calls complete, so Comfy Cloud never files them.
- The upload succeeds only when the hash Comfy Cloud reports for the stored
  asset equals the local blake3. The task result is read first, then the asset.
  A different hash, or none, ends as `asset_upload_unverified`.
- The output's `loader_path` is the value a loader node takes to load the file
  on Comfy Cloud. The CLI uses the server's `loader_path` when it sends one, and
  the file name otherwise.

## Alternatives considered

- Upload with a multipart `POST` through the API server, as #1008 does.
  Rejected because the presigned upload sends bytes straight to storage and
  already has retries, progress and token refresh.
- Copy the file to a temporary snapshot before uploading, as #1008 does.
  Rejected because a model can be several GB. Hashing while streaming catches
  the same change with no second copy.
- Take `--name` to upload under another name. Left out because a workflow
  names the file it loads, and a different name would not match it.
- Trust the upload once storage accepts it. Rejected because only Comfy
  Cloud's hash of the stored asset shows the right bytes landed.

## What we reject

- Any upload of a file the caller did not name on the command line.
- A second file under a folder and name already in use in the library.
- Reporting success before Comfy Cloud's stored hash equals the local hash.

## Consequences

### Positive

- An agent can ask a person about one file by name, show its sha256, and
  upload exactly those bytes into the workspace they approved.
- Running the upload again for a file Comfy Cloud already has sends nothing.

### Negative

- Every upload reads the file twice: once to hash it and once to send it.
- A Comfy Cloud server without `/api/workspaces/current` refuses every
  `--workspace` upload.

### Known limits

- The name check sees the models the listing returns for this sign-in. A model
  in the workspace that the listing does not show is not checked.

## Validation

`tests/comfy_cli/command/test_assets_library.py` covers each refusal above, the
skip for bytes already present, a file changed mid-stream with no complete
call, a stored hash that differs or is missing, and the `--yes` refusal for an
agent caller.

## Revisit triggers

- Comfy Cloud loads models by hash instead of by folder and name.
- Comfy Cloud starts returning `loader_path` values that differ from the file
  name.
