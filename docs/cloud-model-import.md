# Import a local model into the standard environment

`comfy model upload` imports a named local model through Cloud's authenticated
`POST /api/assets` multipart API. It does not start a job or create a custom
Build. The current workspace must match the explicitly approved workspace ID,
`private_models_enabled` must be true, and `--type` must be a folder currently
advertised by `comfy model list-folders --where cloud`.

Preview the exact file and target before asking the user:

```sh
comfy --json model upload /absolute/path/original.safetensors \
  --type loras --workspace <workspace-id> --dry-run --where cloud
```

The plan names the file, SHA256, size, destination and workspace. Null timing and
cost fields mean unknown. Add any known license/NSFW metadata to the approval
question; this command does not infer it. A plan is not upload approval. After
the user approves that exact file and target:

```sh
comfy --json model upload /absolute/path/original.safetensors \
  --type loras --workspace <workspace-id> \
  --expected-sha256 <approved-sha256> --yes --where cloud
```

The command copies the file into a private temporary snapshot, checks the
approved digest, and streams only that snapshot. It needs temporary disk space
for one copy; the snapshot is closed and removed on success or failure. Edits
made to the source during upload cannot change the approved bytes. A different
same-named Cloud asset blocks the import: approve a unique `--name` instead of
silently replacing it. An identical asset already owned by the authenticated
workspace is reused by hash, without re-uploading the file. An identical public
asset alone does not establish ownership; the command uploads the approved
snapshot instead.

Success verifies the returned BLAKE3 digest and model folder, and returns the
actual `loader_path`. Older Cloud responses express BLAKE3 as a storage filename;
these are normalized under the documented asset-hash contract. Legacy loader
names are derived only from the verified model asset's name and folder. Put the
returned path on a workflow copy and validate/run that copy before claiming it
works. Imported bytes alone do not establish a successful model load or job.

Network or malformed-response failures may happen after storage. Inspect the
asset library before retrying; the error's upload state is intentionally unknown
when it cannot be proved. Credentials are sent only to the authenticated Cloud
origin, and upload redirects are refused.
