---
name: comfy-run-on-cloud
description: "Use when someone wants a workflow from their local ComfyUI running on Comfy Cloud, clicked Run on Cloud, or pasted a Run on Cloud prompt. Checks the workflow against Comfy Cloud, uses an existing build or a node swap where Cloud lacks something, runs one real job, saves the workflow to their Cloud workspace and hands back the link that opens it."
---

# comfy-run-on-cloud

One workflow in, one link out. The person wants their workflow running on Comfy
Cloud; how it gets there is your job. Work on copies: never change the person's
original workflow.

## What the prompt allows, and what needs a yes

The Run on Cloud prompt is a yes to:

- reading the workflow and the local server, and checking it against Comfy Cloud;
- running it on Comfy Cloud (each run costs credits);
- lossless changes made on a copy (the table in step 4).

Each of these needs its own yes in chat, asked before you do it:

- **uploading any file from this machine**: a model, a LoRA, an input image,
  video or audio, or a custom node folder. Ask by file name, one line per file:
  path, size, what it is for, where it goes. A yes covers the files you named; a
  new or changed file needs a new yes. A custom build from the install (step 4)
  is asked differently: its plan names each local pack as a whole folder and
  gives the models by count and total size, and the yes to that plan is the
  yes to those uploads;
- a **lossy** change (step 4);
- running **partner-API nodes**, which spend credits on top of the run (step 6);
- a **new build or deployment** (it costs money to build and to keep running);
- stopping, deleting or replacing any existing build, release or deployment.

Put the questions you have now into one short message. If the person says no to
a file the workflow needs, do not upload it and do not swap in another file:
report "prepared, not run" and say what is missing.

## 1. Get comfy-cli ready

- `comfy workflow check --help` and `comfy model upload --help` must both work.
  If one fails, upgrade once with the tool that installed comfy-cli
  (`uv tool upgrade comfy-cli`). Still failing: tell the person the version
  `comfy --version` prints, and stop.
- `comfy --json cloud status`. Not signed in: `comfy cloud login`. Note
  `cloud_workspace.id`; every upload and save goes to this workspace.

## 2. Read the workflow from the local server

The prompt gives `Workflow: workflows/<name>.json` and the server URL, plus a
`ComfyUI user:` line when the server runs with `--multi-user`. Fetch the file
from the server; you do not need its folder on disk. In the URL, percent-encode
the path after `workflows/`: each `/` as `%2F`, a space as `%20`, and every
other character outside letters, digits, `-`, `_` and `.` the same way.

Do all work in one folder, written `<work>` below: `run-on-cloud/` plus the
workflow's name with every character other than a letter, digit, `-` or `_`
turned into `-` (`My flow (2)` gives `run-on-cloud/My-flow--2-`). Quote every
path in the commands. Put each name that comes from the workflow, the server or
the disk (a file, subfolder, folder or workflow name) in single quotes, writing
a `'` inside it as `'\''`: inside double quotes the shell still runs `$(...)`
and backticks.

First make sure the address is ComfyUI: `curl -sf "<server>/api/system_stats"`
must answer JSON with `system.comfyui_version`. Anything else (no answer, a web
page, plain text) means another program holds that port: stop, and tell the
person the address does not answer as ComfyUI. Never run the check on what such
an address returns.

```shell
mkdir -p "<work>/inputs"
curl -sf "<server>/api/userdata/workflows%2F<encoded name>.json" \
  -H "comfy-user: <user>" -o "<work>/original.json"
```

Leave out the `comfy-user` header when there is no user line. If the prompt
names no workflow, ask which one.

Every later comfy command that reads the local server (a `--where local` run)
must reach this same server, not comfy-cli's default of `127.0.0.1:8188`.
Prefix each one with the prompt's address: `COMFY_LOCAL_URL=<server> comfy ...`.

## 3. Check it against Comfy Cloud

Find the install's model folders first, so the check can compare the person's
files with Cloud's copies. `comfy --json which` names the install: for Comfy
Desktop, each `desktop.model_roots[].path`; otherwise `<workspace_path>/models`.
Pass each folder that exists as its own `--models-dir`:

```shell
comfy --json which
comfy --json workflow check "<work>/original.json" --where cloud \
  --models-dir '<models folder>' [--models-dir '<another>']
```

Without `--models-dir` the check cannot see a local file, so a model the person
has in a different version than Cloud's never shows up as `different_file`.
`which` names the install comfy-cli tracks, which may not be the one serving
`<server>`. For a model with no `local_paths`, ask the server whether it has the
file: `curl -sf "<server>/api/models/<folder>"` lists the file names it loads
from that folder. Listed there: the `--models-dir` folders are the wrong ones;
ask the person where that server keeps its models and check again with that
folder. Not listed: the file is not on this machine either, so there is nothing
to ask; go on with the verdict. When the item's `folder` is null or the lookup
fails, the server's answer proves nothing: ask the person where the model is.

`data.verdict` decides the next step:

| verdict | meaning | next |
|---|---|---|
| `covered` | Comfy Cloud has every node type and model | Run `original.json` on Comfy Cloud (step 6) |
| `covered_with_remaps` | Runs once its models point at Cloud's copies; `models[]` lists each remap | Write the remapped copy (below), then treat it as `covered` or `covered_needs_check` |
| `covered_needs_check` | Nothing is missing, but a model is not proven the same file as Cloud's copy | Run on Cloud's copy without asking (step 6); name those models in the report |
| `needs_build` | A node type or model is missing, or the person's model differs from Cloud's; `blocking[]` lists each | Step 4 |

A model that is not proven the same is any `models[]` row whose `identity` is
`unverified`. Cloud's library models carry no content hash, so a workflow on
stock models lands here at best; that is expected, not a reason to ask. A model
proven different (`source` `yours`, `identity` `different`) is never run on
Cloud's copy: it is already in `blocking[]` as `different_file`.

A remap is lossless. Write it without asking, with the same `--models-dir`
folders:

```shell
comfy --json workflow check "<work>/original.json" --where cloud \
  --models-dir '<models folder>' --apply-remaps --out "<work>/cloud.json"
```

Run `cloud.json` from here on. If any of its `models[]` rows is still
`unverified`, it is the `covered_needs_check` case: run it, and name those
models in the report. A `remap_not_applied` warning means `cloud.json` still
names the local file on that node: do not run it; report that node and model
as what blocks the run.

## 4. Something is missing: pick the route in this order

When `blocking[]` lists only models, keep Comfy Cloud: upload each one (step 5),
then check the copy again. A model item's `reason` says why: `not_on_cloud`
(Cloud has no file of that name), `ambiguous` (Cloud has several; `models[]`
lists the `candidates`, and picking one is a lossy swap unless the person's file
is proven the same), or `different_file` (Cloud's file of that name differs from
the person's, so theirs is uploaded). When it lists a node type, go down this list and stop
at the first that works:

1. **An existing build in the workspace.** `comfy --json deploy ls --workspace`,
   then for each running deployment `comfy --json deploy show --deployment <id>`
   for the release it serves. Read that release with
   `comfy --json build release manifest <release id>`: its `customNodes` and
   `models` are what the deployment runs. Judge packs and models from this one
   manifest, never from `build show`: a Build's definition can change after the
   release was cut, and the deployment keeps the old release. It fits when the
   served release has every pack and model the workflow needs. Keep the workflow
   unchanged and run it there (step 6). A build with no running deployment needs
   a yes to start one (`comfy-deploy`).
2. **A lossless node swap**, made on a copy without asking (below).
3. **A lossy swap**, asked first. A yes runs the copy on Comfy Cloud. A no goes
   to a custom build.
4. **A custom build.** Read `comfy skills show comfy-build` and `comfy skills
   show comfy-deploy`, and follow them. Build from the whole install that
   serves `<server>`, by comfy-build's Path A. `comfy --json which` does not
   prove which install that is. The server's `system_stats` (step 2) names it:
   `system.argv` starts with its `main.py`, and a `--base-directory <dir>` or
   `--base-directory=<dir>` in it is the folder that holds `models/` and
   `custom_nodes/`. That path is the server's, not always this machine's:
   ComfyUI in Docker reports `/app/ComfyUI/main.py` for an install that sits
   at `/host/ComfyUI` here. Take the `--base-directory` folder, else the one
   holding `main.py`, only when the server runs on this machine outside a
   container or VM, and this machine has `main.py` and `custom_nodes/` where
   `argv` puts them. In every other case (a container or VM, a path missing
   here, no full path in `argv`, or a relative `--base-directory` value), ask
   the person which folder on this machine holds that install.

   The Python that runs the server sits beside `main.py`, not in a
   `--base-directory` folder: `.venv` or `venv` in the folder holding
   `main.py`, as `bin/python` (`Scripts\python.exe` on Windows). Take it only
   when `'<python>' -c 'import sys; print(sys.version)'` prints the server's
   `system.python_version` exactly and, when `system_stats` has
   `system.pytorch_version`, `'<python>' -c 'import torch; print(torch.__version__)'`
   prints that too. A match only rules a wrong Python out; it does not prove
   this one starts the server, so the plan names it and the person confirms
   it. When neither is there or it differs, ask the person which Python starts
   the server.

   Before `init`, list the symlinks under `custom_nodes/` whose name ends in
   `.disabled`:

   ```shell
   find '<install>/custom_nodes' -maxdepth 1 -name '*.disabled' -type l
   ```

   `init` follows such a link to its folder, and unless that folder is a git
   checkout with a GitHub `origin` or a registry pack, it treats it as a local
   pack, refuses to package a symlinked one (`build_spec_invalid`, `node root
   must not be a symlink`), and stops before writing any spec. You cannot tell
   every case from outside, so ask about each link listed. Name it to the
   person: ComfyUI does not load such a link, but `init` cannot scan past it.
   Ask them to move the link itself (no trailing `/`) out of `custom_nodes/`:

   ```shell
   mv '<install>/custom_nodes/<name>.disabled' '<install>/<name>.disabled'
   ```

   and back after the build with
   `mv '<install>/<name>.disabled' '<install>/custom_nodes/<name>.disabled'`.
   Run `init` only after they move every one; if they decline, stop. Then,
   with every path quoted:

   ```shell
   comfy --json build init '<install>' --name '<name>' --comfy-url '<server>' --python '<python>'
   ```

   Comfy Desktop builds the same way, not from its snapshot, which carries no
   models: `<install>` is `desktop.data_dir`, with `--python
   '<desktop.python>'`.

   `init` writes the spec to `<install>/comfy-build.yaml`, in the person's
   install. If that file is already there, `init` refuses with
   `build_spec_exists`: it is an earlier build of this install, holding its
   Build id and the person's edits. Never pass `--force`, which overwrites it.
   Run `comfy --json build update '<install>' --dry-run` with the same
   `--comfy-url` and `--python` instead, show the person each entry of its
   `diff` by name (a rescan removes a model added by hand earlier) and the
   `pipDependencies` lines it adds and drops, and run it again with `--yes`
   only after they agree. `update` rewrites the definition and keeps the Build
   id and name, so the push replaces that Build's definition.

   The build carries the installed pack versions and the models `init` lists
   with their folder and hash; the push swaps a public model for its download.
   A git pack (`source` `git`) builds from its commit, without the changes
   `git -C '<install>/custom_nodes/<folder>' status --porcelain` lists. Then
   check the spec's `customNodes` before the plan:

   - Drop each entry whose folder name ends in `.disabled`. ComfyUI does not
     load such a folder, but `init` lists it.
   - Drop each entry whose folder declares no nodes. `init` lists every folder
     under `custom_nodes/` as a pack, and the builder fails the release on one
     that declares no nodes (`declared custom nodes failed to import`), though
     ComfyUI runs past it. A folder declares nodes when its `__init__.py`
     defines or imports `NODE_CLASS_MAPPINGS` (the V1 node API) or
     `comfy_entrypoint` (the V3 node API).
   - List the single `.py` files directly under `<install>/custom_nodes/`.
     ComfyUI loads each as a pack (it skips only other extensions and names
     ending in `.disabled`), but `init` scans folders only, so no such file
     reaches the spec. A node type comes from one when
     `<server>/api/object_info/<class>` gives its `python_module` as
     `custom_nodes.<file name without .py>`. When the workflow uses such a
     node type, or you cannot tell, stop: the build cannot carry it unless the
     person makes it a folder pack or names an installable source for it.

   Name each dropped folder in the plan, and any kept pack that imports it
   (`comfy-build`). A workflow model the spec's `models` does not list is not
   carried, wherever it sits (step 3's `local_paths` say where): `init` skips
   a file outside `<install>/models`, a dotfile, and an extension other than
   `.safetensors`, `.ckpt`, `.pt`, `.pth` and `.bin`, such as `.sft`. Hash
   that file with sha256 (`shasum -a 256 '<file>'`) and look it up with
   `comfy build refs resolve` (`comfy-build-authoring`). Add a candidate
   before the push only when its `sha256` equals that hash. A candidate whose
   `sha256` differs, or that has none, is another model: offering it is a
   lossy swap, asked on its own (below), and the build's yes does not cover
   it. Name a candidate added on that swap's yes in the plan as a stand-in
   for the person's file, and any model not added as not carried.

   Before building, show the plan and get a yes. The plan names:

   - the spec file `init` or `update` wrote, `<install>/comfy-build.yaml`;
   - each local pack (`source` `local`) as a whole folder: it uploads with
     everything in it, a key or config file too;
   - the models, by count and total size (the spec's `models`);
   - each dropped folder, each git pack with local changes, each stand-in
     model and each model not carried;
   - each single `.py` file under `custom_nodes/`, as not carried by the
     build;
   - the Python passed as `--python`, by path: the yes confirms it is the one
     that starts their server. When they name another, rescan with
     `comfy --json build update '<install>' --python '<that one>'` (and the
     same `--comfy-url`) and show the plan again;
   - the Build id the push replaces, when the spec has one, asked as a
     question of its own;
   - the GPU and region (cheapest that fits first, from
     `comfy --json deploy refs compute`), the costs, and `--min 0 --max 1`
     workers.

   This plan's yes is the yes to its uploads; nothing is uploaded before it.

| kind | what counts | what you do |
|---|---|---|
| Lossless | A remap from the check; a utility node swapped for the core node that computes the same thing (math, string concat, a LoRA stack as chained `LoraLoaderModelOnly` at the same strengths, a pipe node as direct links); removing a node no output depends on | Do it on a copy and list it in the report |
| Lossy | Anything else that changes what is generated: another model, LoRA, VAE, ControlNet, upscaler or embedding; dropping a LoRA or a node that touches pixels; another implementation of the same job | Ask first (below) |
| Not swappable | A node that does the core work and has no core equivalent on Cloud | Existing build or custom build |

**Making a swap.** Copy the workflow to `"<work>/cloud.json"` (or edit
`cloud.json` if the remap already wrote it) and run every edit on that copy with
`--where cloud`: `comfy --json workflow add-node`, then `workflow connect` for
each link, `workflow set-widget` for each value, and `workflow delete-nodes` for
the nodes it replaces. Name slots in `connect`, never their numbers
(`<id>.INT`, not `<id>.0`): the core node may list its outputs in another
order, so a number can wire a FLOAT where an INT went. `comfy --json nodes show
<class> --where cloud` lists a node's inputs and outputs by name.

**Asking about a lossy swap.** One message: each swap as `original -> replacement`
and how the output may change; then two choices. **Swap**: runs on Comfy Cloud,
nothing to build, output may differ. **Keep the original**: a custom build, with
its costs. Offer only replacements from the same model family, taken from
`comfy --json nodes show <Loader> --where cloud --all-choices`.

**Check every changed copy before running it.** Pack versions move widget values:

```shell
comfy --json run --workflow "<work>/cloud.json" --print-prompt --where cloud > "<work>/preview.json"
jq '.data.prompt' "<work>/preview.json" > "<work>/api.json"
comfy --json workflow validate --workflow "<work>/api.json" --where cloud
```

Fix a widget by name with `comfy --json workflow set-widget "<work>/cloud.json"
<node>.<widget> <value> --where cloud`, only to a value that keeps the saved
behavior.

## 5. Uploads, each asked by name

**A model** from a `blocking[]` item. The item names the file: `filename`,
`folder` (the ComfyUI models folder, or null when unknown), `url` (where the
workflow says it downloads from, or null) and `local_paths` (copies found under
`--models-dir`). No `local_paths`: say the file is not on this machine, give the
`url` if there is one, and do not upload anything for it. Several `local_paths`:
ask which one.

The dry run sends no file; it hashes the file here and reads the library:

```shell
comfy --json model list-folders --where cloud
comfy --json model upload '<file>' --type <folder> --workspace <workspace id> --where cloud --dry-run
```

`<folder>` is the item's `folder` when `model list-folders` lists it; otherwise
pick the folder the loader reads from and say so. If the dry run says
`already_present`, Comfy Cloud has these exact bytes under that name: nothing
needs uploading or asking; use its `loader_path`. Otherwise ask by `filename`,
with the dry run's path, `size`, `sha256` and destination folder. After the yes:

```shell
comfy --json model upload '<file>' --type <folder> --workspace <workspace id> --where cloud \
  --yes --expected-sha256 <approved sha256>
```

Set the loader on the copy to the returned `loader_path`, then run the check on
the copy again. A null `loader_path` (after an upload or on `already_present`)
means Comfy Cloud gave no loader value for that name: do not put the file name
in its place; report the model as what blocks the run. When it fails, nothing ran on Comfy Cloud yet:

| error | what to do |
|---|---|
| `model_upload_needs_confirm` | `--yes` was missing. Ask, if you have not, then run it with `--yes --expected-sha256` |
| `asset_upload_wrong_workspace` | This sign-in uploads into another workspace (`details.current_workspace`). Nothing was sent. Tell the person; upload only after they sign in to the approved one (`comfy cloud login`) or approve the current one |
| `asset_upload_invalid`, `file_changed` | The file changed since the approved dry run. Run the dry run again and ask again with the new `sha256` |
| `asset_upload_invalid`, `name_in_use` or `name_unverified` | The folder already has (or may have) another file of this name, and uploading would change what the name loads for the whole workspace. Copy the file under a new name into `<work>/`, then dry run and ask again with that name |
| `asset_upload_invalid`, `bad_expected_sha256` | Pass the 64-hex `sha256` from the dry run |
| `asset_upload_invalid`, other reasons | The message names the file or `--type` problem; fix it, or report it as what blocks the run |
| `asset_upload_unverified` | The file is stored, but Cloud's hash of it does not match or could not be read. Do not run a workflow that loads it; report it, with `details.asset_id` |

**An input file** (image, video, audio) a loader names: fetch it from the local
server before you ask, then ask with the fetched file's size and its
`shasum -a 256` hash:

```shell
curl -sfG "<server>/api/view" --data-urlencode 'filename=<file>' \
  --data-urlencode 'subfolder=<subfolder>' --data-urlencode "type=input" \
  -H "comfy-user: <user>" --create-dirs -o '<work>/inputs/<subfolder>/<file>'
shasum -a 256 '<work>/inputs/<subfolder>/<file>'
```

Keep the subfolder in the path, so two inputs of the same name in different
subfolders stay two files. A subfolder or file name with `..` or a leading `/`
would write outside `<work>`: do not fetch it; report that input as what blocks
the run.

After the yes, upload that same file. Do not fetch it again: if the server's
file has to be read again, ask again with the new hash.

```shell
comfy --json upload '<work>/inputs/<subfolder>/<file>' --where cloud
```

Set the loader on the copy to the returned `uploads[].cloud_name`.

## 6. Run one real job

Never call it working before a real job succeeds.

- **Comfy Cloud:**
  `comfy --json run --workflow <file> --where cloud --wait`, then
  `comfy --json download <prompt_id> --where cloud --out-dir "<work>/outputs"`.
  `spend_consent_required` means nothing was submitted: the workflow has
  partner-API nodes (`details.partner_nodes`) that spend credits beyond the run.
  Ask, naming those nodes; after a yes, run the same command with
  `--allow-spend`. On a no, report "prepared, not run".
- **A deployment** (existing or custom build): first read
  `comfy --json deploy logs --deployment <id>`; `data.comfyuiLog` must show no
  `IMPORT FAILED` for a pack the workflow uses. Make the API file against the
  local server, which has the packs:
  `COMFY_LOCAL_URL=<server> comfy --json run --workflow <file> --print-prompt --where local`,
  `.data.prompt` into `"<work>/api.json"`.

  `deploy run` sends local files on its own. Any string input in `api.json`
  with a file extension can match a file, whatever the node type, so a custom
  node's text field can too. When two folders hold a file of the same name, it
  picks by its own order, which may not be the one you expect. So do not build
  the list yourself: get it from `deploy run`. Run it first with `--no-upload`,
  which reuses files the deployment already has and sends none:

  ```shell
  comfy --json deploy run --deployment <id> --workflow "<work>/api.json" --no-upload \
    --wait --output-dir "<work>/outputs"
  ```

  If the deployment has every file, it runs and nothing was sent. Otherwise it
  fails with `deploy_asset_missing` before it submits a job.
  `details.missing[]` is the exact upload plan: one entry per file, with
  `local_path` (the full path it picked), `bytes` and `hash`. Ask about every
  entry in one message, by name, with its `local_path` and size (step 5's
  rule). On a no, report "prepared, not run". After a yes that covers every
  entry, run the same command from the same folder, with `--no-upload` replaced
  by one `--allow-upload <hash>` per approved file:

  ```shell
  comfy --json deploy run --deployment <id> --workflow "<work>/api.json" \
    --allow-upload <hash> --allow-upload <hash> \
    --wait --output-dir "<work>/outputs"
  ```

  It uploads only files with those hashes. It hashes a private copy of each
  file it uploads and sends that copy, so a file that changes meanwhile is not
  sent.
  After the run, `data.assets.files[]` names each local file it read; report
  them.

  If either run fails with `deploy_asset_missing`, read its `details`:

  - `details.missing[]` is there: a file differs from the approved plan
    (another file was picked, or one changed). Nothing outside the plan was
    sent. Ask again about the new `details.missing[]`; never pass a hash
    nobody approved.
  - `details.server_code` is `missing_asset`: the workflow names an asset
    already on the deployment's side that this account cannot reach. No local
    file is involved, so there is nothing to approve or upload. Report that
    asset as unreachable, and the run as "prepared, not run".

Keep the job id. On a timeout, watch the same job; never submit a second paid job
to recover a download. Run comfy commands one at a time.

## 7. Save it, then hand back the link

Save the file that ran (the copy, if you changed one):

```shell
comfy --json workflow save <file> --name '<name>' --where cloud
comfy --json workflow save <file> --name '<name>' --where cloud --deployment <id>   # ran on a deployment
```

It saves into a new folder in the Cloud Workflows sidebar and never replaces a
file. Hand back `data.url` as returned. `data.url_opens` says what it opens in
the person's workspace:

- `workflow`: only the saved workflow,
  `?workspace=<id>&workflowId=<data.workflow_id>&deployment=<id or standard>`.
  Cloud checks the link against the saved file when it opens.
- `folder`: every workflow in the new folder,
  `?workspace=<id>&workflows=<folder>&deployment=<id or standard>`.
  `data.folder_link_reason` says why: `workflow_links_off` (the workspace does
  not give workflow links), or `record_not_found`, `record_ambiguous` or
  `record_lookup_failed` (the save could not tell which workflow Cloud made
  from the file, and `data.warnings[]` carries `workflow_record_unknown`).

Treat a `data.url_opens` or `data.folder_link_reason` value not listed here as
`folder`: later versions may add values.

If `data.url` is missing and `data.filename` is set, the file is in the
sidebar but the CLI could not build a link from the Cloud base URL; give
`data.filename`. If both are missing, the save is a record only: nothing was
added to the Workflows sidebar and there is no link. Say that plainly, give
`data.workflow_id`, and do not point them at a sidebar entry. Then read `data.warnings[]` for the reason:

- `api_format_saved_as_record`: sidebar saving is on, but the file is in API
  format, which the sidebar cannot open. Save the editor-format file instead
  (the copy, not `api.json`) to get a folder and a link.
- no warning: the workspace saves workflows as records only. Sidebar saving is
  off, or its setting could not be read.

## Report

Lead with the state: ran, or "prepared, not run" and what blocks it. Then:

- the route: Comfy Cloud, an existing deployment, or a custom build;
- each change made to the copy, and each lossy swap they approved;
- the files uploaded;
- each model that ran on Cloud's copy unproven (`identity` `unverified`), by
  name, with one line: the output may differ if their file is another version;
- the job id and where the outputs are on disk;
- for a deployment: its id, GPU, region, worker bounds, cost, and
  `comfy deploy stop --deployment <id>` to stop the billing.

End with the `data.url` link, or, for a record-only save, the line saying there
is no link.
