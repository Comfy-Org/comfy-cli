---
name: comfy-run-on-cloud
description: "Use when someone wants a workflow from their local ComfyUI running on Comfy Cloud, clicked Run on Cloud, or pasted a Run on Cloud prompt."
---

# comfy-run-on-cloud

One workflow in, one link out. The person wants their workflow running on Comfy
Cloud; how it gets there is your job. Work on copies: never change the person's
original workflow.

## What the prompt allows, and what needs a yes

The Run on Cloud prompt is a yes to reading the workflow and the local server,
checking it against Comfy Cloud, and writing model remaps to a copy (step 3).
Each of these needs its own yes in chat, before you do it:

- **changing the workflow's nodes** so it runs on Comfy Cloud, lossless or not
  (step 4). When a build could keep the workflow as it is, how to proceed is the
  person's choice, never yours;
- **a paid test job**, including validation or preflight jobs (step 6). The Run
  on Cloud prompt does not authorize a paid job. Approval for uploads, swaps,
  builds or deployments is not approval to run a job. A yes to a test job
  stands for this workflow: never ask it twice (step 6);
- **uploading any file from this machine** (a model, LoRA, input image, video
  or audio, or a custom node folder), by file name, one line per file: path,
  size, what it is for, where it goes. A yes covers the files you named; a new
  or changed file needs a new yes. A custom build's plan yes (step 4) covers
  its uploads;
- running **partner-API nodes**, which spend credits on top of the run (step 6);
- a **new build or deployment** (it costs money to build and to keep running);
- stopping, deleting or replacing any existing build, release or deployment.

If the person says no to a file the workflow needs, do not upload it and do not
swap in another file: say what is missing and that the workflow has not been
run. Do not call an incomplete setup ready.

Run comfy commands one at a time: one `comfy` command per shell call, never
chained or in parallel (a parallel watch can lose its sign-in mid-refresh).

## Ask once, then run to the end

The person answers once, then gets the link without being called back.

1. **Prepare, asking nothing.** Steps 1–3; in step 4, the build lookup (4.1),
   what a conversion would take (4.2) and, when a build is possible, its plan
   and costs (**A custom build**), editing nothing; in step 5, the model
   sources, dry runs and input fetches; `cat '<work>/saved.json'` (step 7) and
   `cat '<work>/consent.json'` (step 6), each its own command; and the
   workflow's partner-API nodes (step 6).
2. **One stop with every question that applies:** the route (4.1, 4.2, or
   which Cloud file an `ambiguous` model means), a custom build offered as its
   two build-from options; the GPU, when a build is offered; where each model
   with no `local_paths` is; the uploads (step 5); replace or new copy when
   `saved.json` was there (step 7); and the paid test (step 6), once for
   whichever route they pick. The message before the card holds the plan and
   costs of every route offered, so picking a build is the plan's yes.
3. **Then work to the end without stopping:** edit, upload, build, deploy,
   test, save, link. Right after the answers, show the steps left as a
   checklist with your client's plan tool when it has one (Claude Code's
   `TaskCreate` and `TaskUpdate`, or `TodoWrite` in older versions;
   `update_plan` in Codex; load a deferred one first), one short line each in the person's
   words (for example "Build the custom environment (5–15 min)", "Deploy on RTX
   PRO 6000", "Run one test job", "Save to Comfy Cloud"). Mark each as it starts
   and ends; a step that will not run (a declined or skipped test) is dropped or
   marked with why. Without a plan tool, post the steps once as a numbered list, then start each
   progress line with its number and the total ("2/5 Deploying on RTX PRO
   6000"). Ask again only for what no answer covered: a file in a
   folder the person named (its upload yes, once you have its size and hash),
   a request for a similar model, a file that changed after its yes, an upload
   or partner-API node nobody named, a build that would send more than its plan
   showed or add a pack or model it did not name, a replace conflict, or a
   blocker (with the ways to unblock it).

Ask with your client's question tool when it has one (`AskUserQuestion` in
Claude Code; `request_user_input` or `request_user_input_async` in Codex):
every choice and every yes, with the yes and the no as options, as many
questions per call as the tool takes (two calls in a row when you have more).
A tool that returns before the answer, such as `request_user_input_async`, does
not wait for you: end your turn after asking, and do nothing the question
decides until the reply arrives. Keep each card short but complete on its own:
name the file, GPU, price or change it asks about. Long detail (the plan, the
cost table, the node changes, every GPU that fits) goes in your message right
before the card. Without a question tool, number the options so the person can
answer in a word.

## 1. Get comfy-cli ready

- Run each of these as its own shell call, never joined with `&&`, `;` or
  `||`: `comfy workflow check --help`, then `comfy model upload --help`, then
  `comfy workflow save --help`. The first two must work and the third must list
  `--replace`. If one fails, upgrade once with the tool that installed comfy-cli
  (`uv tool upgrade comfy-cli`). Still failing: tell the person the version
  `comfy --version` prints, and stop.
- `comfy --json cloud status`. Not signed in: `comfy cloud login`. Note
  `cloud_workspace.id`; every upload and save goes to this workspace.

## 2. Read the workflow from the local server

The prompt gives `Workflow: workflows/<name>.json` and the server URL, plus a
`ComfyUI user:` line when the server runs with `--multi-user` (send it as the
`comfy-user` header; without the line, leave the header out). If it names no
workflow, ask which one. In the URL, percent-encode the whole path after
`workflows/`, subfolders included: each `/` as `%2F`, a space as `%20`, and
every other character outside letters, digits, `-`, `_` and `.` the same way
(`workflows/team/My flow.json` gives `workflows%2Fteam%2FMy%20flow.json`).

Work in one folder, `<work>`: `run-on-cloud/` plus the workflow's path after
`workflows/` (subfolders included, no `.json`) with every character other than a
letter, digit, `-` or `_` turned into `-` (`team/My flow (2)` gives
`run-on-cloud/team-My-flow--2-`). Two workflows can still map to one folder, so
`saved.json` and `consent.json` (steps 6 and 7) record the `workflow` path, the
`server` URL and the ComfyUI `user` (empty without a `ComfyUI user:` line);
ignore either file when any of them differs from this run's. Quote
every path. Put each name that comes from the workflow, the server or the disk
in single quotes, writing a `'` inside it as `'\''`: inside double quotes the
shell still runs `$(...)` and backticks.

`curl -sf "<server>/api/system_stats"` must answer JSON with
`system.comfyui_version`. Anything else means another program holds that port:
stop, tell the person the address does not answer as ComfyUI, and never run the
check on what it returns.

```shell
mkdir -p "<work>/inputs"
curl -sf "<server>/api/userdata/workflows%2F<encoded name>.json" \
  -H "comfy-user: <user>" -o "<work>/original.json"
```

Prefix every later comfy command that reads the local server (a `--where
local` run) with `COMFY_LOCAL_URL=<server>`; comfy-cli's default is
`127.0.0.1:8188`.

## 3. Check it against Comfy Cloud

Pass each model folder of the install that exists as its own `--models-dir`
(`comfy --json which`: each `desktop.model_roots[].path` for Comfy Desktop,
otherwise `<workspace_path>/models`); without them the check cannot see a local
file, so a model in another version than Cloud's never shows up as
`different_file`.

```shell
comfy --json which
comfy --json workflow check "<work>/original.json" --where cloud \
  --models-dir '<models folder>' [--models-dir '<another>']
```

`which` names the install comfy-cli tracks, which may not be the one serving
`<server>`. For a model with no `local_paths`,
`curl -sf "<server>/api/models/<folder>"` lists the files the server loads from
that folder. Listed there: the `--models-dir` folders are the wrong ones; ask
the person where that server keeps its models and check again. Not listed: the
first stop asks whether it is anywhere else on this machine (step 5). When the
item's `folder` is null or the lookup fails, ask where the model is.

| verdict | meaning | next |
|---|---|---|
| `covered` | Comfy Cloud has every node type and model | Prepare `original.json` for Comfy Cloud; step 6 asks before a paid test run |
| `covered_with_remaps` | Runs once its models point at Cloud's copies; `models[]` lists each remap | Write the remapped copy (below), then treat it as `covered` or `covered_needs_check` |
| `covered_needs_check` | Nothing is missing, but a model is not proven the same file as Cloud's copy | Use Cloud's copy; step 6 asks before a paid test run; name those models in the report |
| `needs_build` | A node type or model is missing, or the person's model differs from Cloud's; `blocking[]` lists each | Step 4 |

A model not proven the same is a `models[]` row whose `identity` is
`unverified`. Cloud's library models carry no content hash, so stock models
land here at best; that needs no model-choice question. A model proven
different (`source` `yours`, `identity` `different`) is never run on Cloud's
copy: it is in `blocking[]` as `different_file`.

A remap is lossless. Write it without asking, with the same `--models-dir`
folders:

```shell
comfy --json workflow check "<work>/original.json" --where cloud \
  --models-dir '<models folder>' --apply-remaps --out "<work>/cloud.json"
```

Run `cloud.json` from here on; rows still `unverified` make it
`covered_needs_check`. A `remap_not_applied` warning means `cloud.json` still
names the local file on that node: do not run it; report that node and model as
what blocks the run.

## 4. Something is missing: pick the route

When `blocking[]` lists only models, keep Comfy Cloud: upload each one (step 5),
then check the copy again. A model item's `reason`: `not_on_cloud` (no file of
that name), `ambiguous` (several; `models[]` lists the `candidates`, and picking
one changes the workflow unless the person's file is proven the same, so ask
with the choice in 4.2), or `different_file` (Cloud's file of that name differs
from the person's, so theirs is uploaded). When it lists a node type:

1. **An existing build in the workspace.** Always check this first, and look at
   every build, not only running ones:
   - `comfy --json deploy ls --workspace`: every deployment, running or
     stopped, each with its `releaseId` and `status`;
   - `comfy --json build ls`, every time, even when deployments exist; for each
     Build no deployment serves, `comfy --json build release ls --id <build id>`
     and its newest `deployable` release;
   - for each distinct release, `comfy --json build release manifest
     <release id>`: its `customNodes` and `models` are what it runs. Judge from
     the manifest, never from `build show`: a Build's definition can change
     after the release was cut.

   A release fits when it has every pack and model the workflow needs; keep the
   workflow unchanged. A running deployment serves it: use it. Only a stopped
   one: ask for a yes to start that deployment. None: ask for a yes to deploy
   that release (`comfy-deploy`). Either way name the GPU, its $ an hour and 0–1
   workers. Only when no release fits, go to 2.
2. **No build fits: ask whether to change the workflow or build.** Never change
   a node on your own: the workflow is the person's, and a custom build would
   run it as it is. Work out what a conversion would take (table below) without
   editing anything, then ask one question, starting by saying no existing build in the workspace
   runs the workflow:
   - **Change the workflow so it runs on Comfy Cloud.** One line per missing
     node type, with its count: `original ×n -> replacement`, whether the
     output stays the same or may change, and any control the person loses (a
     switch, toggle, group muter or bypasser, or selector fixed at its current
     setting). For example: "`Any Switch (rgthree)` ×7 -> direct links at
     today's settings: output stays the same, but you lose the switches."
     Nothing to build or deploy; it runs on standard Comfy Cloud.
   - **Keep the workflow as it is: create a custom build**, offered as its two
     build-from options (**A custom build**). Their exact nodes and models. A build usually takes 5–15 minutes, and its deployment costs
     money while it runs; you will show the costs before anything is built. A
     build needs every model from somewhere: Comfy Cloud's catalog, a file on
     this machine, or a download `url` (a private upload in the workspace's
     model library is not one, but it can prove a public copy is the same
     file: step 5). Look for a source first and name any model still without one.

   Recommend one in a line, with the reason: converting suits a change that
   keeps the output and takes away nothing they click; a build suits a workflow
   that would lose controls or change its output. If a blocking node is not
   swappable, there is no conversion to offer: say which node and why, and
   offer the build. On "change", make the changes on a copy (below), check it,
   and go on to step 5. On "keep", go to **A custom build**.

| kind | what counts | how to describe it |
|---|---|---|
| Lossless | A utility node swapped for the core node that computes the same thing (math, string concat, a LoRA stack as chained `LoraLoaderModelOnly` at the same strengths, a pipe node as direct links); removing a node no output depends on and nobody operates | "The output stays the same." |
| Lossy | Anything else that changes what is generated: another model, LoRA, VAE, ControlNet, upscaler or embedding; dropping a LoRA or a node that touches pixels; another implementation of the same job | Say how the output may change |
| Removes a control | A node the person operates to choose what runs (a switch, a group muter or bypasser, a selector, a toggle), replaced by fixed wiring or removed | Say it keeps today's setting, and that they lose the control. Never call this lossless, even when the output for the saved settings is the same |
| Not swappable | A node that does the core work and has no core equivalent on Cloud; a loader for weights core cannot load (Nunchaku/SVDQuant, GGUF-only packs), since replacing it means another model | No conversion: custom build |

Offer only replacements from the same model family, taken from
`comfy --json nodes show <Loader> --where cloud --all-choices`. When the person
turns a change down, make `cloud.json` again from `original.json` (never from a
copy you edited); never save a copy that carries a change they did not approve.

### A custom build

Read `comfy skills show comfy-build` and `comfy skills show comfy-deploy`, and
follow them, except that their questions go in the first stop and the release
policy stays open (no restriction on models or partner nodes): say so, do not
ask. A build that cannot include everything the workflow runs cannot run
it: a model with no source after step 5's search, or a node class no pack provides.
While preparing, run `comfy --json build init "<work>/probe" --name probe
--from-workflow "<work>/original.json"`: it names the packs and models the
workflow needs. For a class its advisories say nothing installable provides,
look it up in ComfyUI-Manager's node map, `curl -s
https://raw.githubusercontent.com/ltdrdata/ComfyUI-Manager/main/extension-node-map.json`
(each repository with the classes it provides): prefer one repository that
provides every such class, add it to the build by its GitHub URL, and put any
model its README says it loads on its own (InsightFace's antelopev2 for
InstantID) in the plan, with a source. A class no repository there provides
blocks the build: build nothing, name it and ask how to unblock it
(**Report**); with no way left, stop with no save and no link.

Ask 1 and 2 in the first stop, each its own question, never deciding either for
the person, with 3 in the message before the card. Nothing uploads, builds or
deploys before that yes.

**1. What to build from**, the route question's two build options. Your local installation is the default,
unless the check below finds it cannot run the workflow; then **Just this
workflow** is the default, and you say why:

- **Your local installation.** Every custom node pack and model in the ComfyUI
  install that serves `<server>`: comfy-build's Path A, with
  `--comfy-url <server>`, plus `--custom-nodes-dir` and each `--models-dir` from
  `comfy --json which` when it names them (`desktop.custom_nodes_dir`,
  `desktop.model_roots[].path`). It can then run your other workflows too, but
  it is bigger: slower to start and more storage.
- **Just this workflow.** Only the packs and models this workflow uses:
  comfy-build's Path B, `init --from-workflow` on `<work>/original.json`.
  Smaller, quicker to start and cheaper to store, but made for this workflow.

The check: for each node type in `blocking[]`,
`curl -sf "<server>/api/object_info/<node type, percent-encoded>"`; an empty
`{}` means the server does not load it. Skip node types that exist only in the
browser and never appear there, such as rgthree's Fast Groups Muter and Fast
Groups Bypasser. If the server lacks one the workflow runs, a build from the
installation could not run it: still ask, never choose for the person; name
those node types and recommend **Just this workflow**. When the installation
holds none of the workflow's models, say so in its choice: the build would carry
its other models and still need sources for these. If you cannot tell which
folder holds the installation, ask.

**2. The machine.** The VRAM depends only on the models the workflow loads, not
on 1. Read
`comfy --json deploy refs compute`. Add up the model files the workflow loads (a
local file's size on disk, or for a `url`, the `x-linked-size` or
`content-length` from `curl -sIL '<url>'`); the VRAM it needs is that total plus
about 20%, more for video or very large images. List every GPU that fits,
cheapest first, each with its VRAM and `pricePerHourUsd`, and recommend one with
the reason ("RTX PRO 6000, 96 GB: your models come to about 44 GB"). Ask for a
region only when it matters; otherwise use the widest location that offers the
GPU. Workers are 0 to 1 (`--min 0 --max 1`).

**3. The plan and its cost.** For each build option, show comfy-build's "Before
you cut" plan (the probe's packs and models; for the installation, also its
other packs and the size of its model folders) with this cost table for the
recommended GPU. Picking a build option is the plan's one yes. Take every price from `comfy --json deploy refs compute`
(`pricePerHourUsd`, `storage[].pricePerGbMonthUsd`), never from memory:

| | |
|---|---|
| GPU | `<label>`: $`<pricePerHourUsd>` an hour while a worker runs |
| Workers | 0 to 1: scales to zero when idle, so idle time costs nothing; the first job after idle waits for a cold start |
| Storage | `<GB>` of models stored with the build × $`<pricePerGbMonthUsd>` per GB-month ≈ $`<n>` a month |
| Build | usually 5–15 minutes |

Storage counts the models each option carries (for **Just this workflow**, the
total from 2). List each model the build carries with its
size, mark the ones this workflow does not use, and say they can be left out to
save storage: models in the build are stored with it (the storage cost) and load
quickly when a worker starts; a model left out is downloaded every time a worker
starts, which makes each cold start slower and costs GPU time. The plan's yes covers the uploads it lists. After it, run `push --dry-run`;
ask again only if it would send more than the plan showed or add a pack or model
the plan did not name (a pin moved to another published version of the same
pack is neither).

**4. Build.** Right before the first `comfy build push` without `--dry-run` (or
`release create`, when nothing needs pushing), send this line on its own, even
though the cost table said it: "The build usually takes 5–15 minutes." Then
follow comfy-build to watch it and comfy-deploy to deploy with the approved GPU,
region and workers.

**Making a swap.** Copy the workflow to `"<work>/cloud.json"` (or edit the
remapped `cloud.json`) and run every edit on that copy with `--where cloud`:
`comfy --json workflow add-node`, `workflow connect` per link,
`workflow set-widget` per value, `workflow delete-nodes` for the nodes it
replaces. Name slots in `connect`, never their numbers (`<id>.INT`, not
`<id>.0`): the core node may order its outputs differently. `comfy --json nodes show <class> --where cloud` lists a
node's inputs and outputs by name.

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

**A model** from a `blocking[]` item: `filename`, `folder` (the models folder,
or null), `url` (where the workflow says it downloads from, or null) and
`local_paths` (copies under `--models-dir`). No `local_paths`: look for a
public source (below). Without a verified one, ask once in the first stop, in
one question naming every such model: "Is `<file>` anywhere else on this
machine, or do you have a download link?" A folder: check again with it as
another `--models-dir`. A link: use it as the source (below). Several `local_paths`:
ask which one. A file not on this machine: say so, give the `url` if there is
one, and upload nothing for it. Never offer, suggest or invite another model in its place, except a near-name
file (below). If the person asks for a similar one, do not pick it yourself: list
up to three same-family choices from `comfy --json nodes show <Loader> --where
cloud --all-choices`, say each may change the output, and ask which, with "keep
the workflow as it is" as a choice; change the copy only after they name one. A
model the workflow needs that stays missing blocks the run: no test job, no save
and no link; report it as the blocker.

**Finding a model's source**, while preparing, even when a node also blocks: the
item's `url`; `comfy --json build refs resolve '<file>'`; the README of the pack
whose node loads it, which often links its models; then Hugging Face by name, `curl -sG 'https://huggingface.co/api/models'
--data-urlencode 'search=<file name without extension>' --data-urlencode 'limit=20'`,
and again without a trailing version (`myVae_v10` -> `myVae`) when
that finds nothing. For each repo, `curl -s
'https://huggingface.co/api/models/<repo>/tree/main?recursive=true'` lists its
files with `size` and `lfs.oid` (the sha256). Only a file of the exact name counts. It is the person's file when its sha256
matches their local copy (`shasum -a 256`), or, when their only copy is a
private upload in the Cloud model library, when its `size` matches that copy's
(`comfy --json model show '<file>' --where cloud`) and, downloaded into
`<work>/models/`, its upload dry run (below) says `already_present`. Then use
`https://huggingface.co/<repo>/resolve/main/<path>` as its `url` and say where
it came from. Otherwise offer it in the first stop (repo and size) and use it
after a yes; the same goes for a link the person gives. With no exact-name source, look for
near-name files: same folder, a name that differs only in case, separators or a
version (`AlbedoBaseXL` and `albedobaseXL_v04`), in the person's own uploads
(`comfy --json model search --text '<name without version>' --type <folder>
--where cloud --mine-only`) and in the Hugging Face file lists above. Offer up to
three in the first stop, or in the blocked report's question, each by name,
place and size, saying the output may differ if it is another version; use one only after the person picks it, and only on the copy. A build takes the `url` as the model's source; on standard
Cloud, download it into `<work>/models/` and upload it as below, asked by name.

The dry run sends no file; it hashes the file here and reads the library:

```shell
comfy --json model list-folders --where cloud
comfy --json model upload '<file>' --type <folder> --workspace <workspace id> --where cloud --dry-run
```

`<folder>` is the item's `folder` when `model list-folders` lists it; otherwise
the folder the loader reads from, and say so. `already_present`: Comfy Cloud has
these exact bytes under that name; nothing to upload or ask; use its
`loader_path`. Otherwise ask by `filename`, with the dry run's path, `size`,
`sha256` and destination folder. After the yes:

```shell
comfy --json model upload '<file>' --type <folder> --workspace <workspace id> --where cloud \
  --yes --expected-sha256 <approved sha256>
```

Set the loader on the copy to the returned `loader_path`, then check the copy
again. A null `loader_path` (after an upload or on `already_present`) means
Comfy Cloud gave no loader value for that name: do not put the file name in its
place; report the model as what blocks the run. On a failure, nothing ran on
Comfy Cloud yet:

| error | what to do |
|---|---|
| `model_upload_needs_confirm` | `--yes` was missing: ask if you have not, then add `--yes --expected-sha256` |
| `asset_upload_wrong_workspace` | This sign-in uploads into another workspace (`details.current_workspace`); nothing was sent. Upload only after they sign in to the approved one (`comfy cloud login`) or approve the current one |
| `asset_upload_invalid`, `file_changed` | The file changed since the dry run: dry run and ask again with the new `sha256` |
| `asset_upload_invalid`, `name_in_use` or `name_unverified` | Another file has (or may have) this name, and uploading would change what it loads for the whole workspace. Copy the file under a new name into `<work>/`, then dry run and ask with that name |
| `asset_upload_invalid`, `bad_expected_sha256` | Pass the dry run's 64-hex `sha256` |
| `asset_upload_invalid`, other reasons | Fix the file or `--type` problem the message names, or report it as what blocks the run |
| `asset_upload_unverified` | Stored, but Cloud's hash does not match or could not be read: do not run a workflow that loads it; report it with `details.asset_id` |

**An input file** (image, video, audio) a loader names: fetch it from the local
server before the first stop, then ask there with its name, size, full
64-character `shasum -a 256` hash (never shortened), and destination (the Comfy
Cloud input files of `cloud_workspace`):

```shell
curl -sfG "<server>/api/view" --data-urlencode 'filename=<file>' \
  --data-urlencode 'subfolder=<subfolder>' --data-urlencode "type=input" \
  -H "comfy-user: <user>" --create-dirs -o '<work>/inputs/<subfolder>/<file>'
shasum -a 256 '<work>/inputs/<subfolder>/<file>'
```

If `/api/view` does not return the file, the input is missing. Do not search the
disk for it and do not use another file. A missing input does not block the save
(the person can add it on Cloud): skip the test job, save, and report it as in
**Report**. Keep the subfolder in the path, so two inputs of the same name stay
two files. A subfolder or file name with `..` or a leading `/` would write
outside `<work>`: do not fetch it; report that input as what blocks the run.

After the yes, upload that same file without fetching it again (a file read
again needs a new yes with its new hash), and set the loader on the copy to the
returned `uploads[].cloud_name`:

```shell
comfy --json upload '<work>/inputs/<subfolder>/<file>' --where cloud
```

## 6. The paid test job

Fetch every input file the workflow loads first (step 5). If any is missing,
skip the test and its question: say the test was skipped for the missing input,
and go to step 7.

**A yes stands.** Right after the person says yes, write `<work>/consent.json`:
`{"test_job": true, "workflow": "<path>", "server": "<url>", "user": "<ComfyUI
user or empty>", "target": "<standard
or deployment id>", "gpu": "<GPU or null>", "price": "<what you quoted>",
"sha256": "<shasum -a 256 of the file the test runs>", "partner_nodes": [...],
"approved_at": "<date -u +%Y-%m-%dT%H:%M:%SZ>"}`. When they said yes earlier in
this conversation, do not ask again for a retry they ask for: run the test and
say it ran on their earlier yes; that yes is the ask the prompt requires. A
`consent.json` from an earlier session counts the same only when its
`workflow`, `server`, `user`, `target`, `gpu` and `price` match this run (the
price you would quote now) and the file the test runs still has its `sha256`. Either way, ask again when the target moves to
another deployment or GPU, or the workflow runs partner-API nodes the yes did
not name. A no is not remembered.

Otherwise ask in the first stop, once for whichever route they pick: "Would you like
me to run one test job? It costs money (Cloud credits or deployment GPU time).
You can also run it yourself from the saved workflow link." Name each offered route's price (Cloud credits on standard Cloud, the GPU's $
an hour on a deployment) and any partner-API nodes with their extra charges: the workflow's node
types that `comfy --json nodes ls --api-only --where cloud --limit 1000` lists.
Wait for an explicit yes; the request to run on Cloud and approvals for
preparation do not count. A yes covers one job per test: never submit a second
on your own.

If they decline or have not answered, skip every job-submitting command in this
step, including `deploy run --no-upload`: it can execute a paid job, not just
check files. Use only checks that do not execute a workflow, then continue to
step 7 to save the prepared workflow and return its link, with the route's ready
message from Report and "It hasn't been run yet." A request to only prepare the
link follows this path without asking.

After a yes, follow the route below. Never call it working before a real job
succeeds.

- **Comfy Cloud:**
  `comfy --json run --workflow '<file>' --where cloud --wait`, then
  `comfy --json download <prompt_id> --where cloud --out-dir "<work>/outputs"`.
  `spend_consent_required` means nothing was submitted: the workflow has
  partner-API nodes (`details.partner_nodes`) that spend credits beyond the run.
  If the test's yes named them, run the same command with `--allow-spend`;
  otherwise ask, naming them, and add `--allow-spend` after a yes. On a no, go
  to step 7 without running the job. `run` and `set-widget` look a model up in the workspace's
  Cloud model library before refusing it. A model you uploaded still refused
  as unknown means the library does not hold it under that name, or the lookup
  failed; the error's hint says which. Never hand-edit around it: report it
  as the blocker, with no save.
- **A deployment** (existing or custom build): first read
  `comfy --json deploy logs --deployment <id>`; `data.comfyuiLog` must show no
  `IMPORT FAILED` for a pack the workflow uses. Make the API file against the
  local server, which has the packs:
  `COMFY_LOCAL_URL=<server> comfy --json run --workflow '<file>' --print-prompt --where local`,
  `.data.prompt` into `"<work>/api.json"`.

  `deploy run` picks local files itself (any string input with a file
  extension can match one, and of two same-named files it picks by its own
  order), so take the upload list from it, never build it yourself. Run it
  first with `--no-upload`, which sends nothing:

  ```shell
  comfy --json deploy run --deployment <id> --workflow "<work>/api.json" --no-upload \
    --wait --output-dir "<work>/outputs"
  ```

  If the deployment has every file, it runs. Otherwise it fails with
  `deploy_asset_missing` before submitting: `details.missing[]` is the exact
  upload plan, one entry per file with `local_path`, `bytes` and `hash`. An
  entry whose `shasum -a 256 '<local_path>'` matches an input the person
  already approved is covered; ask about the rest in one message, by name, with
  `local_path` and size. On a no, report the missing inputs and that the
  workflow has not run; do not call the setup ready. Once every entry is
  covered, run the same command from the same folder with `--no-upload`
  replaced by one `--allow-upload <hash>` per file:

  ```shell
  comfy --json deploy run --deployment <id> --workflow "<work>/api.json" \
    --allow-upload <hash> --allow-upload <hash> \
    --wait --output-dir "<work>/outputs"
  ```

  It sends a private copy of only those files, so a file that changes meanwhile
  is not sent. Report the local files `data.assets.files[]` names. If either run
  fails with `deploy_asset_missing`: with `details.missing[]`, a file differs
  from the approved plan and nothing outside it was sent, so ask about the new
  entries and never pass a hash nobody approved; with `details.server_code`
  `missing_asset`, the workflow names an asset on the deployment's side this
  account cannot reach, so report it as unreachable and that the workflow has
  not run, and do not call the setup ready.

Keep the job id. On a timeout, watch the same job; never submit a second paid
job to recover a download. After a failed job, report it and do not resubmit on
your own; a retry the person asks for runs on their earlier yes.

## 7. Save it, then hand back the link

Save the prepared editor-format file whether or not a test job ran (the copy, if
you changed one, never `api.json`), with `--deployment <id>` when a deployment
was selected, even when the person declined the test. Do not save when setup is
blocked (a missing model, a build that cannot be made): end with the blocker
instead. A missing input file does not block the save. Save one Cloud copy per
workflow, not one per attempt. Start each save command with
`COMFY_WORKFLOW_SOURCE=run_on_cloud` (for that command alone) so Cloud can count
saves from this flow. The `cat '<work>/saved.json'` you ran while
preparing decides the case; a save with no `filename` (record-only) is replaced
by a plain save, and `--expected-sha256` goes only with a `sha256` that is set:

- **You already saved this workflow in this conversation:** replace that save,
  without asking, with the `filename` and `sha256` from that save's envelope:

  ```shell
  COMFY_WORKFLOW_SOURCE=run_on_cloud comfy --json workflow save '<file>' --replace '<filename>' --expected-sha256 <sha256> --where cloud
  ```

- **`<work>/saved.json` was already there when you started** (a save from an
  earlier session): the first stop asked once, naming its `filename` and
  `saved_at`: replace that save, or save a new copy? Replace: the command above
  with its `filename` and `sha256`. New copy: the plain save below.
- **Otherwise**, the plain save puts the workflow in a new folder in the Cloud
  Workflows sidebar:

  ```shell
  COMFY_WORKFLOW_SOURCE=run_on_cloud comfy --json workflow save '<file>' --name '<name>' --where cloud
  ```

Right after every save that succeeds, a replace or record-only save included
(null for any field the envelope lacks), and before you reply, write
`<work>/saved.json` with that envelope's `filename`, `workflow_id`, `sha256`,
`url` and `deployment`, plus the `workflow` path, the `server` URL, the ComfyUI
`user` and `saved_at` from `date -u +%Y-%m-%dT%H:%M:%SZ`. A
stale `sha256` there makes the next replace fail.

| replace error | what to do |
|---|---|
| `workflow_replace_changed` | Someone edited that workflow on Cloud after your save. Ask once: replace their edit, or save a new copy. Replace: the same command without `--expected-sha256` |
| `workflow_replace_missing` | The saved file is gone from the sidebar. Use the plain save, and say it is a new copy |
| `workflow_replace_unsupported` | This workspace saves records only. Use the plain save |

Hand back `data.url` as returned. `data.url_opens` says what it opens:
`workflow`, only the saved workflow
(`?workspace=<id>&workflowId=<data.workflow_id>&deployment=<id or standard>`;
Cloud checks the link against the saved file when it opens), or `folder`, every
workflow in the new folder
(`?workspace=<id>&workflows=<folder>&deployment=<id or standard>`), with
`data.folder_link_reason` saying why: `workflow_links_off`, or
`record_not_found`, `record_ambiguous` or `record_lookup_failed` (the save could
not tell which workflow Cloud made; `data.warnings[]` carries
`workflow_record_unknown`), or `record_stale` (a replace whose workflow record
Cloud did not update; `workflow_record_stale`). Read any value not listed here
as `folder`.

No `data.url` but `data.filename`: the file is in the sidebar but the CLI could
not build a link; give `data.filename`. Neither: the save is a record only,
with no sidebar entry and no link. Say so, give `data.workflow_id`, and read
`data.warnings[]`:

- `api_format_saved_as_record`: sidebar saving is on, but the file is in API
  format, which the sidebar cannot open. If you saved `api.json` or another API
  file you made, save the editor-format copy instead. If the workflow on the
  server is itself in API format, this record is the result: say there is no
  sidebar entry or link, give `data.workflow_id`, and add exactly: "Opening it
  in the ComfyUI editor and saving it there gives a file that gets a link." Do
  not ask for an export and never suggest running this again.
- no warning: the workspace saves workflows as records only (sidebar saving is
  off, or its setting could not be read).

## Report

Keep the final message short: a status line, at most six short lines on what
you did (a hard cap), then the link. No step-by-step account, "what's next"
section, tips, lists of model names, account of edits, checks or validation,
notes about your own process or slips, or lines about what you did not do.

1. **Status.** "Your custom build is ready." for a new custom build, or "Your
   workflow is ready on Comfy Cloud." for standard Cloud or an existing
   deployment. With a missing input, lead instead with "Your workflow is saved
   on Comfy Cloud, but it needs `<file names>` before it can run.", naming each
   file, never a count or a description. Never call it ready while an input or
   model is missing. If no job was submitted, add "It hasn't been run yet." and
   whether the test was declined, is awaiting consent, or was skipped for a
   missing input. Report a failed or pending job as failed or pending. If setup
   or saving is blocked, lead with what could not be completed.
2. **What you did**, one short line each, only those that apply:
   - where it runs: standard Comfy Cloud, an existing custom environment, or a
     new custom build;
   - the changes to the copy and the controls removed, in one line; the files
     uploaded;
   - on standard Cloud only, one line for the models that run on Cloud's copy
     unproven: how many (`models[]` rows whose `identity` is `unverified`),
     named only when three or fewer, and that the output may differ if their
     files are other versions;
   - the test job: succeeded (job id, outputs on disk) or failed (job id);
   - inputs to fill in, and partner-API nodes that spend credits beyond the run;
   - for a deployment: GPU, $ an hour, worker bounds, and
     `comfy deploy stop --deployment <id>` to stop the billing.
3. **The link, last**, with the call to action: "Open it to run your workflow
   on Comfy Cloud: `<data.url>`". A record-only save ends with the line saying
   there is no link instead, and never changes the status line. A blocked
   report is two or three lines of one sentence each, with no other findings:
   what is missing, and that the workflow has not been run or saved. Then ask
   how to unblock it, as choices: where to get the missing file (a folder on
   this machine, a download link), or a route that still works, such as
   changing the workflow when 4.2 offered it. Never end on what was or was not
   uploaded, swapped, saved or run, and never name or invite another model in its place except a near-name file
(step 5).
