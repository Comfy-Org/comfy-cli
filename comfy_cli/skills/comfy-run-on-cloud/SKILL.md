---
name: comfy-run-on-cloud
description: Use when a developer clicked "Run on Cloud" in local ComfyUI or Comfy Desktop, pasted a Run-on-Cloud prompt, or wants a local workflow (or the whole install behind it) running on Comfy Cloud with a link to open it. Also use for "Update existing build" from the same modal, and for a pasted repair prompt after such a build or deployment failed.
---

# comfy-run-on-cloud

One workflow in, one link out. Users care that the workflow runs on Cloud, not how. So:

- Use **stock Comfy Cloud** when it already has everything.
- **Convert** the workflow for stock Cloud when the conversion is lossless.
- Reuse a **workspace Build**, or make a **custom build**, for everything else.

Whichever route you take:

- Package **only what the workflow loads**, unless the user chooses the whole instance for a custom build.
- **Never change what the workflow generates without a yes.** A lossless fix is yours to make; a lossy one is the user's call.
- Never upload a file from disk without a per-file yes.
- Hand back the link only once the target can run.

For a custom build, this skill sequences two others. Read them when a step points there; do not restate them.

- `comfy skills show comfy-build`, plus its `comfy-build-authoring`, `comfy-build-pins` and `comfy-build-failures` references.
- `comfy skills show comfy-deploy`, plus `comfy-deploy-failures`.

## What the click authorizes, and what it does not

The modal, or the prompt it generated, names the build name, GPU, workspace, workflow, and whether to create a new Build or update one. Sending it **is** the user's yes to:

- run that workflow on Cloud, including **lossless** conversions;
- if a build is needed, build and deploy it on that GPU with **`--min 0 --max 1`**.

It is **never** a yes to any of these, and each needs its own answer in chat:

- a **lossy** conversion (step 3);
- uploading any file from this machine: models, LoRAs, input images, video or audio (step 4);
- replacing, stopping or deleting an existing Build, release or deployment;
- a different GPU, or worker bounds other than 0/1.

"Don't ask me anything, just do it" answers none of these. Ask once, in one short message, then act.

## The procedure

**1. Find the workflow and the install.**
- comfy-cli is a separate install, not bundled with Desktop. If `comfy` is missing, install it with `uv tool install comfy-cli`, then `comfy skills install --skill comfy-run-on-cloud`. If `comfy build --help` lacks `inventory`, upgrade it the same way it was installed.
- If the prompt names no workflow file, ask which file. Never package an install instead, and never invent a path.
- Do not trust `comfy which`; it can name a directory that does not exist. Use the running server and the Desktop config: `comfy skills show comfy-run-on-cloud-local-install`.

**2. Inventory the workflow against stock Cloud.** Run it against the running server, **every** model root and the ComfyUI input folder:

```shell
comfy --json build inventory <workflow.json> --cloud --server http://127.0.0.1:<port> \
  --desktop --models-root <any root not in Desktop settings> --input-root <ComfyUI input folder>
```

It reports what the workflow loads (active classes, models with subfolders, `embedding:` refs, input files with the name Cloud would store each under, recorded pack versions, `implicit` loaders) and a `cloud` section: `verdict`, `blocking`, `lossless_removable`, and per model a `status` (`present` / `remap` / `absent`) and `weights` (`same_size` / `different_size` / `no_local_copy` / `unknown`). **The report is the scope.** For each `implicit` row, read the pack source and add the weights it loads.

**3. Classify every change Cloud needs, then pick the route.**

| Kind | What counts | What you do |
|---|---|---|
| **Lossless** | Only these: a `remap` (the same file in another folder) whose `weights` is not `different_size`; removing a `lossless_removable` node and wiring its model input straight to its consumer; a utility node swapped for its exact core equivalent (math, string concat, LoRA stack → chained `LoraLoaderModelOnly` at the same strengths, a context pipe → direct links); removing a node no output depends on; setting a widget that a newer pack added to the value that reproduces the old behavior. | Do it on a **copy**, without asking. List each change in the report. |
| **Lossy** | Anything else that changes inputs to generation: a different model, LoRA, VAE, ControlNet, upscaler or embedding, including a Cloud file with your file's name but `different_size` weights; dropping a LoRA, embedding or any node that touches pixels; a different implementation of the same job (e.g. a Nunchaku loader → stock `UNETLoader`). | **Ask** (below). Never apply one without a yes. |
| **Not convertible** | A node class Cloud lacks that does the core work, with no stock equivalent (e.g. InstantID). | Custom build. |

Then the route:

| Route | When | Next |
|---|---|---|
| **A. Stock Cloud** | No blockers, or only lossless changes. | Apply them, then **pre-flight** (below). Step 6. |
| **B. Existing Build** | A workspace Build already contains every blocker, **and its deployment log shows every pack the workflow uses loading** (no `IMPORT FAILED`, no `ModuleNotFoundError`). Check `comfy --json build ls`, `build show --id`, `deploy logs`. | Reuse its deployment (`deploy start` if stopped), then the step-5 smoke run. Step 6. |
| **Ask** | At least one lossy change, and the user has not chosen. | Ask (below), then A or C. |
| **C. Custom build** | Anything not convertible, or the user chose "exact". | Steps 4–6. |

**Asking about a lossy conversion.** One message, in this shape:

1. **What changes:** one line per swap, `original → Cloud replacement`, and what the user will see (a different face, a different style, softer or crisper detail). Offer replacements from the same family only (SD1.5 / SDXL / Flux / Wan version), taken from the loader's `choices` in `comfy --json nodes show <Loader> --where cloud`. If you can't tell the family, offer only option 2.
2. **The options:**
   - **Convert:** runs on stock Cloud now, pays per run, nothing to build. Output will not match the local run.
   - **Deploy exactly what you have (custom build):** output matches. Costs more: about 10 minutes to build, model storage billed monthly, GPU time billed while a worker is up, and a cold start can wait minutes for a free GPU. Uses one of the workspace's release slots. There is no Open UI link for a deployment yet; it is reached by API (`comfy deploy run`).
3. The **build scope** question (below), for option 2.
4. Any **upload** questions from step 4, in the same message.

Recommend one option in a clause, then wait. A user note such as "demo in 20 minutes" informs the recommendation; it does not choose for the user.

**Build scope: ask it whenever a custom build is on the table** (option 2 above, or route C). Same message:

- **Just this workflow** (the default; recommend it unless the user wants more): only the packs and models the inventory lists. Smaller, faster to build, and fewer files to upload.
- **The whole instance:** every custom node pack in the install and every model in its models folders, so any workflow from this install runs on the deployment. State the real size before they choose: the pack count (`ls <data>/custom_nodes`), the model count and total GB per models root (`du -sh`), and how many of those files have no public source and would need an upload. Build time, storage cost and the chance of a dependency conflict all grow with it. List a shared models root (a Desktop `modelsDirs` entry outside the data dir) separately; it holds weights for other work and is never included without its own yes.

On "whole instance", write the spec from the install per `comfy-build` Path A (`comfy build init <install> --name <name> --python <data>/.venv/bin/python`) instead of `--from-workflow`, then prune it per `comfy-build-authoring`. Uploads are still gated: show the full list of files with no public source, and a yes to that listed set counts as the per-file yes. A file added later is not covered.

Before you send it, **check the workspace limits** (below), so the custom-build option says whether a slot must be freed first. **A swap only works where its replacement exists:** stock Cloud's models are not inside a custom build, and a build's models are not on stock Cloud. Never offer "swap on the existing Build" unless that Build lists the replacement.

**Pre-flight every converted copy on the server** before saving or running it. The inventory cannot see settings a pack version shifted:

```shell
comfy run --workflow <copy> --print-prompt --where cloud > api.json   # graph is data.prompt
comfy --json workflow validate --workflow <data.prompt saved as a file> --where cloud
```

A `shape_mismatch` on a widget means the Cloud pack is a different version than the workflow saved. Set that widget by name with `comfy --json workflow set-widget <copy> <node>.<widget> <value> --where cloud`, using the value that reproduces the saved behavior (usually the Cloud default). If no value reproduces it, the change is lossy: ask.

**Workspace limits** (routes B and C, and before asking; see the pricing page): releases (10 per workspace observed; count with `build release ls --id` per Build), active deployments (5; `deploy ls --workspace`, stopped ones don't count), and Builds (10 on Pro; `build_limit` on push). If one is full, ask which release, Build or deployment to remove, ones with no live deployment first. Never pick for the user.

**4. Uploads and the custom-build spec** (route C, or any route that uploads).

For "just this workflow", write the spec workflow-scoped (for "whole instance", see Build scope in step 3):

```shell
comfy --json build init <dir> --name <name> --from-workflow <file> --comfy-version <local version>
```

Then fix it by hand per `comfy-build-authoring`, from the inventory:
- **Pack pins**, in order: the locally installed version; a pin a workspace Build already deployed cleanly; the workflow's recorded `ver`; the newest registry version that is not Flagged (all Flagged → `repository` + `gitRef`).
- **A pack that imports a Python package nothing installs cannot be built.** `pipDependencies` only forces versions of packages something already requires; it never adds one. ComfyUI-nunchaku is the known case: it needs the `nunchaku` wheel, so every Nunchaku node fails to import while the deployment still reports `ready`. Check each pack's imports against its `requirements.txt` before building. If one is missing, tell the user the build cannot run those nodes, and offer the lossy conversion for them instead.
- **Model `type`:** the folder under `models/` plus the widget's subfolder (e.g. `checkpoints/sdxl`).
- **Model entries:** a local file whose sha256 matches a `comfy build refs resolve` candidate becomes a `sourceUri` fetch. A local file with no public match is an **upload**: `{type, filename, localPath, sha256, sizeBytes, source: local}`.

Then `comfy build validate` and `comfy build push --dry-run`. The dry run skips the registry-pin check, so a Flagged version can still fail the real push.

**Skip inputs Cloud already has.** Cloud names an uploaded input by the blake3 of its bytes, and the inventory computes that name. If an input's `on_cloud` is `true`, set the loader to its `cloud_name` and don't ask. `false` only means the loader's list doesn't show it; the lists can omit uploads.

**The upload question** lists one line per file, for models, LoRAs and **input files** (images, video, audio) alike: name, size, why it must come from this machine, destination (private build storage, or the Cloud input library for inputs), and cost or "unknown". Then **Upload** / **Cancel**, per file. Cancel drops the file; say what will then fail. Also give licenses and NSFW flags for any model (Hugging Face `cardData.license`), and say that `init` sent the workflow JSON to the builder.

**5. Build, deploy, verify** (route C).
- `comfy build push`, then `comfy build release create --target linux/nvidia --watch`. On failure read `comfy-build-failures`: one cause per cut, at most three cuts.
- `comfy deploy up <spec> --gpu <gpuClass> --region <region> --min 0 --max 1 --watch`, with the prompt's GPU. On a refusal, say so and ask; never switch GPUs on your own. On "Update existing build", `up` moves the existing deployment onto the new release.
- Report the idle scale-down from the deployment's own `keepWarmSeconds`, not a remembered default. Outside a Build folder, read a deployment with `comfy --json deploy show --deployment <build>/<name>`; `deploy status` needs the spec folder.
- **`ready` is not working.** Require both:
  - the ComfyUI log (`comfy --json deploy logs` → `data.comfyuiLog`) shows no `IMPORT FAILED` and no `ModuleNotFoundError` for any pack the workflow uses;
  - one smoke run: `comfy deploy run --workflow <api.json> --asset-root <dir with the inputs>`. Build that API file from node definitions that include the build's packs: `comfy run --workflow <file> --print-prompt --where local` against a running local ComfyUI that has them. Converting `--where cloud` drops every widget of a class stock Cloud lacks (the node comes out with empty `inputs`). With no such server, fill those nodes' inputs by name from the pack's `INPUT_TYPES`.
- A smoke run on the build can fail validation where Cloud passed (a pack version requires an input Cloud's version lacks, e.g. USDU's `batch_size`). Add the input with the default that reproduces the saved behavior and re-run.

**6. Save, then link.**
- Save the workflow (the converted copy on route A) so it shows in the Workflows sidebar: prefer the Comfy Cloud connector's `save_workflow` with inline `workflow_json`. Its `workflow_path` upload and `comfy workflow save` both create a record the sidebar does not list. Check with `list_saved_workflows`: a usable save has a size and is not `filename_ambiguous`.
- Upload input files only with the step-4 yes; set the loader to the returned `cloud_name` on the copy before saving. Cloud's loader dropdowns show only some uploads, but every uploaded name loads.
- There is no private deep link yet. Hand back "https://cloud.comfy.org → Workflows → `<name>`", the local file path, and a public share link only if the user says yes to publishing.
- Route A can also run it now: `comfy --json run --workflow <copy> --where cloud --wait`, then `comfy --json download <prompt_id>`.

**Report:** the route; every lossless change; every lossy change the user approved; what was uploaded; for routes B and C the Build, release and deployment ids, GPU, region, 0/1 bounds and `comfy deploy stop <spec>`; what still needs the user.

## Common mistakes

| Mistake | Instead |
|---|---|
| Building because Cloud lacks a node class | Check whether the class is `lossless_removable` or has an exact core equivalent first |
| Swapping a model because "the user said don't ask" | A lossy swap changes the output; ask once, with the custom-build alternative |
| Treating a Cloud file with your file's name as your file | Check `weights`; `different_size` is a different model, so using Cloud's is lossy |
| Saving a converted copy the inventory called covered | Pre-flight it on the server; pack versions shift widget meanings |
| Calling a deployment done at `ready`, or reusing one because its Build lists the packs | Read the import log and make one smoke run |
| Scanning the whole install without asking | Ask the build-scope question; default to just this workflow |
| Treating the modal click as upload consent | Ask per file, for models and input files |
| Running several `--watch` commands in parallel | Run comfy commands one at a time |
| Treating a 401 mid-watch as a reason to log in again | Retry once; re-read with `release show` or `deploy status` |
