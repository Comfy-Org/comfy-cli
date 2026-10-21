---
name: comfy-run-on-cloud
description: Use when a developer clicked "Run on Cloud" in local ComfyUI or Comfy Desktop, pasted a Run-on-Cloud prompt, or wants a local workflow (or the whole install behind it) running on Comfy Cloud with a link to open it. Also use for a requested update to an existing build, or a pasted repair prompt after such a build or deployment failed.
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

The local modal is deliberately simple: **Run on Comfy Cloud**, “We’ll start your local agent to get this workflow running on Comfy Cloud. You’ll approve any local file uploads along the way.”, **Let’s go**. The handoff supplies the saved workflow path, running server and available distribution/version/workspace context. Build name, GPU, new versus update and workflow versus whole-instance scope are not required modal fields.

Sending it **is** the user's yes to:

- run that workflow on Cloud, including **lossless** conversions;
- inspect and prepare that workflow, checking stock Cloud and existing workspace Builds before proposing a custom build.

If a custom build is needed, collect only the choices not already supplied: workspace, scope, new versus specifically named update, name, GPU and region. Disclose current costs/timing or unknowns and **`--min 0 --max 1`**, then obtain approval for that plan before building/deploying. Honor an explicit plan already approved in the prompt or earlier conversation; do not ask again for the same target and terms. A bare **Let’s go** does not choose a GPU or authorize a new Build.

It is **never** a yes to any of these, and each needs its own answer in chat:

- a **lossy** conversion (step 3);
- uploading any file from this machine: models, LoRAs, input images, video or audio (step 4);
- replacing, stopping or deleting an existing Build, release or deployment, except the specific update named in the user's prompt;
- a different GPU or region from the approved plan, or worker bounds other than 0/1.

Batch the decisions currently known into one short message. Honor answers already given for the named files and target; a newly discovered upload or changed target needs its own answer. A general "don't ask me anything" does not identify which files may be uploaded.

## The procedure

**1. Find the workflow and the install.**
- comfy-cli is a separate install, not bundled with Desktop. If `comfy` is missing, install it with `uv tool install comfy-cli`, then `comfy skills install --skill comfy-run-on-cloud`. Check support with `comfy build inventory --help` (the command is hidden from `build --help`); if unrecognized, upgrade it the same way it was installed.
- If the prompt names no workflow file, ask which file. Never package an install instead, and never invent a path.
- Use `comfy --json which` / `comfy --json env` when they expose a verified existing path and Desktop code/data/model roots. Older versions can report a nonexistent default as success; check paths on disk. Use the running server and Desktop config when discovery is incomplete: `comfy skills show comfy-run-on-cloud-local-install`.

**2. Inventory the workflow against stock Cloud.** Run it against the running server, **every** model root and the ComfyUI input folder:

```shell
comfy --json build inventory <workflow.json> --cloud --server http://127.0.0.1:<port> \
  --desktop --models-root <any root not in Desktop settings> --input-root <ComfyUI input folder>
```

It reports what the workflow loads (active classes, models with subfolders, `embedding:` refs, input files with the name Cloud would store each under, recorded pack versions, `implicit` loaders) and a `cloud` section: `verdict`, `blocking`, `unverified`, `lossless_removable`, and per model a `status` (`present` / `remap` / `absent`) and `weights` (`same_size` / `different_size` / `no_local_copy` / `unknown` / `not_checked`). **The report is the scope.** Resolve `unverified` before treating the workflow as covered. For each `implicit` row, read the pack source and add the weights it loads. A size match only rules out an obvious mismatch: establish model identity with a verified digest/public source before calling a local-to-Cloud model match lossless. If identity cannot be verified, disclose the uncertainty and offer the Cloud copy versus an exact-file build as a user decision.

The UI inventory is a conservative candidate list: remove disconnected branches no executable output depends on. `model_hints` are template download metadata, not evidence that a file is selected; reconcile them with the actual loader/source before including them. Resolve linked primitive/Get/Set widgets and subgraph host overrides through the local server's API export, then inventory that API graph too. Keep the original UI workflow for the Cloud canvas. If no trustworthy API export is possible, report the unresolved dependencies instead of declaring coverage.

**3. Classify every change Cloud needs, then pick the route.**

| Kind | What counts | What you do |
|---|---|---|
| **Lossless** | Only these: a `remap` of a verified identical model file into another folder; removing a `lossless_removable` node, bypassing its model passthrough where present and disconnecting its settings from the loader; a utility node swapped for its verified exact core equivalent (math, string concat, a model-only LoRA stack → chained `LoraLoaderModelOnly` at the same strengths, a context pipe → direct links); removing a node no output depends on; setting a widget that a newer pack added to the value that reproduces the old behavior. | Do it on a **copy**, without asking. List each change in the report. Preserve both model and CLIP effects for stacks that modify both. |
| **Lossy** | Anything else that changes inputs to generation: a different model, LoRA, VAE, ControlNet, upscaler or embedding, including a Cloud file with your file's name but `different_size` weights; dropping a LoRA, embedding or any node that touches pixels; a different implementation of the same job (e.g. a Nunchaku loader → stock `UNETLoader`). | **Ask** (below). Never apply one without a yes. |
| **Not convertible** | A node class Cloud lacks that does the core work, with no stock equivalent (e.g. InstantID). | Custom build. |

Then the route:

Before proposing or creating a custom build, **always check existing workspace Builds** for all required packs, versions and model files, including when a conversion is impossible or the user chooses the original dependencies. Reuse one that passes the log and smoke checks. Only when none fits, ask for new versus the specifically named update target, workflow versus whole-instance scope, and GPU/region if the prompt did not already choose them. Preserve those choices when already supplied.

| Route | When | Next |
|---|---|---|
| **A. Stock Cloud** | No blockers or unresolved checks, or only verified lossless changes. | Apply them, then **pre-flight** (below). Approve and prepare inputs in step 4, run one real test, then step 6. |
| **B. Existing Build** | A workspace Build already contains every required pack and model with suitable versions, **and its deployment log shows every pack the workflow uses loading** (no `IMPORT FAILED`, no `ModuleNotFoundError`). Check `comfy --json build ls`, `build show --id`, `deploy logs`. | Approve input uploads in step 4 before any `deploy run --asset-root`. Reuse its deployment (`deploy start` if stopped), then the step-5 smoke run. Step 6. |
| **Ask** | At least one lossy change, and the user has not chosen. | Ask (below), then A or C. |
| **C. Custom build** | Anything not convertible, or the user chose "exact". | Steps 4–6. |

**Asking about a lossy conversion.** One message, in this shape:

1. **What changes:** one line per swap, `original → Cloud replacement`, and what the user will see (a different face, a different style, softer or crisper detail). Offer replacements from the same family only (SD1.5 / SDXL / Flux / Wan version), taken from the loader's `choices` in `comfy --json nodes show <Loader> --where cloud`. If you can't tell the family, offer only option 2.
2. **The options:**
   - **Convert:** runs on stock Cloud now, pays per run, nothing to build. Output will not match the local run.
   - **Keep the original models and graph (custom build):** preserves the workflow's dependencies; runtime and hardware differences can still affect results. Disclose build, storage and GPU charges from current information, build time and cold-start estimate or "unknown", and the workspace's release usage. Check whether deployment Open UI is available; otherwise explain the API handoff (`comfy deploy run`) before the user chooses.
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

A `shape_mismatch` can indicate shifted widget layout or a malformed value. Compare the saved pack's schema/source with Cloud's schema before changing a widget. Set it by name with `comfy --json workflow set-widget <copy> <node>.<widget> <value> --where cloud` only when the value reproduces the saved behavior. A Cloud default alone is not evidence of equivalence. If equivalence is unknown, ask.

**Workspace limits** (routes B and C, and before asking; see the pricing page): releases (10 per workspace observed; count with `build release ls --id` per Build), active deployments (5; `deploy ls --workspace`, stopped ones don't count), and Builds (10 on Pro; `build_limit` on push). If one is full, ask which release, Build or deployment to remove, ones with no live deployment first. Never pick for the user.

**4. Uploads and the custom-build spec** (route C, or any route that uploads).

For "just this workflow", write the spec workflow-scoped (for "whole instance", see Build scope in step 3):

```shell
comfy --json build init <dir> --name <name> --from-workflow <file> --comfy-version <local version>
```

**For a named update**, first read `build show --id <approved-build-id>` and the intended deployment. In a separate build folder, initialize a staging spec, then `comfy build pull <spec> --id <approved-build-id> --yes` to bind the existing id and `syncedRevision`. Apply the approved workflow-scoped changes to that pulled spec; preserve unrelated existing dependencies unless removal was approved. Verify its id before pushing and the returned id afterward. A matching name does not bind a Build, and pushing a fresh `id: null` spec creates a new one. Handle a stale revision by re-reading and reconciling; do not force-overwrite another person's changes.

Then fix it by hand per `comfy-build-authoring`, from the inventory:
- **Pack pins**, in order: the locally installed version; a pin a workspace Build already deployed cleanly; the workflow's recorded `ver`; the newest registry version that is not Flagged (all Flagged → `repository` + `gitRef`).
- **A pack that imports a Python package nothing installs cannot be built.** `pipDependencies` only forces versions of packages something already requires; it never adds one. ComfyUI-nunchaku is the known case: it needs the `nunchaku` wheel, so every Nunchaku node fails to import while the deployment still reports `ready`. Check each pack's imports against its `requirements.txt` before building. If one is missing, tell the user the build cannot run those nodes, and offer the lossy conversion for them instead.
- **Model `type`:** the folder under `models/` plus the widget's subfolder (e.g. `checkpoints/sdxl`).
- **Model entries:** a local file whose sha256 matches a `comfy build refs resolve` candidate becomes a `sourceUri` fetch. A local file with no public match is an **upload**: `{type, filename, localPath, sha256, sizeBytes, source: local}`.

Then `comfy build validate` and `comfy build push --dry-run`. Newer CLI versions check public registry/repository pins with a read-only builder request while uploading and saving nothing. Older versions skip that check; verify the chosen registry version/status separately before the real push. A Flagged version must not be selected.

**Skip inputs Cloud already has.** Cloud names an uploaded input by the blake3 of its bytes, and the inventory computes that name. If an input's `on_cloud` is `true`, set the loader to its `cloud_name` and don't ask. `false` only means the loader's list doesn't show it; the lists can omit uploads.

**The upload question** lists one line per file, for models, LoRAs and **input files** (images, video, audio) alike: name, size, why it must come from this machine, destination and workspace (private build storage, or the Cloud input library for inputs), estimated upload duration or "unknown", and applicable upload/storage/deployment costs or "unknown". Use **Upload models** / **Cancel upload** for models and **Upload assets** / **Cancel upload** for inputs. Declining a required file leaves this route blocked; preserve the original workflow and describe what cannot run. Give known licenses and NSFW flags (for example Hugging Face `cardData.license`); mark missing metadata unknown. Before `init --from-workflow`, disclose that it sends workflow JSON, including prompts, to the builder. The Run on Cloud handoff permits sending that workflow; it does not permit sending unrelated files or embedded secrets.

**5. Build, deploy, verify** (route C).
- `comfy build push`, then `comfy build release create --target linux/nvidia --watch`. On failure read `comfy-build-failures`: one cause per cut, at most three cuts.
- `comfy deploy up <spec> --gpu <gpuClass> --region <region> --min 0 --max 1 --watch`, with the GPU and region approved in the agent conversation or an explicit prior plan. For an update, select the approved deployment explicitly with `--deployment <id>`. Check its `revision`/update capability first: updates-enabled workspaces move it in place; older workspaces can create a second deployment and keep the first billing. Disclose and obtain that additional-deployment decision before proceeding. Read returned `supersedes` and verify target ids; stopping an older deployment still needs the user's approval. On a refusal, say so and ask; never switch GPUs on your own.
- Report the idle scale-down from the deployment's own `keepWarmSeconds`, not a remembered default. Outside a Build folder, read a deployment with `comfy --json deploy show --deployment <build>/<name>`; `deploy status` needs the spec folder.
- **`ready` is not working.** Require both:
  - the ComfyUI log (`comfy --json deploy logs` → `data.comfyuiLog`) shows no `IMPORT FAILED` and no `ModuleNotFoundError` for any pack the workflow uses;
  - one smoke run against the selected deployment. Build the API file from node definitions that include the build's packs: `comfy run --workflow <file> --print-prompt --where local` against a running local ComfyUI that has them (the graph is `data.prompt`). Converting `--where cloud` can drop widgets of a class stock Cloud lacks. With no such server, inspect the pack's `INPUT_TYPES`; do not guess missing inputs.
- **Reconcile the complete asset plan before any deployment run.** From the same build folder, run `comfy --json build inventory <api.json> --input-root <dir with inputs>` (repeat roots as needed). Its `deployment_assets` and `deployment_asset_roots` come from the exact deployment scanner, including custom-node input strings and automatic model/input/output roots. Match every planned local path to its per-file approval; files found here but absent from the earlier UI inventory need approval too. Use the same API file, cwd and root list for `comfy deploy run <spec> --deployment <id> --workflow <api.json> --asset-root <dir>`. Until the plan is reconciled, keep `--no-upload`; this flag prevents file uploads but can still submit a job if assets already exist, so it is not a free dry run. If a file changes after approval, review its new content/size before sending it.
- A smoke run on the build can fail validation where Cloud passed (a pack version requires an input Cloud's version lacks, e.g. USDU's `batch_size`). Add the input with the default that reproduces the saved behavior and re-run.

**6. Save, then link.**
- Upload input files only with the step-4 yes; set each loader to the returned `cloud_name` on the copy, then pre-flight and test. Cloud's dropdowns can omit uploads; verify the actual load in the test job.
- **Test every route before claiming it runs.** Route A: `comfy --json run --workflow <copy> --where cloud --wait`, then `comfy --json download <prompt_id>`. Routes B/C: the step-5 smoke run. If a required input or approval is missing, report "prepared, not run" with the exact blocker.
- Save the workflow (the converted copy on route A) so it shows in the Workflows sidebar. Use `comfy --json workflow save <copy> --name <name> --no-overwrite --where cloud` when its version writes the sidebar userdata file and verifies the save. A name collision needs a different name or explicit permission to overwrite that saved workflow. Older versions create only a record: use the Comfy Cloud connector's inline `save_workflow` if available, or return the local file and the CLI limitation. The skill must work without that connector. Confirm read-back/listing; with `list_saved_workflows`, a usable save has a size and is not `filename_ambiguous`.
- Return an authenticated workflow/deployment Open UI URL only when the target provides one and access/readiness is verified. Today private saved-workflow deep links and deployment Open UI are separate platform gaps; `#<workflow_id>` does not open a private saved workflow. Fallback: "https://cloud.comfy.org → Workflows → `<name>`" for stock Cloud. For B/C, label any stock-Cloud save as an archive: it does not select the custom environment, and can show missing nodes. Return verified deployment ids/API commands and state the missing UI handoff. Offer a public share link only with permission to publish.

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
