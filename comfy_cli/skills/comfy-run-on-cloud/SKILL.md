---
name: comfy-run-on-cloud
description: Use when a developer clicked "Run on Cloud" in local ComfyUI or Comfy Desktop, pasted a Run-on-Cloud prompt, or wants a local workflow (or the whole install behind it) running on Comfy Cloud with a link to open it. Also use for a requested update to an existing build, or a pasted repair prompt after such a build or deployment failed.
---

# comfy-run-on-cloud

One workflow in, one link out. Users care that the workflow runs on Cloud, not how.

Comfy Cloud has two kinds of environment. Use these names with the user, never "stock Cloud":

- **Comfy Cloud – standard environment:** the shared node and model catalog every account gets. Pay per run; nothing to build or deploy.
- **Comfy Cloud – custom environment:** a deployment of a workspace Build, with its own packs and models. Build, release and deployment are the CLI's nouns for how one is made.

So:

- Use the **standard environment** with the original dependencies when supported, including bringing the user's own models/LoRAs only through a verified supported import path and with upload approval.
- Otherwise, reuse a compatible **existing custom environment** (a workspace Build) **before considering conversion**, preserving the original graph and dependencies.
- With no compatible Build, **convert** for the standard environment: automatically for verified lossless changes, or with an informed yes for lossy changes.
- Make a new **custom environment** (or explicitly update one) only when conversion is impossible or the user chooses to keep the original dependencies.

Whichever route you take:

- Package **only what the workflow loads**, unless the user chooses the whole instance for a custom build.
- **Never change what the workflow generates without a yes.** A lossless fix is yours to make; a lossy one is the user's call.
- Never upload a file from disk without a per-file yes.
- Hand back the link only once the target can run.

Run this orchestration in the **local agent**: inspect local custom-node source, Git origins, file paths and model hashes as needed. Reuse the agent's existing sign-in flow if the user is not signed in. When a local workflow is slow, memory-constrained or would benefit from batch/parallel runs, suggest Cloud with that concrete benefit; wait for the user's choice before starting the handoff.

For Build and deployment operations, this skill sequences two others. Read them when a step points there; do not restate them.

- `comfy skills show comfy-build`, plus its `comfy-build-authoring`, `comfy-build-pins` and `comfy-build-failures` references.
- `comfy skills show comfy-deploy`, plus `comfy-deploy-failures`.

## What the click authorizes, and what it does not

The local modal is deliberately simple: **Run on Comfy Cloud**, “We’ll start your local agent to get this workflow running on Comfy Cloud. You’ll approve any local file uploads along the way.”, **Let’s go**. The handoff supplies the saved workflow path, running server and available distribution/version/workspace context. Build name, GPU, new versus update and workflow versus whole-instance scope are not required modal fields.

Sending it **is** the user's yes to:

- run that workflow on Cloud, including **lossless** conversions;
- inspect and prepare that workflow, checking the standard environment and then compatible existing custom environments before considering conversion or proposing a custom build.

If a custom build is needed, collect only the choices not already supplied: workspace, scope, new versus specifically named update, name, GPU and region. Disclose current costs/timing or unknowns and **`--min 0 --max 1`**, then obtain approval for that plan before building/deploying. Honor an explicit plan already approved in the prompt or earlier conversation; do not ask again for the same target and terms. A bare **Let’s go** does not choose a GPU or authorize a new Build.

If reusing a Build requires a new deployment, ask only the missing deployment choices (target/release, name, GPU/region), costs and 0/1 bounds. Obtain that deployment approval without reopening Build creation or scope choices.

It is **never** a yes to any of these, and each needs its own answer in chat:

- a **lossy** conversion (step 3);
- uploading any file from this machine: models, LoRAs, input images/video/audio, or local custom-node source bundles (step 4);
- replacing, stopping or deleting an existing Build, release or deployment, except the specific update named in the user's prompt;
- a different GPU or region from the approved plan, or worker bounds other than 0/1.

Batch the decisions currently known into one short message. Honor answers already given for the named files and target; a newly discovered file, changed file bytes or changed destination/workspace needs its own answer. A general "don't ask me anything" does not identify which files may be uploaded.

## The procedure

**1. Find the workflow and the install.**
- comfy-cli is a separate install, not bundled with Desktop. If `comfy` is missing, install it with `uv tool install comfy-cli`. Check `comfy --version`, `comfy skills show comfy-run-on-cloud` and `comfy build inventory --help` (inventory is hidden from `build --help`) before installing the skill. If unsupported, upgrade once through the existing package manager and recheck. If still unavailable, report the installed version and missing feature, preserve the workflow and stop this handoff. A release may predate this skill; never loop upgrades, silently install an unreviewed Git checkout, or treat a manual guess as the missing helper. A separately approved, reviewed pinned distribution is another installation choice.
- If the prompt names no workflow file, ask which file. Never package an install instead, and never invent a path.
- Use `comfy --json which` / `comfy --json env` when they expose a verified existing path and Desktop code/data/model roots. Older versions can report a nonexistent default as success; check paths on disk. Use the running server and Desktop config when discovery is incomplete: `comfy skills show comfy-run-on-cloud-local-install`.

**2. Inventory the workflow against the standard environment.** Run it against the running server, **every** model root and the ComfyUI input folder:

```shell
comfy --json build inventory <workflow.json> --cloud --server http://127.0.0.1:<port> \
  --desktop --models-root <any root not in Desktop settings> --input-root <ComfyUI input folder>
```

It reports what the workflow loads (active classes, models with subfolders, `embedding:` refs, input files with the name Cloud would store each under, recorded pack versions, `implicit` loaders) and a `cloud` section: `verdict`, `blocking`, `unverified`, `lossless_removable`, and per model a `status` (`present` / `remap` / `absent`) and `weights` (`same_size` / `different_size` / `no_local_copy` / `unknown` / `not_checked`). **The report is the scope.** Resolve `unverified` before treating the workflow as covered. For each `implicit` row, read the pack source and add the weights it loads. A size match only rules out an obvious mismatch: establish model identity with a verified digest/public source before calling a local-to-Cloud model match lossless. Compare local SHA256 with verified public candidates from `comfy --json build refs resolve <filename>`, then establish that the selected target serves those same bytes/source. `asset.hash` and `metadata.source_hash` are not file SHA256 unless their response explicitly identifies that algorithm. A resolver candidate alone does not establish target availability. If identity cannot be verified, retain that unresolved check for step 3 and ask before selecting the uncertain Cloud copy.

An `absent` model/LoRA is missing from the catalog, not proof that the standard environment cannot run the original. Discover the supported model import/upload path and its type, workspace and access constraints using current tools/help. Keep the original model when supported and ask before uploading it. Do not substitute a similar catalog model or require a custom Build solely because the original is not listed. Verify that destination before asking for upload approval. `comfy upload` is an input-asset upload, not evidence of model import; library deduplication is not evidence that arbitrary model bytes can be imported. If the available tools cannot establish a supported model path, report the limitation and continue to compatible Builds; do not offer an unsupported upload, invent a command or claim coverage.

**Supported standard-environment model import:** when `comfy model upload --help` exposes it, discover the exact folder with `comfy --json model list-folders --where cloud`, then preview `comfy --json model upload <file> --type <advertised-folder> --workspace <workspace-id> --where cloud --dry-run`. The preview checks the current workspace, private-model entitlement, advertised model folder and destination-name collisions; it uploads nothing. Use its exact path/size/SHA256/destination in the step-4 question, supplementing known license/NSFW metadata rather than replacing known values with unknown.

After that named-file approval, execute the same command with `--yes --expected-sha256 <approved-sha256>` instead of `--dry-run`. An approved destination filename can be supplied with `--name <name>`; a collision or changed name/target needs review. The command uploads an immutable snapshot matching the approved SHA256 and verifies the returned content hash/loader path. Verify workspace and returned asset identity, put the returned `loader_path` on a workflow copy, then pre-flight and test that exact model. A refusal is a blocker to report, not permission to omit approval flags or select different weights. On older CLI versions or unentitled accounts, inspect enabled tools for an equivalent verified capability; if none exists, continue to the existing-Build check.

The UI inventory is a conservative candidate list: remove disconnected branches no executable output depends on. `model_hints` are template download metadata, not evidence that a file is selected; reconcile them with the actual loader/source before including them. Resolve linked primitive/Get/Set widgets and subgraph host overrides through the local server's API export, then inventory that API graph too. Keep the original UI workflow for the Cloud canvas. If no trustworthy API export is possible, report the unresolved dependencies instead of declaring coverage.

**3. Pick the route in this order; classify conversions only after the reuse check.**

1. **Standard environment, original workflow:** check whether every executable dependency is supported, including the original models/LoRAs through a verified bring-your-own path. Only an unchanged original with resolved identities qualifies here. `covered_after_lossless_fixes`, a required remap or a removable compatibility node goes through the existing-Build check before applying that fix: a lossless fix is still a conversion. Resolve unknowns; if the unchanged original is supported, go to route A.
2. **Existing custom environment:** if the standard environment cannot run the original, check workspace Builds for the original packs, versions and model identities **before converting anything or asking about a lossy swap**. Inspect `comfy --json build ls`, `build show --id`, `comfy --json deploy show --deployment <id>` and `comfy --json build release manifest <served-release-id>`. Compare required pack versions, model type/name and verified SHA256 against the release actually served, plus the deployment logs; the latest undeployed release and a Build blobId are insufficient. A differing version is compatible only when its source/schema establishes the same required behavior; otherwise explain the difference and ask before selecting it. Node-class names alone do not establish equivalence. A verified match takes route B; keep the graph and dependencies. Approve any required local uploads and run a smoke job on that deployment before claiming success. If the Build has no usable deployment, ask only the missing deployment choices/approval and reuse the Build; do not create a new Build merely to deploy it.
3. **Conversion:** only when no compatible existing Build fits, classify the proposed changes below. All verified lossless changes → apply on a copy and take route A. Any lossy change → explain the substitutions and ask; approval → route A, declining conversion/choosing exact dependencies → route C. No viable standard-environment equivalent → route C.

Honor an explicit route or named update already chosen by the user. Read-only checks do not authorize changing that target.

| Kind | What counts | What you do |
|---|---|---|
| **Lossless** | Only these: a `remap` of a verified identical model file into another folder; removing a `lossless_removable` node, bypassing its model passthrough where present and disconnecting its settings from the loader; a utility node swapped for its verified exact core equivalent (math, string concat, a model-only LoRA stack → chained `LoraLoaderModelOnly` at the same strengths, a context pipe → direct links); removing a node no output depends on; setting a widget that a newer pack added to the value that reproduces the old behavior. | On the conversion route, do it on a **copy**, without asking. List each change in the report. Preserve both model and CLIP effects for stacks that modify both. |
| **Lossy** | Anything else that changes inputs to generation: a different model, LoRA, VAE, ControlNet, upscaler or embedding, including a Cloud file with your file's name but `different_size` weights; dropping a LoRA, embedding or any node that touches pixels; a different implementation of the same job (e.g. a Nunchaku loader → core `UNETLoader`). | **Ask** (below). Never apply one without a yes. |
| **Not convertible** | A node class Cloud lacks that does the core work, with no standard-environment equivalent (e.g. InstantID). | Custom environment (route C). |

| Route | When | Next |
|---|---|---|
| **A. Standard environment** | Original dependencies supported, or no compatible Build fits and all needed conversions are verified lossless or explicitly approved as lossy. No blockers or unresolved checks remain. | Approve any required model/LoRA/input uploads in step 4. Pre-flight the runnable copy after assets are prepared, then run a real Cloud job and save/handoff in step 6. |
| **B. Existing custom environment** | The original dependency check found a compatible Build. Verify the selected deployment's required-pack imports before running; if a deployment must be created, check its logs after it is ready. | Keep the original graph. Use step 4's upload gate for any needed local models/assets; skip custom-build spec creation. Reuse its deployment (`deploy start` if stopped), perform the shared step-5 checks/smoke run, then hand off to that environment in step 6. |
| **Ask** | No compatible Build fits, and at least one proposed change is lossy or model identity remains uncertain. | Explain the difference and ask (below), then A or C. |
| **C. New or updated custom environment** | No compatible Build fits and conversion is impossible, or the user declined conversion/chose exact dependencies or a named update. | Ask only missing new/update, scope and GPU/region choices; then steps 4–6. Recommend GPUs that fit the model/workflow, ordered cheapest to most expensive using current prices and availability. Preserve choices already supplied. |

**Unverified model identity uses the same informed two-option question.** After checking supported original-model import and existing Builds, give both named exits: **Use the unverified catalog copy** (name it and explain that its weights/output may differ) / **Keep my exact local weights** (a verified supported import or a buildable custom environment, with its plan/cost/upload choices). If preserving the original is currently blocked, name that blocker rather than promising it works. A yes/no question about the catalog copy alone omits the required original-dependency alternative.

**Asking about a lossy conversion.** One message, in this shape:

1. **What changes:** one line per swap, `original → Cloud replacement`, naming the affected nodes/models and expected behavior/quality differences or uncertainty. A visual before/after is not required. Offer replacement models from the same family only (SD1.5 / SDXL / Flux / Wan version), taken from the loader's complete `choices` in `comfy --json nodes show <Loader> --where cloud --all-choices` (check help on older versions). Read truncation metadata; the first page is not the complete catalog. If you can't tell the family, offer only option 2.
2. **The options:**
   - **Convert:** uses the standard environment after required files and validation, pays per run, nothing to build. Output may differ from the local run. Apply the substitutions only after a yes, then pre-flight and test.
   - **Keep the original models and graph (custom environment):** preserves the workflow's dependencies; runtime and hardware differences can still affect results. Disclose build, storage and GPU charges from current information, build time and cold-start estimate or "unknown", and the workspace's release usage. Check whether deployment Open UI is available; otherwise explain the API handoff (`comfy deploy run`) before the user chooses.
3. The **build scope** question (below), for option 2.
4. Any **upload** questions from step 4, in the same message.

Recommend one option in a clause, then wait. An approved conversion continues to route A; a declined conversion/choice to keep the original continues to route C's build choices, not to the conversion step. If both are declined, stop and report the blocker. A user note such as "demo in 20 minutes" informs the recommendation; it does not choose for the user.

**Check buildability before offering the exact-dependency option.** Inspect required packs' imports and declared requirements against the supported builder environment. Packages already declared in requirements are installed by the builder; `pipDependencies` only pins packages otherwise installed. If a required wheel/import has no supported installation path (for example `nunchaku`), explain that blocker before recommending or approving a Build. Offer a viable informed lossy alternative if one exists; removing a core task node does not preserve the task. If neither exact nor converted execution is viable, report the blocker.

**Verify workspace selection:** read `comfy --json cloud status` and compare `cloud_workspace.id` with the requested/approved workspace before uploads or custom-environment mutations. `deploy ls --workspace` is a boolean listing flag for the authenticated workspace; it does not select a workspace by name or ID. If they differ, use a supported explicit workspace-selection flow or ask how to switch; do not silently operate in the current one.

**Current custom-environment prices:** use `comfy --json deploy refs compute` and current workspace data before the plan or lossy/custom choice. List suitable GPUs cheapest first, distinguishing available from requestable capacity. Disclose build/storage/compute charges, build/cold-start estimates or explicit unknowns. For reuse, read the selected deployment's `warmHourlyCents`, `storageMonthlyCents` and `computeConfig.keepWarmSeconds`; use unknown only when the current surface omits a value.

**Build scope: ask it whenever a custom build is on the table** (option 2 above, or route C). Same message:

- **Just this workflow** (the default; recommend it unless the user wants more): only the packs and models the inventory lists. Smaller, faster to build, and fewer files to upload.
- **The whole instance:** every custom node pack in the install and every model in its models folders, so any workflow from this install runs on the deployment. State the real size before they choose: the pack count (`ls <data>/custom_nodes`), the model count and total GB per models root (`du -sh`), and how many of those files have no public source and would need an upload. Build time, storage cost and the chance of a dependency conflict all grow with it. List a shared models root (a Desktop `modelsDirs` entry outside the data dir) separately; it holds weights for other work and is never included without its own yes.

On "whole instance", write the spec from the install per `comfy-build` Path A (`comfy build init <install> --name <name> --python <verified instance interpreter>`) instead of `--from-workflow`, then prune it per `comfy-build-authoring`. Uploads are still gated: show the full list of files with no public source, and a yes to that listed set counts as the per-file yes. A file added later is not covered.

Before you send it, **check the workspace limits** (below), so the custom-build option says whether a slot must be freed first. **A swap only works where its replacement exists:** the standard environment's models are not inside a custom environment, and a Build's models are not in the standard environment. Never offer "swap on the existing Build" unless that Build lists the replacement.

**Pre-flight every converted copy on the server** before saving or running it. The inventory cannot see settings a pack version shifted:

```shell
comfy --json run --workflow <copy> --print-prompt --where cloud > preview.json
jq '.data.prompt' preview.json > api.json
comfy --json workflow validate --workflow api.json --where cloud
```

A `shape_mismatch` can indicate shifted widget layout or a malformed value. Compare the saved pack's schema/source with Cloud's schema before changing a widget. Set it by name with `comfy --json workflow set-widget <copy> <node>.<widget> <value> --where cloud` only when the value reproduces the saved behavior. A Cloud default alone is not evidence of equivalence. If equivalence is unknown, ask.

**Workspace limits** (before proposing an operation that consumes a slot; see the pricing page): releases (10 per workspace observed; count with `build release ls --id` per Build), active deployments (5; `deploy ls --workspace`, stopped ones don't count), and Builds (10 on Pro; `build_limit` on push). Reusing a running deployment needs no new Build/release/deployment slot; creating or restarting a deployment needs no new Build/release slot. Check current limits rather than treating these observed counts as guarantees. If the planned operation is blocked, fetch the applicable resource lists and deployment references, show a short list with names/IDs and live-use status (unused resources first), then ask which specific resource to stop/delete. Never pick or mutate one before that answer. No cleanup is needed merely because a full workspace can still reuse a running target.

**4. Approve uploads on every route; prepare a custom-build spec only for route C.**

Routes A/B use the shared upload question below for any required local file, including models/LoRAs and input assets. They skip spec initialization, pinning and pushing. After the required files are approved and prepared for the selected target, route A goes to step 6; route B goes to the shared verification in step 5.

**Route C only:** for "just this workflow", write the spec workflow-scoped (for "whole instance", see Build scope in step 3):

```shell
comfy --json build init <dir> --name <name> --from-workflow <file> --comfy-version <local version>
```

**For a named update**, first read `build show --id <approved-build-id>` and the intended deployment. In a separate build folder, initialize a staging spec, then `comfy build pull <spec> --id <approved-build-id> --yes` to bind the existing id and `syncedRevision`. Apply the approved workflow-scoped changes to that pulled spec; preserve unrelated existing dependencies unless removal was approved. Verify its id before pushing and the returned id afterward. A matching name does not bind a Build, and pushing a fresh `id: null` spec creates a new one. Handle a stale revision by re-reading and reconciling; do not force-overwrite another person's changes.

Then fix it by hand per `comfy-build-authoring`, from the inventory:
- **Pack pins**, in order: the locally installed version; a pin a workspace Build already deployed cleanly; the workflow's recorded `ver`; the newest registry version that is not Flagged (all Flagged → `repository` + `gitRef`).
- Recheck the step-3 buildability decision after selecting pins. A missing required import blocks the build plan until a supported installation path or informed viable conversion is selected.
- **Model `type`:** the folder under `models/` plus the widget's subfolder (e.g. `checkpoints/sdxl`).
- **Model entries:** a local file whose SHA256 matches a verified `comfy build refs resolve` candidate becomes a disclosed `sourceUri` fetch. A local file with no public match is an **upload**: `{type, filename, localPath, sha256, sizeBytes, source: local}`. Model `localPath` is relative to `<spec>/models/` (for `<spec>/models/loras/private.safetensors`, use `loras/private.safetensors`); local-node `localPath` is relative to `<spec>/custom_nodes/` (use `my-pack` for `<spec>/custom_nodes/my-pack`). Validate these files before push.

Then `comfy build validate` and `comfy build push --dry-run`. Newer CLI versions check public registry/repository pins with a read-only builder request while uploading and saving nothing. Older versions skip that check; verify the chosen registry version/status separately before the real push. A Flagged version must not be selected.

**Skip inputs the selected target already has.** For the standard environment, uploaded inputs are named by the blake3 of their bytes, and the inventory computes that name. If an input's `on_cloud` is `true`, set the loader to its `cloud_name` and don't ask. `false` only means the loader's list doesn't show it; the lists can omit uploads. A standard-environment asset listing does not establish availability in a custom environment; verify the exact file on that target before skipping an upload.

**The upload question** lists one line per file, for models, LoRAs, **input files** (images, video, audio) and **local custom-node source bundles** alike: name, size, why it must come from this machine, destination and workspace (the selected target's model storage or input library), estimated upload duration or "unknown", and applicable upload/storage/deployment costs or "unknown". Use **Upload models** / **Cancel upload** for models and **Upload assets** / **Cancel upload** for inputs/source bundles. For a source bundle, list its included files, exclusions and resulting size; exclude secrets, caches and unrelated files before requesting approval. Keep an approval ledger of exact path, size, digest, destination/workspace and decision. Immediately before upload, recheck those bytes and target; changed content, a new file or a changed destination needs a new answer. An explicitly listed set can be approved together. These checks reduce races; prose is not an atomic upload lock. Declining a required file leaves this route blocked; preserve the original workflow and report "prepared, not run" with what cannot run. Do not upload it or silently substitute another file. Give known licenses and NSFW flags (for example Hugging Face `cardData.license`); mark missing metadata unknown. Before `init --from-workflow`, disclose that it sends workflow JSON, including prompts, to the builder. The Run on Cloud handoff permits sending that workflow; it does not permit sending unrelated files or embedded secrets.

Use this question shape, filling every field (use `unknown` for unavailable metadata):

```text
Prepared, not run: waiting for the required file approval.
<absolute path> | <size> | SHA256 <digest> | <purpose> | <destination and workspace>
Upload time: <estimate/unknown>; upload/storage/compute cost: <values/unknown>.
License: <value/unknown>; NSFW: <true/false/unknown>.
Upload assets / Cancel upload: approve these exact bytes to this destination?
```

Use `Upload models / Cancel upload` for models. Repeat the file line and metadata for each file in a listed set. Include source-bundle contents/exclusions when applicable. A changed-file question uses the same complete shape, plus the old/new size or digest; an earlier question does not supply metadata for changed bytes.

**5. Build/deploy only when needed; verify routes B/C on the selected deployment.**
- **Route C only:** `comfy build push`, then `comfy build release create --target linux/nvidia --watch`. On failure read `comfy-build-failures`: one cause per cut, at most three cuts. Before another missing-import cut, verify the exact supported mechanism that installs the missing distribution. A constraint/pin alone does not install it. If no declared requirement, supported wheel or other verified installation mechanism exists, stop and report the blocker; prior repair approval does not authorize a speculative paid cut.
- **Route C:** `comfy deploy up <spec> --gpu <gpuClass> --region <region> --min 0 --max 1 --watch`, with the GPU and region approved in the agent conversation or an explicit prior plan. For an update, select the approved deployment explicitly with `--deployment <id>`. Check its `revision`/update capability first: updates-enabled workspaces move it in place; older workspaces can create a second deployment and keep the first billing. Disclose and obtain that additional-deployment decision before proceeding. Read returned `supersedes` and verify target ids plus the actual served release after the update; stopping an older deployment still needs the user's approval. On a refusal, say so and ask; never switch GPUs on your own.
- **Route B without a deployment:** read `comfy-deploy`, verify a compatible deployable release of the selected Build, and use `comfy deploy up --release <verified-release-id>` with the approved deployment choices. Do not push a spec or create another Build/release to reuse an already deployable release.
- **Both B/C:** the remaining checks apply to the selected deployment, including reused deployments. Route B skips build push/release creation and leaves the existing Build's dependencies unchanged. Without a local spec, address it with the verified `--deployment <id>` on show/logs/start/run/stop rather than initializing a new spec.
- Report the idle scale-down from the deployment's own `keepWarmSeconds`, not a remembered default. Outside a Build folder, read a deployment with `comfy --json deploy show --deployment <build>/<name>`; `deploy status` needs the spec folder.
- **`ready` is not working.** Require both:
  - the ComfyUI log (`comfy --json deploy logs` → `data.comfyuiLog`) shows no `IMPORT FAILED` and no `ModuleNotFoundError` for any pack the workflow uses;
  - one smoke run against the selected deployment. Build the API file from node definitions that include the build's packs: `comfy --json run --workflow <file> --print-prompt --where local > preview.json` against a running local ComfyUI that has them, then extract `.data.prompt` to a separate `api.json`. Converting `--where cloud` can drop widgets of a class the standard environment lacks. With no such server, inspect the pack's `INPUT_TYPES`; do not guess missing inputs.
- **Reconcile the complete asset plan before any deployment run.** From the intended run directory, run `comfy --json build inventory <api.json> --input-root <dir with inputs>` (repeat roots as needed). Its `deployment_assets` and `deployment_asset_roots` come from the exact deployment scanner, including custom-node input strings and automatic model/input/output roots. Match every planned local path to its per-file approval; files found here but absent from the earlier UI inventory need approval too. Use the same API file, cwd and root list for `comfy deploy run --deployment <verified-id> --workflow <api.json> --asset-root <dir> --wait --output-dir <workflow-folder>/outputs`; no spec is needed for an explicit deployment id. `deploy run` waits/downloads by default and accepts `--wait`, while `--watch` is a release/deployment-monitor flag; check command-specific help before adding flags. Until the plan is reconciled, keep `--no-upload`; this flag prevents file uploads but can still submit a job if assets already exist, so it is not a free dry run. If a file changes after approval, review its new content/size before sending it.
- Run comfy commands sequentially. On a transient 401, retry/re-read once with `comfy --json build release show <release-id>` or `comfy --json deploy show --deployment <id>`; use `deploy status` only inside its valid spec folder. If auth still fails, distinguish persistent authentication from a job/resource failure before asking for login; never resubmit an already accepted job to repair a watch error.
- A smoke run on the build can fail validation where Cloud passed (a pack version requires an input Cloud's version lacks, e.g. USDU's `batch_size`). Add the input with the default that reproduces the saved behavior and re-run.

**6. Save, then link.**
- Upload input files only with the step-4 yes; set each loader to the returned `cloud_name` on the copy, then pre-flight and test. Cloud's dropdowns can omit uploads; verify the actual load in the test job.
- **Test every route before claiming it runs.** Route A: `comfy --json run --workflow <copy> --where cloud --wait`, then `comfy --json download <prompt_id> --where cloud --out-dir <workflow-folder>/outputs`. Routes B/C: the step-5 `deploy run --wait --output-dir ...` already waits and downloads deployment outputs; root `download` targets ordinary local/standard-environment prompt history, so do not use it with a deployment job ID. If deployment output recovery is needed, use only supported job links/tools for that deployment or report the missing recovery capability; never submit another paid job just to recover a download. Retain accepted job/prompt IDs across timeouts and resume watching the same job. Download outputs beside the workflow copy in a durable user-accessible directory, and verify returned files; a download failure is not a job failure. If a required input, approval or working dependency is missing, lead the report with **Prepared, not run** and the exact blocker. Simulated jobs must be labeled simulated.
- Check `comfy workflow save --help` for the installed save capability. Save the workflow (the converted copy on route A) so it shows in the Workflows sidebar. Use `comfy --json workflow save <copy> --name <name> --no-overwrite --where cloud` when its version writes the sidebar userdata file and verifies the save. A name collision needs a different name or explicit permission to overwrite that saved workflow. Older versions create only a record: use the Comfy Cloud connector's inline `save_workflow` if available, or return the local file and the CLI limitation. The skill must work without that connector. Confirm read-back/listing; with `list_saved_workflows`, a usable save has a size and is not `filename_ambiguous`.
- **Prefer the normal private saved-workflow identity when verified.** Check `comfy workflow ui --help`. After a sidebar-backed save returns an actual userdata `saved_file_id`, use `comfy --json workflow ui --saved-file <saved_file_id> --workspace <verified-workspace-id> --environment standard` for route A, or `--environment custom --deployment <verified-id>` for routes B/C. This read-only producer requires the release capability `private_saved_file_links_enabled: true`, exact current OAuth workspace, and the authenticated `/api/userdata-files/<saved_file_id>` snapshot containing that exact UUID, canonical stored path and original workflow JSON text; custom links also verify the selected ready deployment and served release. The URL uses `?savedFile=<saved_file_id>` and the existing workspace/environment selectors. `workflow_id` remains a separate record ID; same-name or equal-content records do not certify a file. Legacy positional `workflow ui <workflow_id>` and `?workflow=` links are unavailable and require newly generated saved-file links. Return only the verified URL. The standard link explicitly selects the standard environment in the browser; neither kind changes compute, starts/stops deployments or creates an unsaved scratch copy.
- A new CLI does not enable this backend/frontend release capability. If the capability or `saved_file_id` is absent, false, malformed, refused, record-only or mismatched, report that limit. Older servers may save successfully without a certifiable file UUID; retain that success and do not repeat saves or approved uploads to obtain an ID. The capability gates CLI publication only; it is not a browser kill switch. Route A falls back to "https://cloud.comfy.org → Workflows → `<name>`" after verified sidebar read-back. Routes B/C may discover `comfy deploy ui --help` and use `comfy --json deploy ui --deployment <verified-id>` for a verified environment-only editor URL; label it as opening the environment, not the saved workflow. Return the retained UI/API files and verified deployment run/stop commands when the complete workflow handoff is unavailable. A standard-environment archive does not select a custom environment. A UUID or `#<workflow_id>` fragment alone is not a private workflow link. Offer a public share link only with permission to publish.

**Report:** start with the execution state (verified job succeeded, simulated verification, or prepared/not run with blocker). Include the route; each lossless change and approved lossy change; upload approvals and files actually uploaded/reused; job/prompt ID, result and durable output paths; save/read-back evidence and verified UI/sidebar/API handoff or limitation. For routes B/C include workspace, Build/release/deployment IDs, GPU, region, actual worker bounds, actual idle policy/`keepWarmSeconds`, current disclosed costs and `comfy deploy stop --deployment <verified-id>`. State unresolved checks and the next user decision. A ready deployment or saved record alone is not a successful run.

## Common mistakes

| Mistake | Instead |
|---|---|
| Converting or building because Cloud lacks a node class | Check for a compatible existing Build first; only then consider verified lossless or approved lossy conversion |
| Substituting a model/LoRA because it is absent from the catalog | Check a supported bring-your-own path and ask before uploading the original |
| Sending Build reuse straight to save/link | Approve required local uploads, check logs and run a real job on the selected deployment; hand off to that environment |
| Swapping a model because "the user said don't ask" | A lossy swap changes the output; ask once, with the custom-build alternative |
| Treating a Cloud file with your file's name as your file | Check `weights`; `different_size` is a different model, so using Cloud's is lossy |
| Saving a converted copy the inventory called covered | Pre-flight it on the server; pack versions shift widget meanings |
| Calling a deployment done at `ready`, or reusing one because its Build lists the packs | Read the import log and make one smoke run |
| Scanning the whole install without asking | Ask the build-scope question; default to just this workflow |
| Treating the modal click as upload consent | Ask per file, for models and input files |
| Running several `--watch` commands in parallel | Run comfy commands one at a time |
| Treating a 401 mid-watch as a reason to log in again | Retry once; re-read with `build release show <id>` or `deploy show --deployment <id>`; keep the accepted job ID |
