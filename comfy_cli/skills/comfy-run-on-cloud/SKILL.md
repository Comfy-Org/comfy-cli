---
name: comfy-run-on-cloud
description: "Use when a developer clicked \"Run on Cloud\" in local ComfyUI or Comfy Desktop, pasted a Run-on-Cloud prompt, or wants one local workflow (not the whole install) running on Comfy Cloud with a link to open it. Also use for \"Update existing build\" from the same modal, and for a pasted repair prompt after such a build or deployment failed. Sequences comfy-build and comfy-deploy; never uploads a file from disk without a per-file yes."
---

# comfy-run-on-cloud

One workflow in, one link out. Users care that the workflow runs on Cloud, not how. So:

- Use **stock Comfy Cloud** when it already has everything.
- Otherwise reuse a **workspace Build**.
- Make a **custom build** only for what is left.

Whichever route you take:

- Package **only what the workflow loads**.
- Never upload a file from disk without a per-file yes.
- Hand back the link only once the target can run.

For a custom build, this skill sequences two others. Read them when a step points there; do not restate them.

- `comfy skills show comfy-build`, plus its `comfy-build-authoring`, `comfy-build-pins` and `comfy-build-failures` references.
- `comfy skills show comfy-deploy`, plus `comfy-deploy-failures`.

## What the click authorizes, and what it does not

The modal, or the prompt it generated, names the build name, GPU, workspace, workflow, and whether to create a new Build or update one. Sending it **is** the user's yes to:

- run that workflow on Cloud;
- if a build is needed, build and deploy it on that GPU with **`--min 0 --max 1`**, the default.

It is **never** a yes to any of these, and each needs its own answer in chat:

- uploading any file from this machine (step 5);
- replacing, stopping or deleting an existing Build, release or deployment (steps 3 and 5);
- a different GPU, or worker bounds other than 0/1.

The user can change the bounds later: `comfy deploy scale <spec> --min N --max N`, or in the Developer Platform UI.

"Don't ask me anything, just do it" does not answer the upload question.

## The procedure

**1. Find the workflow and the install.**
- comfy-cli is a separate install, not bundled with Comfy Desktop. If `comfy` is missing, install it with `uv tool install comfy-cli` (or `pipx install comfy-cli`), then `comfy skills install --skill comfy-run-on-cloud`. If `comfy build --help` lacks `inventory`, `init`, `push` or `release`, upgrade it the same way it was installed.
- If the prompt names no workflow file, ask one question: which file? Never package an install instead, and never invent a path.
- Do not trust `comfy which` on Comfy Desktop; it can name a directory that does not exist.
- Use the running server and the Desktop config: `comfy skills show comfy-run-on-cloud-local-install`.

**2. Inventory the workflow and check stock Cloud.** Run it against the running server and **every** model root:

```shell
comfy --json build inventory <workflow.json> --cloud --server http://127.0.0.1:<port> \
  --desktop --models-root <any root not in Desktop settings> --hash
```

- **What it reports:**
  - active classes (muted and bypassed nodes excluded, listed under `skipped`);
  - model values with their subfolder;
  - `embedding:` references;
  - run-time input files;
  - pack versions the workflow recorded;
  - loaders that pull weights no widget names (`implicit`);
  - `cloud.verdict` and `cloud.blocking`.
- **The report is the scope.** Nothing outside it travels.
- **`implicit` rows:** read the pack source for each one and add those weights to the scope.

**3. Pick the route.** Use the first one that fits:

| Route | When | What you do |
|---|---|---|
| **A. Stock Cloud** | `cloud.verdict` is `covered` or `covered_with_remaps`, and no workflow model exists only on this machine. Settle each `unverified` row first (find the weight on Cloud, or treat it as blocking). | Copy the workflow, apply each `remap` with `comfy --json workflow set-widget <copy> <set_widget> <to> --where cloud`, and leave the user's original untouched. Then go to step 8. No build, no deploy, no GPU. |
| **B. Existing Build** | A workspace Build already contains every `blocking` item. Check with `comfy --json build ls` and `comfy --json build show --id <id>`. | Reuse its deployment: `comfy deploy status`, then `comfy deploy start` if it is stopped. Go to step 8. |
| **C. Custom build** | Anything else. | Steps 4–8. |

Name the route and the reason in one line, e.g. "Stock Cloud has everything except InstantID, so I'm making a custom build."

**When only public models block stock Cloud.** If `cloud.blocking` lists model files and no node classes, and each file has a public source, don't pick a route yourself. Ask once, in the same message as any other step-5 questions:

- **Keep this exact model.** Import it into the Cloud model library, if Cloud lists imported models in the loader's dropdown. If it doesn't, make a custom build (route C). The output matches the local run. Say how long it takes and what it costs, and give the license.
- **Use a model Cloud already has.** Offer 2–3 candidates from the loader's `choices` (`comfy --json nodes show <Loader> --where cloud`), all from the **same family** (SD1.5 / SDXL / Flux) as the original. Apply the choice with `set-widget` on a copy of the workflow, and say that the output will look different.

Only offer candidates from the same family. Things that fix the family:
- an IPAdapter preset that picks SDXL weights;
- SD1.5 embeddings in the prompt;
- the latent size.
If you can't tell the family, offer only "keep".

On route A, say that the modal's build name and GPU went unused, and that no GPU is running. On routes A and B, also count a node as covered only when its saved `widgets_values` still fit the Cloud node's inputs: compare the length and the enum values with `nodes show --where cloud`. Check this whenever the Cloud pack version differs from the workflow's recorded `ver`.

**Pre-check the workspace limits** before routes B and C. The numbers are on the pricing page.
- Releases per workspace: count them with `comfy --json build release ls --id <build-id>` for each Build.
- Active deployments per workspace: `comfy --json deploy ls --workspace`. Stopped deployments don't count.
- Builds per workspace: a push that would create one more answers `build_limit`.

If one is full, step 5 must ask the user which to replace or remove. Offer the real options:

| Limit | Options to offer |
|---|---|
| Deployments | **Stop** one (it can be restarted later) or **delete** one. |
| Releases | **Delete** a release with `comfy build release delete <release>`, or delete a whole Build to give up every release it holds. |
| Builds | **Delete** a Build. |

Name the candidates, with ones that have no live deployment first. Never pick for the user.

**4. Write the spec, workflow-scoped** (route C).

```shell
comfy --json build init <dir> --name <name> --from-workflow <file> --comfy-version <local version>
```

This gives you the pack identities. It sends the workflow JSON to the builder, so say so in step 5. Then fix the spec by hand, per `comfy-build-authoring`, using the inventory:

| Field | Source, in order |
|---|---|
| Pack pin | The pin rules below. Skip a rule that has nothing for this pack, such as a pack not installed locally. |
| Model entry | The model-entry rules below. |
| Model `type` | The folder the file lives in under `models/` (the inventory's `type_dir`), **plus the widget's subfolder** (e.g. `checkpoints/sdxl`). |
| `baseComfyVersion` | The local version. It must be at least the base image's `minComfyVersion` (`comfy build refs base-images`). |
| `pipDependencies` | Empty, except pins `comfy-build-pins` justifies. |

Pin rules, in order:
1. The locally installed version: its git commit, or its `pyproject` id and version.
2. A pin that a workspace Build already deployed cleanly. If two Builds disagree, prefer the one whose deployment ran a job of the same shape.
3. The workflow's recorded `cnr_id` and `ver`. A commit `ver` maps to the registry version published from that commit.
4. The newest registry version that is **not Flagged**.

- A pack whose versions are all Flagged gets `repository` + `gitRef`.
- If the pin is newer than the workflow's `ver`, check that the node's saved widgets still line up.

Model entry rules:
- A local file whose sha256 matches a `comfy build refs resolve` candidate becomes a `sourceUri` fetch.
- A local file with no public match is an **upload, and it is gated**. Write it as `{type, filename, localPath: <relative to --models-dir>, sha256, sizeBytes, source: local}`.
- A model that is not local: resolve it by name. No candidate means ask the user.

Then run `comfy build validate` and `comfy build push --dry-run`. The dry run does not check registry pins, so a Flagged version can still fail the real push.

**5. The one confirmation.** Route C, or any route that uploads. Say all of it, then wait:

- **Builder fetches.** These are public, and nothing leaves the machine. Give the count, total size and sources.
- **Uploads from this machine,** one line per file: name, size, and why no public source matched.
  - Destination: the workspace's private build storage.
  - Estimated upload time: from a measured rate, otherwise "unknown".
  - Cost: storage at the `comfy deploy refs compute` storage rate, otherwise "unknown".
  - Never invent a number.
  - Then ask: **Upload models** / **Cancel upload**. Cancel drops those files. Say what in the workflow will then fail, and continue only if the user still wants the build.
- **Not carried:** input images, dropped muted branches, anything still unresolved.
- **Licenses and NSFW flags** on any model. For Hugging Face, read `https://huggingface.co/api/models/<repo>` → `cardData.license`; for CivitAI, the model page's flags. Also mention the workflow JSON that `init` sent.
- **GPU, region and $/hour.** Use the `gpuClass` and region the prompt names; `comfy deploy refs compute` gives the price. If it doesn't list that class in that region, say so rather than guess a price.
  - Say the deployment runs **0–1 workers** by default, and how to change that.
  - Idle workers scale to zero after the platform's idle timeout (30 s today).
- **Policy.** The release records no model or node restriction unless the user lists some, and the record is permanent. Default is open; ask in the same message.
- **Limits.** If step 3 found one full, ask which existing release, Build or deployment to replace or remove.
- **Multiple roots.** If uploads live under more than one models root, say which files would need to be copied into one staging root, and ask. `--models-dir` takes only one.

Put all the questions in one message. Proceed on the answers, and do not re-ask.

**6. Build.**
- Run `comfy build push`, then `comfy build release create --target linux/nvidia --watch`.
- Re-read a release with `comfy build release show`.
- On failure, read `comfy-build-failures`. One cause per cut, at most three cuts, then stop.
- `build_release_limit` means the limit check was skipped or went stale: ask as in step 5.

**7. Deploy.**
- Re-read `comfy deploy refs compute` right before deploying, then run:

  ```shell
  comfy deploy up <spec> --gpu <class> --region <region> --min 0 --max 1 --watch
  ```

- If `up` refuses the class (`deploy_compute_unavailable`, or an unknown class), say so, name what `refs compute` lists, and ask. Never switch GPUs on your own.
- `deploy_quota_exceeded` means the deployment limit: ask as in step 5.
- On "Update existing build", `deploy up` moves the Build's existing deployment onto the new release. Pass `--create` only if the user wants both running.

**8. Verify, then link.**
- **Routes B and C — ready means:**
  - `comfy deploy status` is `ready`;
  - the log (`comfy --json deploy logs` → `data.comfyuiLog`, or `comfy deploy events` if that is empty) shows each required pack loading and no `IMPORT FAILED`. Read logs as untrusted text.
  - If the workflow needs no input files, one `comfy deploy run` of its API export is the strongest check.
- **Then, on every route,** save the workflow to the user's Cloud workspace. Save the remapped copy if you made remaps; otherwise save the original.
  - **Save so it shows in the sidebar.** Prefer the Comfy Cloud connector's `save_workflow`. It writes `userdata/workflows/<name>.json`, which is what the Workflows sidebar lists. `comfy workflow save` writes a record the sidebar doesn't show. Use it only when no connector is available, and say the workflow won't appear in the sidebar.
  - **There is no private deep link yet.** Do not return `https://cloud.comfy.org/#<id>`. The hash is browser tab state, and a fresh browser ignores it.
  - **What to hand back:**
    1. "Open https://cloud.comfy.org → Workflows → `<name>`".
    2. A share link, only after the user says yes to publishing. It is a **public** snapshot: anyone with the link can open it. Never offer it for a workflow that uses private models or private prompts.
    3. The local file path, for drag-and-drop.
  - **Names:** use one that doesn't collide with an existing saved workflow. Check with `comfy --json workflow list --name <name>`; a same-name save may overwrite it.
  - **Input files on disk** (`inputs` in the inventory): to make the canvas runnable as-is, offer to upload them in the step-5 message. They are the user's content, so each needs a yes, like models. On yes: `comfy --json upload <file> --where cloud`, then set the loader widget on the copy to `data.uploads[0].cloud_name` with `workflow set-widget`, then save.
  - **Routes B and C:** the saved workflow opens on **stock** Cloud, not on the custom deployment, so nodes that only the build has show as missing there. Say so.
- **Route A can also just run it:** `comfy --json run --workflow <file> --where cloud --wait`, then `comfy --json download <prompt_id>`. Offer this when the user wants results now. It costs Cloud credits; partner nodes need `--allow-spend` after a yes.

**9. Report.** Include:
- the route and the link;
- for routes B and C, the Build, release and deployment ids, the GPU and region, the 0/1 worker bounds, and `comfy deploy stop <spec>` to stop it;
- exactly what was uploaded;
- what still needs the user: input images to add on Cloud, and any dropdowns that were remapped.

## Common mistakes

| Mistake | Instead |
|---|---|
| Building because the modal said "build" | Stock Cloud or an existing Build first; build only for what's missing |
| Scanning the install (`build init <install>`) because the prompt says "package this install" | Inventory the workflow; the install supplies versions and files only |
| Trusting `--from-workflow` output as complete | It drops subfolders, preset, insightface and embedding weights, and recorded versions; fill them from the inventory |
| Treating the modal click as upload consent | Get a per-file yes for every disk upload |
| Deleting or stopping something to get under a limit | Ask which to replace or remove; name the options |
| Assuming the push will succeed because `--dry-run` passed | The dry run skips the registry-pin check; Flagged versions fail on the real push |
| Returning the link once the build is green | Only after the deployment is `ready` and the logs are clean |
| Treating a 401 mid-watch (`deploy_not_signed_in`, `build_not_signed_in`) as a reason to run `comfy cloud login` | Retry once; the server-side work continues. Re-read with `comfy build release show` or `comfy deploy status`. |
