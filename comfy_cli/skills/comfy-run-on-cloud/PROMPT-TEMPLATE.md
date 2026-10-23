# The Run on Cloud handoff

Modal title: **Run on Comfy Cloud**

Body: **We’ll start your local agent to get this workflow running on Comfy Cloud. You’ll approve any local file uploads along the way.**

CTA: **Let’s go**

Let’s go sends this prompt to the integrated local agent. **Copy prompt** supplies the same handoff for Claude or Codex when integration is unavailable; keep **Open in platform** as a fallback. Do not put build name, GPU, new/update or scope fields in this modal. Ask build choices in the agent conversation only when a custom Build is needed; if reusing a Build needs a new deployment, ask only the missing deployment choices and approval.

Save the canvas to a workflow JSON file first. The **Workflow** path is required and must be real. Fill known context from the running app; omit unknown optional lines rather than guessing. If the local agent cannot read that path, resolve it before proceeding.

```text
Run my ComfyUI workflow on Comfy Cloud.

Read the skill first and follow it:
  comfy skills show comfy-run-on-cloud
(If comfy is missing: uv tool install comfy-cli — it is separate from ComfyUI Desktop.)

Workflow: {absolute path to the saved workflow JSON}
ComfyUI: {running local server URL} · {Desktop | source | portable} {version}
Workspace: {workspace name and id, if known}

Check stock Cloud with my original dependencies, including supported bring-your-own
models/LoRAs. If it cannot run the original, check compatible existing workspace Builds
before considering conversion. Reuse a compatible Build without changing my graph.
Only if none fits, make verified lossless changes on a copy, or explain quality-changing
node/model substitutions and ask: convert, or keep the original with a custom Build.
An approved conversion goes to stock Cloud; a declined conversion or non-convertible
workflow goes to custom-build choices. Preserve my original workflow.

If a custom build is needed, ask for any choices I have not already supplied: just this
workflow versus whole instance, new versus specifically named update, build name, GPU and
region. Suggest suitable GPUs from cheapest to most expensive using current information.
Explain current costs/timing or unknowns and 0–1 worker bounds before I approve it.
If reusing a Build needs a new deployment, ask only its missing target/release, name,
GPU/region and cost choices; reuse the Build without rebuilding it.
Ask before uploading every local model or input asset, identifying each file, size,
purpose, destination/workspace, estimated time and costs or explicit unknowns.

Approval for uploads applies to every route, including existing-Build reuse. If I decline
a required file, do not upload or substitute it; report the route as prepared, not run.
Pre-flight converted workflows and verify a real job on the selected target with its
required approved inputs before saying it runs. Return a verified authenticated UI link
only when the target supports it; otherwise explain the Workflows-sidebar or selected
deployment API handoff. A stock-Cloud save does not select a Build's environment.
```

If the user already explicitly chose a plan, append only those choices as **Previously approved choices**, including the exact Build/deployment IDs for an update. Do not derive approval from the bare modal click or silently fill defaults. Honor existing answers; newly discovered files or a changed target need an answer.

The prompt carries the workflow and its prompts to the requested service as part of the handoff. It does not authorize sending unrelated files or embedded secrets. A named-file upload approval is still required on every route.
