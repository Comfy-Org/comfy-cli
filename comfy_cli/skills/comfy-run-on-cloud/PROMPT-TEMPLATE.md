# The Run on Cloud handoff

Modal title: **Run on Comfy Cloud**

Body: **We’ll start your local agent to get this workflow running on Comfy Cloud. You’ll approve any local file uploads along the way.**

CTA: **Let’s go**

Let’s go sends this prompt to the integrated local agent. **Copy prompt** supplies the same handoff for Claude or Codex when integration is unavailable; keep **Open in platform** as a fallback. Do not put build name, GPU, new/update or scope fields in this modal. Ask those in the agent conversation only when a custom build is needed.

Save the canvas to a workflow JSON file first. The **Workflow** path is required and must be real. Fill known context from the running app; omit unknown optional lines rather than guessing. If the local agent cannot read that path, resolve it before proceeding.

```text
Run my ComfyUI workflow on Comfy Cloud.

Read the skill first and follow it:
  comfy skills show comfy-run-on-cloud
(If comfy is missing: uv tool install comfy-cli — it is separate from ComfyUI Desktop.)

Workflow: {absolute path to the saved workflow JSON}
ComfyUI: {running local server URL} · {Desktop | source | portable} {version}
Workspace: {workspace name and id, if known}

Check stock Cloud and verified lossless conversion first, preserving my original workflow.
Ask before a lossy conversion and offer the exact-dependency alternative. Check existing
workspace Builds before proposing a custom build, even if conversion is impossible.

If a custom build is needed, ask for any choices I have not already supplied: just this
workflow versus whole instance, new versus specifically named update, build name, GPU and
region. Explain current costs/timing or unknowns and 0–1 worker bounds before I approve it.
Ask before uploading every local model or input asset, identifying each file, size,
purpose, destination/workspace, estimated time and costs or explicit unknowns.

Verify a real job with its required approved inputs before saying it runs. Return a
verified authenticated UI link only when the target supports it; otherwise explain the
Workflows-sidebar or deployment API handoff and the remaining limitation.
```

If the user already explicitly chose a plan, append only those choices as **Previously approved choices**, including the exact Build/deployment IDs for an update. Do not derive approval from the bare modal click or silently fill defaults. Honor existing answers; newly discovered files or a changed target need an answer.

The prompt carries the workflow and its prompts to the requested service as part of the handoff. It does not authorize sending unrelated files or embedded secrets. A named-file upload approval is still required on every route.
