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
(If comfy is missing: uv tool install comfy-cli — it is separate from ComfyUI Desktop.
Verify this skill and build inventory are supported. Upgrade once through the existing
package manager if needed; if still unavailable, report the version/blocker and stop.)

Workflow: {absolute path to the saved workflow JSON}
ComfyUI: {running local server URL} · {Desktop | source | portable} {version}
Workspace: {workspace name and id, if known}

Check Comfy Cloud's standard environment with my original dependencies, including
supported bring-your-own models/LoRAs. If it cannot run the original, check compatible
existing custom environments (workspace Builds) before considering conversion. Reuse a
compatible custom environment without changing my graph. An inventory verdict of
covered_after_lossless_fixes still requires this reuse check before applying any fix.
Verify model SHA256/source identity and the selected deployment's served release,
not filename/size or the newest undeployed release; ask about unresolved differences.
Only if none fits, make verified lossless changes on a copy, or explain quality-changing
node/model substitutions and ask: convert, or keep the original in a custom environment.
An approved conversion runs in the standard environment; a declined conversion or non-convertible
workflow goes to custom-build choices. Preserve my original workflow.

If a custom build is needed, ask for any choices I have not already supplied: just this
workflow versus whole instance, new versus specifically named update, build name, GPU and
region. Suggest suitable GPUs from cheapest to most expensive using current information.
Explain current costs/timing or unknowns and 0–1 worker bounds before I approve it.
If reusing a custom environment's Build needs a new deployment, ask only its missing target/release, name,
GPU/region and cost choices; reuse the Build without rebuilding it.
Check required pack installability before recommending the exact custom option.
Ask before uploading every local model, input asset or custom-node source bundle,
identifying each file, size, purpose, destination/workspace, estimated time and costs
or explicit unknowns, plus known license/NSFW metadata. Bind approval to the exact
bytes and target; changed bytes, new files or a changed destination need a new answer.
Verify a supported model-import destination before offering that upload. When supported,
preview model upload --type <advertised-folder> --workspace <id> --dry-run, then use
--yes --expected-sha256 <approved-digest> only after named-file approval; use the
verified returned loader_path on a copy. Otherwise continue to existing Builds.
If a resource cap blocks the plan, list named/ID candidates and live-use status, then
ask which specific resource to stop/delete; never choose one silently.

Approval for uploads applies to every route, including reusing a custom environment. If I decline
a required file, do not upload or substitute it; report the route as prepared, not run.
Pre-flight converted workflows and verify a real job on the selected target with its
required approved inputs before saying it runs. Return a verified authenticated UI link
only when the target supports it. Prefer read-only workflow ui <UUID> --workspace <id>
--environment standard, or --environment custom --deployment <id>, after the normal
saved record/version/content and sidebar file match and the release capability is true.
This retains the normal saved-workflow identity and selects the browser environment
without changing compute. When unavailable, explain the Workflows-sidebar/API fallback;
a verified deploy ui URL opens only the custom environment, not the saved workflow. A standard-environment save does not select a custom environment.
Retain accepted job IDs across timeouts; download outputs beside the workflow copy.
Lead the report with execution state, then include the route, changes/approvals, files,
job result/output paths and handoff. For custom environments include workspace and
Build/release/deployment IDs, GPU/region, actual worker bounds, current costs, actual
idle policy and stop command. Prepared or simulated work must be labeled as such.
```

If the user already explicitly chose a plan, append only those choices as **Previously approved choices**, including the exact Build/deployment IDs for an update. Do not derive approval from the bare modal click or silently fill defaults. Honor existing answers; newly discovered files or a changed target need an answer.

The prompt carries the workflow and its prompts to the requested service as part of the handoff. It does not authorize sending unrelated files or embedded secrets. A named-file upload approval is still required on every route.
