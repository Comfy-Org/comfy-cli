# The Run on Cloud prompt

This is the text **Copy prompt** puts on the clipboard in local ComfyUI. The
person pastes it into Claude Code or Codex, and the agent loads the
`comfy-run-on-cloud` skill from it.

```text
Run my ComfyUI workflow on Comfy Cloud.
If comfy is missing: uv tool install comfy-cli
Read the skill first: comfy skills show comfy-run-on-cloud
Workflow: workflows/<name>.json
ComfyUI: http://127.0.0.1:8188
```

Fill it from the running app:

- `Workflow:` is the saved workflow's path under the user's data folder, as
  the server's `/api/userdata` takes it (`workflows/<name>.json`, subfolders
  included). Save the workflow before copying; an unsaved tab has no path.
- `ComfyUI:` is the address the page was loaded from.
- When the server runs with `--multi-user`, add one more line with the user
  the page is signed in as, which the agent sends as the `comfy-user` header:

  ```text
  ComfyUI user: <user>
  ```

The prompt names no build, GPU or upload. The skill asks for those in the chat,
only when the workflow needs them.
