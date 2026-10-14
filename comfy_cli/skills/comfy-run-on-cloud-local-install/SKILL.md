---
name: comfy-run-on-cloud-local-install
description: "Reference skill cited by comfy-run-on-cloud. Read it with `comfy skills show comfy-run-on-cloud-local-install` when finding the local ComfyUI install behind a workflow: the running server, Comfy Desktop's split layout on macOS, extra model roots, what to do when the server is not running, and what each finding means for the build."
---

# comfy-run-on-cloud-local-install

`comfy which` can report a default path that does not exist, with `ok: true`. Confirm
every path with `ls` before using it.

## Best source: the running server

When ComfyUI is running, it already knows its version, its classes and which pack provides
each one.

| Install | Default port |
|---|---|
| Comfy Desktop | 8000 |
| OSS / comfy-cli | 8188 |

```shell
curl -s http://127.0.0.1:<port>/system_stats   # system.comfyui_version, argv, python
curl -s http://127.0.0.1:<port>/object_info    # per class: python_module = custom_nodes.<pack dir>
```

`comfy --json build inventory <workflow> --server <url>` joins these against the workflow.
It returns the classes missing locally and the local pack directory behind each custom class.

## Comfy Desktop (macOS) layout

Desktop splits one install across up to three places:

| What | Where to read it |
|---|---|
| Data dir: `models/`, `custom_nodes/`, `.venv/`, `user/` | `~/Library/Application Support/ComfyUI/config.json` → `basePath` |
| Code dir: the ComfyUI git checkout | `~/Library/Application Support/Comfy Desktop/installations.json` → `installPath` (code under `<installPath>/ComfyUI`). This file can list several installs. |
| Extra or shared model roots | `~/Library/Application Support/Comfy Desktop/settings.json` → `modelsDirs` (a list, e.g. a shared models folder plus `<data>/models`), and `…/Comfy Desktop/instance-model-paths/<inst>.yaml`. The startup log `user/comfyui_<port>.log` prints every model search path. OSS installs use `extra_model_paths.yaml` beside ComfyUI. |
| ComfyUI version | `installations.json` → `comfyVersion.baseTag`. `git -C <code dir> describe --tags` works too. |

`build inventory --desktop` reads `modelsDirs` for you.

If several installs are listed and their versions or data dirs differ, ask which one runs
the workflow; the newest is not necessarily the one. If they agree, use it without asking.

## When the server is not running

Do not start it just to read it. Fall back to files:

- version: `installations.json` → `comfyVersion.baseTag`, or `git describe` in the code dir;
- packs: one directory per pack under `<data>/custom_nodes/`. Read each `pyproject.toml`
  (`[project] name`, `version`, `[tool.comfy] PublisherId`) and `git -C <dir> rev-parse HEAD`
  if it is a checkout. Without `object_info`, map a workflow class to a pack by the node's
  `cnr_id`, or by grepping `NODE_CLASS_MAPPINGS` in the pack source;
- models: `build inventory --models-root` over every root; it does not need the server.

## Consequences for the build

- **Search every model root,** not just `<data>/models`. `--models-dir` takes one directory; pass every root to `build inventory` with `--models-root`.
- **Never point a scan at a shared root.** Shared libraries hold weights for other workflows. Some may be private and must not travel. A model enters the build only because the workflow loads it.
- **Pass `--python <data>/.venv/bin/python`** to any `comfy build` command that needs one.
- **A spec written with `-o` outside the install needs `--models-dir` and `--custom-nodes-dir`** on every later `validate`, `push` and `status`. Without them, local `localPath` entries fail to resolve.
- **Do not use the Desktop snapshot path (`--from-snapshot`) as the definition source.** It can predate the installed packs and return an empty build with no warning.
