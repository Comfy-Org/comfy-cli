---
name: comfy-run-on-cloud-local-install
description: "Reference skill cited by comfy-run-on-cloud. Read it with `comfy skills show comfy-run-on-cloud-local-install` when finding the local ComfyUI install behind a workflow: the running server, Comfy Desktop's split layout on macOS, extra model roots, what to do when the server is not running, and what each finding means for the build."
---

# comfy-run-on-cloud-local-install

Use `comfy --json which` / `comfy --json env` for verified paths and the `desktop`
block (code/data directories, interpreter, model roots, selection reason and candidate
installs). Older CLI versions can report a nonexistent default with `ok: true`;
confirm paths on disk. Last-launched selection is a discovery hint, not evidence
that it is the server running the workflow.
A missing selected workspace leaves the Desktop candidate in `which` error `details.desktop` or `env` data `workspace.desktop`; that candidate is neither the selected workspace nor proof of the running process.

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
| Data dir: `models/`, `custom_nodes/`, `.venv/`, `user/` | Selected install's `adoptedBaseDir` for adopted installs (legacy `ComfyUI/config.json` → `basePath` fallback); otherwise its code directory. The running server's `--base-directory` overrides these records. |
| Code dir: the ComfyUI git checkout | Both `~/Library/Application Support/comfyui-desktop-2/installations.json` and the older `…/Comfy Desktop/installations.json` → `installPath` (usually `<installPath>/ComfyUI`). Each registry can list several installs. |
| Extra or shared model roots | The selected registry's `settings.json` → `modelsDirs` and install record's `modelDirs`/`useSharedModels`. With a verified local process, use its `--extra-model-paths-config` YAML files at their full registry-owned paths instead of stale settings. The startup log prints effective search paths. OSS installs use `extra_model_paths.yaml`. Keep libraries from different registries separate. |
| ComfyUI version | `installations.json` → `comfyVersion.baseTag`. `git -C <code dir> describe --tags` works too. |

`build inventory --desktop --server <url>` requires direct-endpoint, local Python
process and shared-filesystem proof before automatic Desktop selection; a loopback
address alone is insufficient. It then matches an absolute code/data path or the full generated
`<owning registry>/instance-model-paths/<install_id>.yaml` path to one recorded
install, reads its effective model roots, and uses `--input-directory` (default
`<data>/input`) unless explicit `--input-root` values were supplied. It reports
`desktop` and `input_roots`.
The selected server's `--output-directory` (default `<data>/output`) is reported
as `desktop.output_dir`, with `output_dir_source` distinguishing an explicit
argument from the server default. That directory joins the exact deployment asset
scan so a referenced prior output can be approved as an input; unrelated outputs
are not included. Pass the reported `deployment_asset_roots` explicitly to the
run's `--asset-root` values. `local_server.launch_paths` contains only known path
arguments, without unrelated credentials or the raw process argument list.
Remote, forwarded or unverified endpoints retain reported catalog/metadata but
need explicit verified local roots, as do unmatched or ambiguous installs and
relative launch paths with unknown working directory. Missing roots produce
`root_not_found` warnings; resolve them before claiming complete inventory.
Without a server, `--desktop` uses one unambiguous install or one settings-only
library. It does not combine several unrelated installations.

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
- **Keep shared-root inventory separate from upload scope.** Read-only inventory must search every active root, including shared libraries. A workflow-scoped build includes only selected dependencies. Whole-instance packaging includes a shared root only after the user approves that named root and its listed uploads.
- **Pass the verified interpreter** from the selected install's `desktop.python` (or the running process's interpreter) to any `comfy build` command that needs one. Check it exists. Adopted installs may use `adoptedPythonPath`; do not assume every instance uses `<data>/.venv/bin/python`.
- **A spec written with `-o` outside the install needs `--models-dir` and `--custom-nodes-dir`** on every later `validate`, `push` and `status`. Without them, local `localPath` entries fail to resolve.
- **Do not use the Desktop snapshot path (`--from-snapshot`) as the definition source.** It can predate the installed packs and return an empty build with no warning.
