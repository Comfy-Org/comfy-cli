# comfy-cli: A Command Line Tool for ComfyUI

[![Run pytest](https://github.com/Comfy-Org/comfy-cli/actions/workflows/pytest.yml/badge.svg)](https://github.com/Comfy-Org/comfy-cli/actions/workflows/pytest.yml)
[![codecov](https://codecov.io/github/Comfy-Org/comfy-cli/graph/badge.svg?token=S64WJWD2ZX)](https://codecov.io/github/Comfy-Org/comfy-cli)
[![PyPI](https://img.shields.io/pypi/v/comfy-cli.svg)](https://pypi.org/project/comfy-cli/)
[![Downloads](https://static.pepy.tech/badge/comfy-cli/month)](https://pepy.tech/project/comfy-cli)
[![Python](https://img.shields.io/pypi/pyversions/comfy-cli)](https://pypi.org/project/comfy-cli/)
[![License](https://img.shields.io/pypi/l/comfy-cli)](https://github.com/Comfy-Org/comfy-cli/blob/main/LICENSE)

comfy-cli is a command-line tool for installing, running, and extending
[ComfyUI](https://github.com/comfyanonymous/ComfyUI) — the open-source
generative-media engine. Set up ComfyUI, install custom nodes and models, run
workflows, and call hosted partner image models, all from your terminal.

## Demo

<img src="https://github.com/yoland68/comfy-cli/raw/main/assets/comfy-demo.gif" width="400" alt="Comfy Command Demo">

## Features

- 🚀 One-command ComfyUI install and launch
- 🎨 Direct calls to partner image and video nodes (Flux, Ideogram, DALL·E, Recraft, Stability, Gemini/nano-banana, Kling, Luma, Runway, Pika, Vidu, Hailuo, Seedance, …) via `comfy generate`, no workflow JSON required
- 🔧 Custom node management — install, update, snapshot, bisect
- 📦 Fast dependency resolution with `uv` (`--fast-deps`, `--uv-compile`)
- 🗄️ Model downloads from CivitAI, Hugging Face, and direct URLs
- 🎬 Run workflows against a local ComfyUI server, including auto-conversion of UI-format JSON
- 🧪 Test ComfyUI and frontend pull requests with one flag
- 💻 Cross-platform: Windows, macOS, Linux
- ☁️  Route any workflow to **Comfy Cloud** with `--where cloud` (no GPU required)
- 🤖 Agent-friendly: every command emits structured `--json` envelopes
- 📚 Bundled skills teach Claude / Cursor to drive comfy natively

## Quick Start

```bash
pip install comfy-cli
comfy setup
```

`comfy setup` walks you through everything — local or cloud routing, authentication, and agent skill installation — in one interactive wizard. Pass `-y` for non-interactive (CI/scripted) installs.

## Run a workflow in the cloud

No local GPU? Route any API-format workflow to [Comfy Cloud](https://www.comfy.org/) with `--where cloud`:

```bash
comfy cloud login                                        # sign in via your browser (OAuth)
comfy run --workflow ./workflow.json --where cloud       # submits, prints a prompt_id, returns immediately
comfy jobs wait <prompt_id> --where cloud                # block until the job finishes (or: jobs status for a one-shot check)
comfy download <prompt_id> --where cloud -o ./outputs    # save the results locally
```

`comfy run` submits asynchronously and prints the `prompt_id` you feed to `jobs`/`download`; add `--wait` to block inline instead. Check your sign-in anytime with `comfy cloud whoami`. Export the workflow JSON from ComfyUI via **File → Export (API)** (UI-format JSON is auto-converted). Set cloud as your default target so you can drop the flag: `comfy set-default --where cloud`.

**Credits:** cloud generation (`comfy run --where cloud`, `comfy generate`) consumes Comfy Cloud credits and needs an active subscription. Discovery and inspection commands — `comfy cloud whoami`, `comfy cloud status`, `comfy jobs status/ls`, `comfy templates ls`, `comfy generate list` — don't. Check your balance and tier with `comfy cloud status`.

## Installation

1. (Recommended) Activate a virtual environment ([venv](https://docs.python.org/3/library/venv.html) or [conda](https://conda.io/projects/conda/en/latest/user-guide/getting-started.html)).

2. Install with `pip` (requires Python 3.10+):

   ```bash
   pip install comfy-cli
   ```

### Shell Autocomplete

Install shell completion so `comfy <TAB>` expands commands and options:

```bash
comfy --install-completion
```

## Usage

### Installing ComfyUI

To install ComfyUI using comfy, simply run:

`comfy install`

This command will download and set up the latest version of ComfyUI and ComfyUI-Manager on your
system. If you run in a ComfyUI repo that has already been setup. The command
will simply update the comfy.yaml file to reflect the local setup

- `comfy install --skip-manager`: Install ComfyUI without ComfyUI-Manager.
  To use a custom Manager fork or specific version, skip the default installation
  and install your own into the workspace venv:
  ```bash
  comfy install --skip-manager
  # Then install your custom Manager:
  pip install -e /path/to/your-manager-fork   # editable install
  # or
  pip install comfyui-manager==4.1b8          # specific version
  ```
- `comfy --workspace=<path> install`: Install ComfyUI into `<path>/ComfyUI`.
- `comfy install --fast-deps`: Use `uv` instead of `pip` for faster dependency resolution
  during initial ComfyUI installation. comfy-cli's built-in resolver compiles all requirements (core + custom nodes)
  into a single lockfile and installs from it. Also handles GPU-specific PyTorch wheel selection automatically.
- For `comfy install`, if no path specification like `--workspace, --recent, or --here` is provided, it will be implicitly installed in `<HOME>/comfy`.

#### Python environment handling

When you run `comfy install`, comfy-cli picks a Python environment for ComfyUI
dependencies using the following precedence:

1. An **active virtualenv or conda** environment (`VIRTUAL_ENV` / `CONDA_PREFIX`) is used as-is.
2. An **existing `.venv` or `venv`** directory inside the workspace is reused.
3. Otherwise the choice depends on how comfy-cli was installed:
   - **`pip install comfy-cli`** (global / system Python): dependencies go
     directly into the same Python environment. This is the typical Docker setup.
   - **`pipx install comfy-cli`** or **`uv tool install comfy-cli`** (isolated
     tool environment): a `.venv` is created inside the ComfyUI workspace.
     Use `comfy launch` to start ComfyUI with the correct Python.

### Updating ComfyUI

`comfy update` brings an existing workspace up to date:

- `comfy update` (or `comfy update comfy`): pull the branch the workspace is currently on and reinstall `requirements.txt`.
- `comfy update all`: also update every installed custom node.
- `comfy update cli`: upgrade comfy-cli itself.

By default `comfy update all` exits 0 even when the custom-node update step fails — the error is printed, but the exit code says success, so scripts and wrappers can't tell. Pass `--exit-on-fail` (the same flag name and default as `comfy node install --exit-on-fail`) to have that failure exit non-zero instead. The flag only changes the `all` target; `comfy update comfy` and `comfy update cli` already exit non-zero when they fail, and accept the flag as a no-op so it can be forwarded unconditionally.

One limit worth knowing: `--exit-on-fail` can only surface what ComfyUI-Manager's `cm-cli update` reports. `cm-cli` currently handles a *single* pack failing to update by printing `ERROR: ...` and carrying on, still exiting 0, so that particular case stays invisible until ComfyUI-Manager propagates it. Unlike `comfy node install`, the flag is not forwarded to `cm-cli` — only its `install` subcommand accepts `--exit-on-fail`; its `update` subcommand has no such option and would reject it.

When the flag does fire, the exit code is `cm-cli`'s own, normalized so it is always usable: a process killed by a signal becomes `128+N` rather than a truncated value, an exit code that would truncate to 0 becomes 1, and 2 becomes 1 so it can't be mistaken for a CLI usage error. In `--json` mode the failure is also reported as an `update_custom_nodes_failed` error envelope carrying `cm-cli`'s raw status in `details.cm_cli_returncode`.

#### Switching to a specific version

`comfy update comfy --version <X>` moves an existing workspace to a specific ComfyUI version — a downgrade (rollback) or an upgrade — without prompting for anything, so it is safe to run headlessly or from a script. `<X>` is `nightly` (the repo's default branch), `latest` (the newest stable release), or a version number such as `0.3.0` (a leading `v` is optional).

```bash
comfy update comfy --version 0.3.0      # roll back to the v0.3.0 release
comfy update comfy --version latest     # newest stable release
comfy update comfy --version nightly    # roll forward to the default branch
```

Behavior worth knowing:

- **The target is validated before anything is touched.** An unknown version exits non-zero, lists the nearest available versions, and leaves the working tree exactly as it was.
- **Uncommitted changes are stashed by default** (`git stash push -u`) and are *never* popped or dropped automatically — the stash ref is printed so you can restore them with `git stash pop`. Pass `--no-stash` if you would rather the command refuse to run on a dirty tree.
- **A version number checks out a tag, which leaves a detached HEAD.** That is expected. Roll forward again with `comfy update comfy --version nightly` (or `--version latest`); a plain `comfy update` cannot advance a detached HEAD.
- **Dependencies are reinstalled** from the target version's `requirements.txt`. PyTorch is deliberately left alone: the ComfyUI version doesn't determine your torch build, your machine does. If the dependency install fails, the command exits non-zero and says so — the tree is already on the new version, and re-running the same command is safe.
- `--version` and `--no-stash` apply only to target `comfy`; combining `--version` with `all` or `cli` is an error.

### Specifying execution path

- You can specify the path of ComfyUI where the command will be applied through path indicators as follows:
  - `comfy --workspace=<path>`: Run from the ComfyUI installed in the specified workspace.
  - `comfy --recent`: Run from the recently executed or installed ComfyUI.
  - `comfy --here`: Run from the ComfyUI located in the current directory.
- --workspace, --recent, and --here options cannot be used simultaneously.
- If there is no path indicator, the following priority applies:

  - Run from the default ComfyUI at the path specified by `comfy set-default <path>`.
  - Run from the recently executed or installed ComfyUI.
  - Run from the ComfyUI located in the current directory.

- Example 1: To run the recently executed ComfyUI:
  - `comfy --recent launch`
- Example 2: To install a package on the ComfyUI in the current directory:
  - `comfy --here node install comfyui-impact-pack`
- Example 3: To update the automatically selected path of ComfyUI and custom nodes based on priority:

  - `comfy node update all`

- You can use the `comfy which` command to check the path of the target workspace.
  - e.g `comfy --recent which`, `comfy --here which`, `comfy which`, ...

### Default Setup

The default sets the option that will be executed by default when no specific workspace's ComfyUI has been set for the command.

`comfy set-default <workspace path> ?[--launch-extras="<extra args>"]`

- `--launch-extras` option specifies extra args that are applied only during launch by default. However, if extras are specified at the time of launch, this setting is ignored.

### Launch ComfyUI

Comfy provides commands that allow you to easily run the installed ComfyUI.

`comfy launch`

- To run with default ComfyUI options:

  `comfy launch -- <extra args...>`

  `comfy launch -- --cpu --listen 0.0.0.0`

  - When you manually configure the extra options, the extras set by set-default will be overridden.

- To run background

  `comfy launch --background`

  `comfy --workspace=~/comfy launch --background -- --listen 10.0.0.10 --port 8000`

  - Instances launched with `--background` are displayed in the "Background ComfyUI" section of `comfy env`, providing management functionalities for a single background instance only.
  - Background-running ComfyUI can be stopped with `comfy stop`.

- To point **every** command (not just one) at a ComfyUI on a non-default
  address — e.g. a server you started _outside_ comfy-cli on `:8189` — export
  the `COMFY_LOCAL_URL` environment variable:

  `export COMFY_LOCAL_URL=http://127.0.0.1:8189`

  - Accepts `http://host:port`, `host:port`, or `http://host` (the port
    defaults to `8188`; the scheme is optional and, if present, must be
    `http`). IPv6 literals are bracketed: `COMFY_LOCAL_URL=http://[::1]:8189`.
  - Honored by `comfy env`, `comfy run`, `comfy jobs`, `comfy upload`/`download`,
    `comfy nodes`, and every other local-targeting command, so `comfy env`'s
    "Comfy Server Running" line now probes and reports the resolved address.
  - Precedence (per command): a per-command `--host`/`--port` flag wins, then
    `COMFY_LOCAL_URL`, then a comfy-cli-launched background server, then the
    `127.0.0.1:8188` default. A malformed value is ignored with a one-line
    stderr warning rather than breaking the command.

- to run ComfyUI with a specific pull request:

  `comfy install --pr "#1234"`

  `comfy install --pr "jtydhr88:load-3d-nodes"`

  `comfy install --pr "https://github.com/comfyanonymous/ComfyUI/pull/1234"`

  - If you want to run ComfyUI with a specific pull request, you can use the `--pr` option. This will automatically install the specified pull request and run ComfyUI with it.
  - Important: The --pr option cannot be combined with --version or --commit and will be rejected if used together.

- To test a frontend pull request:

  ```
  comfy launch --frontend-pr "#456"
  comfy launch --frontend-pr "username:branch-name"
  comfy launch --frontend-pr "https://github.com/Comfy-Org/ComfyUI_frontend/pull/456"
  ```

  - The `--frontend-pr` option allows you to test frontend PRs by automatically cloning, building, and using the frontend for that session.
  - Requirements: Node.js and npm must be installed to build the frontend.
  - Builds are cached for quick switching between PRs - subsequent uses of the same PR are instant.
  - Each PR is used only for that launch session. Normal launches use the default frontend.

  **Managing PR cache**:
  ```
  comfy pr-cache list              # List cached PR builds
  comfy pr-cache clean             # Clean all cached builds
  comfy pr-cache clean 456         # Clean specific PR cache
  ```

  - Cache automatically expires after 7 days
  - Maximum of 10 PR builds are kept (oldest are removed automatically)
  - Cache limits help manage disk space while keeping recent builds available

- To check VRAM/RAM usage: `comfy system-stats` (add `--where cloud` to target Comfy Cloud instead of local)
- To unload models / free the executor cache: `comfy free` (pass `--free-memory` to also reset the executor cache)

### Managing Custom Nodes

comfy provides a convenient way to manage custom nodes for extending ComfyUI's functionality. Here are some examples:

- Show custom nodes' information:

```
comfy node [show|simple-show] [installed|enabled|not-installed|disabled|all|snapshot|snapshot-list]
                             ?[--channel <channel name>]
                             ?[--mode [remote|local|cache]]
```

- `comfy node show all --channel recent`

  `comfy node simple-show installed`

  `comfy node update all`

  `comfy node install comfyui-impact-pack`

  > **Note:** the argument is the node's **Comfy Registry ID**, which is
  > lowercase (e.g. `comfyui-impact-pack`), not the GitHub repository name
  > (`ComfyUI-Impact-Pack`). Passing the repo-name casing fails to resolve with
  > an error like `Node 'ComfyUI-Impact-Pack@unknown' not found`. Find a node's
  > ID on its [Comfy Registry](https://registry.comfy.org) page or via
  > `comfy node show all`.

- Managing snapshot:

  `comfy node save-snapshot`

  `comfy node restore-snapshot <snapshot name>`

- Install dependencies:

  `comfy node install-deps --deps=<deps .json file>`

  `comfy node install-deps --workflow=<workflow .json/.png file>`

- Generate deps:

  `comfy node deps-in-workflow --workflow=<workflow .json/.png file> --output=<output deps .json file>`

#### `install` vs `registry-install`

comfy-cli offers two ways to install a custom node, backed by different
mechanisms:

- **`comfy node install <id>...`** delegates to
  [ComfyUI-Manager](https://github.com/Comfy-Org/ComfyUI-Manager)'s `cm-cli`.
  It accepts one or more node IDs, resolves them through the Manager's channel
  database (`--channel`, `--mode`), and installs dependencies via the Manager
  (so it also supports `--fast-deps`, `--no-deps`, and `--uv-compile`). This is
  the recommended command for day-to-day node management, and it requires
  ComfyUI-Manager to be present in the workspace.

  `comfy node install comfyui-impact-pack`

- **`comfy node registry-install <id> [--version <v>]`** talks directly to the
  [Comfy Registry](https://registry.comfy.org) API. It downloads the published
  archive for a single node (optionally pinned with `--version`), extracts it
  into `custom_nodes/`, and runs the node's own install script. It does **not**
  go through ComfyUI-Manager, so it does not require the Manager to be
  installed — this is the command the Registry's own install instructions use.

  `comfy node registry-install comfyui-impact-pack`

  `comfy node registry-install comfyui-impact-pack --version 1.0.0`  # install a specific version

  Because `registry-install` bypasses the Manager's dependency machinery and
  simply runs the node's bundled install script, it only accepts
  `--force-download` — it does **not** accept `--fast-deps`, `--no-deps`, or
  `--uv-compile`. If you want fast/unified dependency resolution, use
  `comfy node install` instead.

#### Unified Dependency Resolution (--uv-compile)

Requires ComfyUI-Manager v4.1+. Instead of installing dependencies per-node with
`pip install`, `--uv-compile` delegates to ComfyUI-Manager's unified resolver which batch-resolves
all custom node dependencies via `uv pip compile` with **cross-node conflict detection** —
it can identify which node packs have incompatible dependencies and why.

- Install with unified resolution:

  `comfy node install comfyui-impact-pack --uv-compile`

- Available on: `install`, `reinstall`, `update`, `fix`, `restore-snapshot`,
  `restore-dependencies`, `install-deps`

- Run standalone (resolve all existing custom node dependencies):

  `comfy node uv-sync`

- `--uv-compile` is mutually exclusive with `--fast-deps` and `--no-deps` —
  except on `restore-snapshot`, where `--fast-deps` selects the `--uv-compile`
  path instead (see [--fast-deps](#--fast-deps) below).

- To make `--uv-compile` the default for all commands, see
  [uv-compile default](#uv-compile-default) below.

- Use `--no-uv-compile` to override the default for a single command:

  `comfy node install comfyui-impact-pack --no-uv-compile`

#### --fast-deps

`--fast-deps` swaps comfy-cli's dependency installation from `pip` to comfy-cli's
built-in `uv`-based resolver (`DependencyCompiler`), which is significantly
faster and only requires `uv` (no ComfyUI-Manager). On a dependency version
conflict it prompts you interactively to pick a version.

- Accepted by: `comfy install`, `comfy node install`, `comfy node reinstall`,
  and `comfy node restore-snapshot`.

  `comfy install --fast-deps`

  `comfy node install comfyui-impact-pack --fast-deps`

- **Not** accepted by `comfy node registry-install`: that command installs a
  node directly from the Comfy Registry by running the node's own bundled
  install script, so it never touches comfy-cli's dependency resolver (see
  [`install` vs `registry-install`](#install-vs-registry-install) above).
- Mutually exclusive with `--no-deps` and `--uv-compile`, where those flags
  exist — the exact pairing is per-command:

  | Command | Rejected combination |
  |---------|----------------------|
  | `comfy node install` | `--fast-deps --no-deps`, `--fast-deps --uv-compile` |
  | `comfy node reinstall` | `--fast-deps --uv-compile` (no `--no-deps` on this command) |
  | `comfy node restore-snapshot` | `--fast-deps --no-uv-compile` (see below) |
  | `comfy install` | none (neither flag exists on this command) |

- **Exception — `comfy node restore-snapshot`:** here `--fast-deps` *selects*
  the `--uv-compile` fast path rather than conflicting with it. ComfyUI-Manager's
  `cm-cli` has no `--no-deps` on `restore-snapshot`, and its `--uv-compile`
  already implies no-deps internally, so the two are synonyms on this command.
  Combining `--fast-deps` with `--no-uv-compile` is the contradiction, and it
  errors: `Cannot use --fast-deps with --no-uv-compile`.

#### --fast-deps vs --uv-compile

Both flags use `uv` for faster dependency resolution, but they work differently:

|                       | `--fast-deps`                                   | `--uv-compile`                                |
|-----------------------|-------------------------------------------------|-----------------------------------------------|
| **Resolver**          | comfy-cli built-in (`DependencyCompiler`)       | ComfyUI-Manager (`UnifiedDepResolver`)        |
| **Scope**             | `comfy install`, `comfy node install/reinstall/restore-snapshot` | Custom node commands only    |
| **Conflict handling** | Interactive prompt to pick a version            | Automatic detection with node attribution     |
| **Config default**    | No                                              | Yes (`comfy manager uv-compile-default true`) |
| **Requires**          | Only `uv`                                       | ComfyUI-Manager v4.1+                         |

**When to use which:**
- For initial ComfyUI installation with uv: `comfy install --fast-deps`
- For custom node management with Manager v4.1+: `--uv-compile` (recommended)
- For custom node management with older Manager: `--fast-deps`

#### Bisect custom nodes

If you encounter bugs only with custom nodes enabled, and want to find out which custom node(s) causes the bug,
the bisect tool can help you pinpoint the custom node that causes the issue.

- `comfy node bisect start`: Start a new bisect session with optional ComfyUI launch args. It automatically marks the starting state as bad, and takes all enabled nodes when the command executes as the test set.
- `comfy node bisect good`: Mark the current active set as good, indicating the problem is not within the test set.
- `comfy node bisect bad`: Mark the current active set as bad, indicating the problem is within the test set.
- `comfy node bisect reset`: Reset the current bisect session.

### Managing Models

- Model downloading

  `comfy model download --url <URL> ?[--relative-path <PATH>] ?[--set-civitai-api-token <TOKEN>] ?[--set-hf-api-token <TOKEN>]`

  - URL: CivitAI page, Hugging Face file URL, etc...
  - You can also specify your API tokens via the `CIVITAI_API_TOKEN` and `HF_API_TOKEN` environment variables. The order of priority is `--set-X-token` (always highest priority), then the environment variables if they exist, and lastly your config's stored tokens from previous `--set-X-token` usage (which remembers your most recently set token values).
  - Tokens provided via the environment variables are never stored persistently in your config file. They are intended as a way to easily and safely provide transient secrets.

- Model remove

  `comfy model remove ?[--relative-path <PATH>] --model-names <model names>`

- Model list

  `comfy model list ?[--relative-path <PATH>]`

### Running on Comfy Cloud (`--where cloud`)

Comfy Cloud runs your workflow on Comfy's GPUs — no local ComfyUI install and no GPU required. The same verbs you use locally take `--where cloud`; the only extra step is signing in once.

Prerequisites — a Comfy account with a credit balance ([add credits](https://docs.comfy.org/interface/credits); cloud runs are metered per GPU-second):

```bash
comfy cloud login                   # opens your browser (OAuth + PKCE), stores a session
comfy cloud whoami                  # confirm who you're signed in as
comfy cloud status                  # workspace, credit balance, tier, concurrency limit
```

Then submit, watch, and collect:

```bash
comfy run --workflow my_workflow_api.json --where cloud   # submits, prints a prompt_id, returns
comfy jobs ls --where cloud                               # queue + history for your account
comfy jobs status <prompt_id> --where cloud               # one job's state
comfy jobs watch <prompt_id> --where cloud                # live progress until it finishes
comfy download <prompt_id> --where cloud                  # fetch the outputs
```

Notes:

- `--where cloud` is per-invocation. Make it the default with `comfy set-default --where cloud`; every command then honors it without the flag, and `--where local` overrides it for one call. Clear it again with `comfy set-default --clear-where`.
- Add `--wait` to `comfy run` to block until the job completes instead of returning immediately.
- `comfy run --prompt "<text>"` (no `--workflow`) runs a bundled default text2img graph. **That graph loads the SD1.5 checkpoint `v1-5-pruned-emaonly.ckpt`, which comfy-cli does not download for you** — install it into `models/checkpoints`, or point the graph at a checkpoint you do have with `comfy run --prompt "…" --set checkpoint=<name>`. The same applies on cloud, where the checkpoint must exist in your cloud assets.
- `comfy download` also reads a `prompt_id` from piped stdin, so `comfy run ... --where cloud | comfy download --where cloud` works.
- Models and custom nodes must exist on the cloud side. `comfy models search --where cloud` lists the cloud asset catalog, and `comfy nodes ls --where cloud` lists the node classes cloud can run.
- Sign out with `comfy cloud logout`. If a run fails with `cloud_unauthorized`, your session expired — re-run `comfy cloud login`.

### Calling partner nodes (`comfy generate`)

`comfy generate` calls Comfy's partner nodes directly from the terminal — no
local ComfyUI or workflow JSON required. It hits the same hosted partner nodes
you'd otherwise wire into a ComfyUI workflow, but as one-shot CLI calls. Image
models (Flux, Ideogram, DALL·E, Recraft, Stability, Runway, Reve, xAI Grok,
Google Gemini Flash Image aka **nano-banana**, …) and video models (Kling,
Luma, Runway Gen-3, Pika, Vidu, Moonvalley, Hailuo, Grok video, ByteDance
**Seedance**) are all covered; video jobs run async and the CLI polls until
the result is ready.

Prerequisites — a Comfy API key and a credit balance:

- [Create an API key](https://docs.comfy.org/development/comfyui-server/api-key-integration)
- [Browse partner nodes and per-call credit costs](https://docs.comfy.org/tutorials/partner-nodes/overview) · [pricing table](https://docs.comfy.org/tutorials/partner-nodes/pricing)
- [Add credits](https://docs.comfy.org/interface/credits)

Set the key once, then go:

```bash
export COMFY_API_KEY=comfyui-...   # or pass --api-key on each call

comfy generate list                                  # browse available models
comfy generate schema flux-pro                       # see params for one model
comfy generate flux-pro --prompt "a cat on the moon" \
    --width 1024 --height 1024 --download cat.png
```

Reference images can be passed as local paths — the CLI uploads them through
the cloud's storage endpoint (or base64-encodes inline, as each partner
requires):

```bash
comfy generate flux-kontext --prompt "add a top hat" \
    --input_image ./photo.jpg --download out.png

comfy generate upload ./photo.jpg                    # explicit upload
```

Async models (every video model plus the Flux family) block until ready by
default. Pass `--async` to return immediately with a job id, then resume later
with `comfy generate resume <model> <job_id>`. Examples:

```bash
comfy generate kling --prompt "a paper boat drifting on a river at dusk" \
    --duration 5 --download boat.mp4

comfy generate luma --prompt "..." --aspect_ratio 16:9 --async
# → prints job id; resume with:
comfy generate resume luma <job_id> --download out.mp4
```

**Gemini Flash Image (nano-banana)** — text-to-image and image edits in one
alias. Pass `--image` (repeatable) for reference images. The response is
inline base64, so `--download` is required to save:

```bash
comfy generate nano-banana --prompt "a watercolor of a sleeping fox" \
    --download fox.png

# Image edit — reference accepted as a local path, http(s) URL, or data URI:
comfy generate nano-banana --prompt "add a top hat" \
    --image ./cat.png --download out.png

# Switch model variants:
comfy generate nano-banana --prompt "..." --model gemini-3-pro-image-preview \
    --download out.png
```

**Seedance** — text-to-video and image-to-video, up to 1080p / 12s clips.
Resolution, ratio, duration, fps, etc. get passed through as flags; the CLI
inlines them into Seedance's prompt syntax for you:

```bash
comfy generate seedance --prompt "a hummingbird hovering over a flower" \
    --resolution 1080p --duration 5 --download bird.mp4

# Image-to-video: pick a lite/i2v variant and pass a first frame.
comfy generate seedance --model seedance-1-0-lite-i2v-250428 \
    --prompt "the wave crests and crashes" \
    --image ./still.jpg --download wave.mp4
```

### Managing ComfyUI-Manager

- Disable ComfyUI-Manager completely (no manager flags passed to ComfyUI):

  `comfy manager disable`

- Enable ComfyUI-Manager with new GUI:

  `comfy manager enable-gui`

- Enable ComfyUI-Manager without GUI (manager runs but UI is hidden):

  `comfy manager disable-gui`

- Enable ComfyUI-Manager with legacy GUI:

  `comfy manager enable-legacy-gui`

- Clear reserved startup action:

  `comfy manager clear`

- Migrate legacy git-cloned ComfyUI-Manager to pip package:

  `comfy manager migrate-legacy`

#### uv-compile default

Set `--uv-compile` as the default behavior for all custom node operations:

  `comfy manager uv-compile-default true`

When enabled, all node commands (`install`, `reinstall`, `update`, `fix`,
`restore-snapshot`, `restore-dependencies`, `install-deps`) will automatically
use `--uv-compile`. Use `--no-uv-compile` on any individual command to override.

To disable:

  `comfy manager uv-compile-default false`

## Beta Feature: format of comfy-lock.yaml (WIP)

```
basic:

models:
  - model: [name of the model]
    url: [url of the source, e.g. https://huggingface.co/...]
    paths: [list of paths to the model]
      - path: [path to the model]
      - path: [path to the model]
    hashes: [hashes for the model]
      - hash: [hash]
        type: [AutoV1, AutoV2, SHA256, CRC32, and Blake3]
    type: [type of the model, e.g. diffuser, lora, etc.]

  - model:
  ...

# compatible with ComfyUI-Manager's .yaml snapshot
custom_nodes:
  comfyui: [commit hash]
  file_custom_nodes:
  - disabled: [bool]
    filename: [.py filename]
    ...
  git_custom_nodes:
    [git-url]:
      disabled: [bool]
      hash: [commit hash]
    ...
```

## Analytics

We track analytics using Mixpanel to help us understand usage patterns and know where to prioritize our efforts. When you first download the cli, it will ask you to give consent. If at any point you wish to opt out:

```
comfy tracking disable
```

Check out the usage here: [Mixpanel Board](https://mixpanel.com/p/13hGfPfEPdRkjPtNaS7BYQ)

## Contributing

We welcome contributions to comfy-cli! For ideas, suggestions, or bug reports,
open an issue at [Comfy-Org/comfy-cli](https://github.com/Comfy-Org/comfy-cli/issues).
For code changes, fork the repo and open a pull request.

See [CONTRIBUTING.md](/CONTRIBUTING.md) for setup, the checks CI enforces,
and PR conventions. Notable changes are recorded in [CHANGELOG.md](/CHANGELOG.md).

## License

Released under the [GNU General Public License v3.0](https://github.com/Comfy-Org/comfy-cli/blob/main/LICENSE).

## Support

Questions or issues? [Open an issue](https://github.com/Comfy-Org/comfy-cli/issues)
or reach us on [Discord](https://discord.com/invite/comfyorg).

Happy diffusing with ComfyUI and comfy-cli! 🎉
