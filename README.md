# comfy-cli: A Command Line Tool for ComfyUI

[![Run pytest](https://github.com/Comfy-Org/comfy-cli/actions/workflows/pytest.yml/badge.svg)](https://github.com/Comfy-Org/comfy-cli/actions/workflows/pytest.yml)
[![codecov](https://codecov.io/github/Comfy-Org/comfy-cli/graph/badge.svg?token=S64WJWD2ZX)](https://codecov.io/github/Comfy-Org/comfy-cli)
[![PyPI](https://img.shields.io/pypi/v/comfy-cli.svg)](https://pypi.org/project/comfy-cli/)
[![Downloads](https://static.pepy.tech/badge/comfy-cli/month)](https://pepy.tech/project/comfy-cli)
[![Python](https://img.shields.io/pypi/pyversions/comfy-cli)](https://pypi.org/project/comfy-cli/)
[![License](https://img.shields.io/pypi/l/comfy-cli)](https://github.com/Comfy-Org/comfy-cli/blob/main/LICENSE)

comfy-cli is a command line tool that helps users easily install and manage
[ComfyUI](https://github.com/comfyanonymous/ComfyUI), a powerful open-source
machine learning framework. With comfy-cli, you can quickly set up ComfyUI,
install packages, and manage custom nodes, all from the convenience of your
terminal.

## Demo

<img src="https://github.com/yoland68/comfy-cli/raw/main/assets/comfy-demo.gif" width="400" alt="Comfy Command Demo">

## Features

- 🚀 Easy installation of ComfyUI with a single command
- 📦 Seamless package management for ComfyUI extensions and dependencies
- 🔧 Custom node management for extending ComfyUI's functionality
- 🗄️ Download checkpoints and save model hash
- 💻 Cross-platform compatibility (Windows, macOS, Linux)
- 📖 Comprehensive documentation and examples
- 🎉 install pull request to ComfyUI automatically

## Installation

1. (Recommended, but not necessary) Enable virtual environment ([venv](https://docs.python.org/3/library/venv.html)/[conda](https://conda.io/projects/conda/en/latest/user-guide/getting-started.html))

2. To install comfy-cli, make sure you have Python 3.9 or higher installed on your system. Then, run the following command:

   `pip install comfy-cli`

### Shell Autocomplete

To install autocompletion hints in your shell run:

`comfy --install-completion`

This enables you to type `comfy [TAP]` to autocomplete commands and options

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
  - Since "Comfy Server Running" in `comfy env` only shows the default port 8188, it doesn't display ComfyUI running on a different port.
  - Background-running ComfyUI can be stopped with `comfy stop`.

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

- Managing snapshot:

  `comfy node save-snapshot`

  `comfy node restore-snapshot <snapshot name>`

- Install dependencies:

  `comfy node install-deps --deps=<deps .json file>`

  `comfy node install-deps --workflow=<workflow .json/.png file>`

- Generate deps:

  `comfy node deps-in-workflow --workflow=<workflow .json/.png file> --output=<output deps .json file>`

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

- `--uv-compile` is mutually exclusive with `--fast-deps` and `--no-deps`.

- To make `--uv-compile` the default for all commands, see
  [uv-compile default](#uv-compile-default) below.

- Use `--no-uv-compile` to override the default for a single command:

  `comfy node install comfyui-impact-pack --no-uv-compile`

#### --fast-deps vs --uv-compile

Both flags use `uv` for faster dependency resolution, but they work differently:

|                       | `--fast-deps`                                   | `--uv-compile`                                |
|-----------------------|-------------------------------------------------|-----------------------------------------------|
| **Resolver**          | comfy-cli built-in (`DependencyCompiler`)       | ComfyUI-Manager (`UnifiedDepResolver`)        |
| **Scope**             | `comfy install`, `comfy node install/reinstall` | Custom node commands only                     |
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

We welcome contributions to comfy-cli! If you have any ideas, suggestions, or
bug reports, please open an issue on our [GitHub
repository](https://github.com/yoland68/comfy-cli/issues). If you'd like to contribute code,
please fork the repository and submit a pull request.

Check out the [Dev Guide](/DEV_README.md) for more details.

## License

comfy is released under the [GNU General Public License v3.0](https://github.com/yoland68/comfy-cli/blob/master/LICENSE).

## Support

If you encounter any issues or have questions about comfy-cli, please [open an issue](https://github.com/comfy-cli/issues) on our GitHub repository or contact us on [Discord](https://discord.com/invite/comfyorg). We'll be happy to assist you!

Happy diffusing with ComfyUI and comfy-cli! 🎉
