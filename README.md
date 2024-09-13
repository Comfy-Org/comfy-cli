# Comfy-Cli: A Command Line Tool for ComfyUI

[![Test ComfyUI Example Workflows](https://github.com/Comfy-Org/ComfyUI-Mirror/actions/workflows/test-workflows.yaml/badge.svg)](https://github.com/Comfy-Org/ComfyUI-Mirror/actions/workflows/test-workflows.yaml)
[![Test ComfyUI Windows with Default Workflow](https://github.com/Comfy-Org/ComfyUI-Mirror/actions/workflows/test-workflows-windows.yaml/badge.svg)](https://github.com/Comfy-Org/ComfyUI-Mirror/actions/workflows/test-workflows-windows.yaml)

[![codecov](https://codecov.io/github/Comfy-Org/comfy-cli/graph/badge.svg?token=S64WJWD2ZX)](https://codecov.io/github/Comfy-Org/comfy-cli)

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
- `comfy --workspace=<path> install`: Install ComfyUI into `<path>/ComfyUI`.
- For `comfy install`, if no path specification like `--workspace, --recent, or --here` is provided, it will be implicitly installed in `<HOME>/comfy`.

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
  - `comfy --here node install ComfyUI-Impact-Pack`
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

  `comfy node install ComfyUI-Impact-Pack`

- Managing snapshot:

  `comfy node save-snapshot`

  `comfy node restore-snapshot <snapshot name>`

- Install dependencies:

  `comfy node install-deps --deps=<deps .json file>`

  `comfy node install-deps --workflow=<workflow .json/.png file>`

- Generate deps:

  `comfy node deps-in-workflow --workflow=<workflow .json/.png file> --output=<output deps .json file>`

#### Bisect custom nodes

If you encounter bugs only with custom nodes enabled, and want to find out which custom node(s) causes the bug,
the bisect tool can help you pinpoint the custom node that causes the issue.

- `comfy node bisect start`: Start a new bisect session with optional ComfyUI launch args. It automatically marks the starting state as bad, and takes all enabled nodes when the command executes as the test set.
- `comfy node bisect good`: Mark the current active set as good, indicating the problem is not within the test set.
- `comfy node bisect bad`: Mark the current active set as bad, indicating the problem is within the test set.
- `comfy node bisect reset`: Reset the current bisect session.

### Managing Models

- Model downloading

  `comfy model download --url <URL> ?[--relative-path <PATH>] ?[--set-civitai-api-token <TOKEN>]`

  - URL: CivitAI, huggingface file url, ...

- Model remove

  `comfy model remove ?[--relative-path <PATH>] --model-names <model names>`

- Model list

  `comfy model list ?[--relative-path <PATH>]`

### Managing ComfyUI-Manager

- disable GUI of ComfyUI-Manager (disable Manager menu and Server)

  `comfy manager disable-gui`

- enable GUI of ComfyUI-Manager

  `comfy manager enable-gui`

- Clear reserved startup action:

  `comfy manager clear`

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

If you encounter any issues or have questions about comfy-cli, please [open an issue](https://github.com/comfy-cli/issues) on our GitHub repository or contact us on [Discord](https://discord.gg/comfycontrib). We'll be happy to assist you!

Happy diffusing with ComfyUI and comfy-cli! 🎉
