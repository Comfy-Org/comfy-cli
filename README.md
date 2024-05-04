# comfy-cli

comfy-cli is a command line tool that helps users easily install and manage [ComfyUI](https://github.com/comfyanonymous/ComfyUI), a powerful open-source machine learning framework. With comfy-cli, you can quickly set up ComfyUI, install packages, and manage custom nodes, all from the convenience of your terminal.

## Features

- 🚀 Easy installation of ComfyUI with a single command
- 📦 Seamless package management for ComfyUI extensions and dependencies
- 🔧 Custom node management for extending ComfyUI's functionality
- 🗄️ Download checkpoints and save model hash
- 💻 Cross-platform compatibility (Windows, macOS, Linux)
- 📖 Comprehensive documentation and examples

## Installation

To install comfy-cli, make sure you have Python 3.7 or higher installed on your system. Then, run the following command:

`pip install comfy-cli`


## Usage

### Installing ComfyUI

To install ComfyUI using comfy, simply run:

`comfy install`

This command will download and set up the latest version of ComfyUI and ComfyUI-Manager on your
system. If you run in a ComfyUI repo that has already been setup. The command
will simply update the comfy.yaml file to reflect the local setup

  * `comfy install --skip-manager`: Install ComfyUI without ComfyUI-Manager.
  * `comfy --workspace=<path> install`: Install ComfyUI into `<path>/ComfyUI`.
  * For `comfy install`, if no path specification like `--workspace, --recent, or --here` is provided, it will be implicitly installed in `<HOME>/comfy`.


### Specifying execution path

* You can specify the path of ComfyUI where the command will be applied through path indicators as follows:
  * `comfy --workspace=<path>`: Run from the ComfyUI installed in the specified workspace.
  * `comfy --recent`: Run from the recently executed or installed ComfyUI.
  * `comfy --here`: Run from the ComfyUI located in the current directory.
* --workspace, --recent, and --here options cannot be used simultaneously.
* If there is no path indicator, the following priority applies:
  * Run from the default ComfyUI at the path specified by `comfy set-default <path>`.
  * Run from the recently executed or installed ComfyUI.
  * Run from the ComfyUI located in the current directory.

* Example 1: To run the recently executed ComfyUI:
  * `comfy --recent launch`
* Example 2: To install a package on the ComfyUI in the current directory:
  * `comfy --here node install ComfyUI-Impact-Pack`
* Example 3: To update the automatically selected path of ComfyUI and custom nodes based on priority:
  * `comfy node update all`

* You can use the `comfy which` command to check the path of the target workspace.
  * e.g `comfy --recent which`, `comfy --here which`, `comfy which`, ...

### Launch ComfyUI

Comfy provides commands that allow you to easily run the installed ComfyUI.

  `comfy launch`

- To run with default ComfyUI options:

  `comfy launch -- <extra args...>`

  `comfy launch -- --cpu --listen 0.0.0.0`

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
-
  `comfy node show all --channel recent`

  `comfy node simple-show installed`

  `comfy node update all`

  `comfy node install ComfyUI-Impact-Pack`


- Managing snapshot:

  `comfy node save-snapshot`

  `comfy node restore-snapshot <snapshot name>`


### Managing Models

- Model downloading

  `comfy model get`

  *Downloading models that have already been installed will 

- Model remove

  `comfy model enable-gui`

- Model list

  `comfy model list`


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

## Contributing

We welcome contributions to comfy-cli! If you have any ideas, suggestions, or
bug reports, please open an issue on our [GitHub
repository](https://github.com/Comfy-Org/comfy-cli/issues). If you'd like to contribute code,
please fork the repository and submit a pull request.

Check out the [Dev Guide](/DEV_README.md) for more details.

## License

comfy is released under the [GNU General Public License v3.0](https://github.com/drip-art/comfy-cli/blob/master/LICENSE).

## Support

If you encounter any issues or have questions about comfy-cli, please [open an issue](https://github.com/comfy-cli/issues) on our GitHub repository. We'll be happy to assist you!

Happy diffusing with ComfyUI and comfy-cli! 🎉

