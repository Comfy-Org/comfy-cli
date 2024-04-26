# comfy-cli

comfy-cli is a command line tool that helps users easily install and manage [ComfyUI](https://github.com/comfyanonymous/ComfyUI), a powerful open-source machine learning framework. With comfy-cli, you can quickly set up ComfyUI, install packages, and manage custom nodes, all from the convenience of your terminal.

## Features

- 🚀 Easy installation of ComfyUI with a single command
- 📦 Seamless package management for ComfyUI extensions and dependencies
- 🔧 Custom node management for extending ComfyUI's functionality
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
  * `comfy install --workspace=<path>`: Install ComfyUI into `<path>/ComfyUI`.


### Launch ComfyUI

Comfy provides commands that allow you to easily run the installed ComfyUI.

- To execute specifying the path of the workspace where ComfyUI is installed:

  `comfy launch --workspace <path>`

- To run ComfyUI from the current directory, if you are inside the ComfyUI repository:
  
  `comfy launch`
  
- To execute the ComfyUI that was last run or last installed, if you are outside of the ComfyUI repository:

  `comfy launch`

- To run in CPU mode:

  `comfy launch --cpu`


### Managing Packages

comfy allows you to easily install, update, and remove packages for ComfyUI. Here are some examples:

- Install a package:

  `comfy package install package-name`

- Update a package:

  `comfy package update package-name`

- Remove a package:

  `comfy package remove package-name`

- List installed packages:

  `comfy package list`


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


## Format of comfy.yaml (WIP)

```
basic:

models:
  - model: [name of the model] 
    url: [url of the source, e.g. https://huggingface.co/...]
    paths: [list of paths to the model]
      - path: [path to the model]
      - path: [path to the model]
    hash: [md5hash for the model]
    type: [type of the model, e.g. diffuser, lora, etc.]

  - model:
  ...

custom_nodes:
  - ???
```

## Contributing

We welcome contributions to comfy-cli! If you have any ideas, suggestions, or
bug reports, please open an issue on our [GitHub
repository](https://github.com/Comfy-Org/comfy-cli/issues). If you'd like to contribute code,
please fork the repository and submit a pull request.


## License

comfy is released under the [GNU General Public License v3.0](https://github.com/drip-art/comfy-cli/blob/master/LICENSE).

## Support

If you encounter any issues or have questions about comfy-cli, please [open an issue](https://github.com/comfy-cli/issues) on our GitHub repository. We'll be happy to assist you!

Happy diffusing with ComfyUI and comfy-cli! 🎉

