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

This command will download and set up the latest version of ComfyUI on your
system. If you run in in a ComfyUI repo that has already been setup. The command
will simply update the comfy.yaml file to reflect the local setup

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

- Create a new custom node:

  `comfy-cli node create node-name`

- Edit an existing custom node:

  `comfy-cli node edit node-name`

- Remove a custom node:

  `comfy-cli node remove node-name`

- List available custom nodes:

  `comfy-cli node list`

## Format of comfy.yaml

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
repository](https://github.com/???/comfy-cli). If you'd like to contribute code,
please fork the repository and submit a pull request.


## License

comfy is released under the [GNU General Public License v3.0](https://github.com/drip-art/comfy-cli/blob/master/LICENSE).

## Support

If you encounter any issues or have questions about comfy-cli, please [open an issue](https://github.com/comfy-cli/issues) on our GitHub repository. We'll be happy to assist you!

Happy diffusing with ComfyUI and comfy-cli! 🎉

