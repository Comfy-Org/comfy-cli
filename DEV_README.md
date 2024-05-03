
# Development Guide

This guide provides an overview of how to develop in this repository.

## General guide

1. Clone the repo, create and activate a conda env

2. Install the package to local

  `pip install -e .`

3. Test script running

  `comfy --help`

4. Use pre commit hook

  `pre-commit install`

## Make changes to the code base

There is a potential need for you to reinstall the package. You can do this by
either run `pip install -e .` again (which will reinstall), or manually
uninstall `pip uninstall comfy-cli` and reinstall, or even cleaning your conda
env and reinstalling the package (`pip install -e .`)

## Add New Command

- Register it under `comfy_cli/cmdline.py` 

If it's contains subcommand, create folder under comfy_cli/command/[new_command] and
add the following boilerplate

`comfy_cli/command/[new_command]/__init__.py`

```
from .command import app
```

`comfy_cli/command/[new_command]command.py`

```
import typer

app = typer.Typer()

@app.command()
def option_a(name: str):
  """Add a new custom node"""
  print(f"Adding a new custom node: {name}")


@app.command()
def remove(name: str):
  """Remove a custom node"""
  print(f"Removing a custom node: {name}")

```

## Code explainer:

- `comfy_cli/cmdline.py` is the entry point of the CLI
- `comfy_cli/command/` contains definition for some commands (e.g. `node`,
  `model`, etc)
- `comfy_cli/config_manager.py` implements ConfigManager class that handles
  comfy_cli configuration (config.ini) file reading and writing
- `comfy_cli/workspace_manager.py` implements WorkspaceManager class that
  handles which ComfyUI workspace (path) and defines workspace comfy-lock.yaml
  file that register the state fo the comfy workspace.
- `comfy_cli/env_checker.py` implements EnvChecker class that helps with python
  env related variables
- `comfy_cli/tracking.py` handles opt-in anonymous telemetry data from users.


## Guide

- Use `typer` for all command args management
- Use `rich` for all console output
  - For progress reporting, use either [`rich.progress`](https://rich.readthedocs.io/en/stable/progress.html)

## Contact

If you have any questions or need further assistance, please contact the project maintainer at [???](mailto:???@drip.art).

Happy coding!
