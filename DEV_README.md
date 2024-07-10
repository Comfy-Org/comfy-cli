# Development Guide

This guide provides an overview of how to develop in this repository.

## General guide

1. Clone the repo, create and activate a conda env. Minimum Python version is 3.9.

2. Install the package to local

`pip install -e .`

3. Set ENVIRONMENT variable to DEV.

`export ENVIRONMENT=dev`

4. Test script running

`comfy --help`

5. Use pre commit hook

`pre-commit install`

## Debug

You can add following config to your VSCode `launch.json` to launch debugger.

```json
{
  "name": "Python Debugger: Run",
  "type": "debugpy",
  "request": "launch",
  "module": "comfy_cli.__main__",
  "args": [],
  "console": "integratedTerminal"
}
```

## Linting and Type Checking

Here is the recommended VScode settings. We use Black to format python code. We use Pylance's to do some basic type checking.

```
{
    "[python]": {
        "editor.defaultFormatter": "ms-python.black-formatter",
        "editor.formatOnSave": true

      },
    "python.languageServer": "Pylance",
    "python.analysis.typeCheckingMode": "basic"
}
```

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

## Guide

- Use `typer` for all command args management
- Use `rich` for all console output
  - For progress reporting, use either [`rich.progress`](https://rich.readthedocs.io/en/stable/progress.html)

## Contact

If you have any questions or need further assistance, please contact the project maintainer at [???](mailto:???@drip.art).

Happy coding!
