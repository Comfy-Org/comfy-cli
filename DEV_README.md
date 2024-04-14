
# Development Guide

This guide provides an overview of how to develop in this repository.

## General guide

1. Clone the repo

2. Install requirements

   `pip install -r requirements.txt`

3. Add commands or modify existing commands

4. Test package:

   pip install .

## Add New Command

- Register it under `comfy/comfy.py` 

If it's contains subcommand, create folder under comfy/command/[new_command] and
add the following boilerplate

`comfy/command/[new_command]/__init__.py`

```
from .command import app
```

`comfy/command/[new_command]command.py`

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
