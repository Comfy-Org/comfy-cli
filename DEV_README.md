# Development Guide

This guide provides an overview of how to develop in this repository.

## General guide

1. Clone the repo, create and activate a conda env. Minimum Python version is 3.9.

2. Install the package to your conda env.

`pip install -e .`

3. Set ENVIRONMENT variable to DEV.

`export ENVIRONMENT=dev`

4. Check if the "comfy" package can run.

`comfy --help`

5. Install the pre-commit hook to ensure that your code won't need reformatting later.

`pre-commit install`

6. To save time during code review, it's recommended that you also manually run
   the unit tests before submitting a pull request (see below).

## Running the unit tests

1. Install pytest into your conda env. You should preferably be using Python 3.9
   in your conda env, since it's the version we are targeting for compatibility.

`pip install pytest pytest-cov`

2. Verify that all unit tests run successfully.

`pytest --cov=comfy_cli --cov-report=xml .`

## Debugging

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

## Making changes to the code base

There is a potential need for you to reinstall the package. You can do this by
either run `pip install -e .` again (which will reinstall), or manually
uninstall `pip uninstall comfy-cli` and reinstall, or even cleaning your conda
env and reinstalling the package (`pip install -e .`)

## Packaging custom nodes with `.comfyignore`

`comfy node pack` and `comfy node publish` now read an optional `.comfyignore`
file in the project root. The syntax matches `.gitignore` (implemented with
`PathSpec`'s `gitwildmatch` rules), so you can reuse familiar patterns to keep
development-only artifacts out of your published archive.

- Patterns are evaluated against paths relative to the directory you run the
  command from (usually the repo root).
- Files required by the pack command itself (e.g. `__init__.py`, `web/*`) are
  still forced into the archive even if they match an ignore pattern.
- If no `.comfyignore` is present the command falls back to the original
  behavior and zips every git-tracked file.

Example `.comfyignore`:

```gitignore
docs/
frontend/
tests/
*.psd
```

Commit the file alongside your node so teammates and CI pipelines produce the
same trimmed package.

## Adding a new command

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

## Important notes

- Use `typer` for all command args management
- Use `rich` for all console output
  - For progress reporting, use either [`rich.progress`](https://rich.readthedocs.io/en/stable/progress.html)

## Develop comfy-cli and ComfyUI-Manager (cm-cli) together
### Making changes to both
1. Fork your own branches of `comfy-cli` and `ComfyUI-Manager`, make changes
2. Be sure to commit any changes to `ComfyUI-Manager` to a new branch, and push to remote

### Trying changes to both
1. clone the changed branch of `comfy-cli`, then live install `comfy-cli`:
  - `pip install -e comfy-cli`
2. Go to a test dir and run:
  - `comfy --here install --manager-url=<path-or-url-to-fork-of-ComfyUI-Manager>`
3. Run:
  - `cd ComfyUI/custom_nodes/ComfyUI-Manager/ && git checkout <changed-branch> && cd -`
4. Further changes can be pulled into these copies of the `comfy-cli` and `ComfyUI-Manager` repos

### Debugging both simultaneously
1. Follow instructions above to get working install with changes
2. Add breakpoints directly to code: `import ipdb; ipdb.set_trace()`
3. Execute relevant `comfy-cli` command


## Contact

If you have any questions or need further assistance, please contact the project maintainer at [???](mailto:???@drip.art).

Happy coding!
