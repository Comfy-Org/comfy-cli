# Contributing to comfy-cli

Thanks for contributing! `comfy-cli` is a Python/[Typer](https://typer.tiangolo.com/)
CLI for installing and running ComfyUI, published to PyPI as
[`comfy-cli`](https://pypi.org/project/comfy-cli/).

- **Bugs and feature requests** — open an issue at
  [Comfy-Org/comfy-cli/issues](https://github.com/Comfy-Org/comfy-cli/issues)
  using one of the issue templates.
- **Code changes** — fork the repo, branch off `main`, and open a pull request.
- **Questions** — ask in [Discord](https://discord.com/invite/comfyorg).

This file is the development guide as well as the contribution guide: the
sections below cover environment setup, the checks CI enforces, PR conventions,
and how to add a new command.

## Before you open a pull request

Run the same three checks CI enforces on every PR to `main`:

```bash
ruff check .            # lint          (ruff_check.yml)
ruff format --check .   # formatting    (CI runs `ruff format --diff`)
pytest                  # unit tests    (pytest.yml)
```

All three must pass. `ruff` is pinned in
[`.pre-commit-config.yaml`](.pre-commit-config.yaml) and in the `ruff_check`
workflow — match that version locally so a formatting pass does not fight CI.

Changes under `comfy_cli/**` or `tests/e2e/**` additionally trigger
[`build-and-test.yml`](.github/workflows/build-and-test.yml), which runs the E2E
and platform tests on Linux, Windows, and macOS.

Coverage is reported to Codecov, so new code should come with tests.

### Contributor License Agreement

First-time contributors are asked to sign the CLA. The CLA Assistant bot
comments on your PR with the text and the phrase to reply with; the check
clears once you have.

## Setup

Minimum Python version is **3.10** (the version CI targets, and the floor
declared in `pyproject.toml`).

The project is `uv`-based, and `uv.lock` is authoritative — CI installs with
`--locked`, so a dependency change must land as a deliberate lockfile bump.

```bash
uv sync --extra dev        # runtime + dev deps (ruff, pytest, pytest-cov, pre-commit, jsonschema)
uv run comfy --help        # check the `comfy` entry point works
```

If you prefer a hand-managed virtualenv or conda env, an editable install works
too:

```bash
pip install -e '.[dev]'
export ENVIRONMENT=dev
comfy --help
```

Install the pre-commit hook so commits are checked before they reach CI:

```bash
pre-commit install
```

The hook runs ruff (lint + format), a few file-hygiene checks, `pyproject-fmt`,
and the `uv` lock/export/sync hooks.

## Branching and pull requests

The default branch is **`main`** — branch off `main` and open PRs against it.

> **Gotcha:** a local checkout may already sit on a working branch, _not_ `main`.
> Don't assume the current branch is the base:
>
> ```bash
> git fetch origin
> git switch -c my-feature origin/main
> # or, without disturbing the current checkout:
> git worktree add ../comfy-cli-my-feature -b my-feature origin/main
> ```

Commit messages follow [Conventional Commits](https://www.conventionalcommits.org/),
scoped by the area touched — this is what the vast majority of the history looks
like, and what release notes are assembled from:

```
fix(jobs): stream live progress in `jobs watch` by attaching as the submitting client
feat(templates): serve stale gallery cache immediately, revalidate in background
docs: ...  test: ...  refactor: ...  ci: ...  chore: ...
```

PRs are squash-merged, so the PR title becomes the commit subject — write it in
the same form. Keep a PR to one reviewable change; describe the user-visible
effect, and link the issue it closes.

Notable user-facing changes should be summarized in
[`CHANGELOG.md`](CHANGELOG.md) under `## [Unreleased]`.

## Running the unit tests

```bash
uv run --extra dev pytest --cov=comfy_cli --cov-report=xml .
```

Or, in an environment where the dev extras are already installed:

```bash
pytest --cov=comfy_cli --cov-report=xml .
```

## Running E2E tests

E2E tests perform real `comfy install`, `comfy launch`, and `comfy node` operations.
They are **disabled by default** and must be explicitly enabled.

```bash
TEST_E2E=true pytest tests/e2e/
```

For pre-release testing against alternate ComfyUI repositories (e.g. Manager v4):

```bash
TEST_E2E=true \
TEST_E2E_COMFY_URL="https://github.com/ltdrdata/ComfyUI.git@dr-bump-manager" \
pytest tests/e2e/ -v
```

See [docs/TESTING-e2e.md](docs/TESTING-e2e.md) for the full guide including
environment variables, test suite details, and scenario descriptions.

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
uninstall `pip uninstall comfy-cli` and reinstall, or even cleaning your virtual
env and reinstalling the package (`pip install -e .`). With `uv`, `uv sync
--extra dev` re-resolves the environment in place.

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

ComfyUI-Manager is now installed as a pip package (via `manager_requirements.txt`
in the ComfyUI root) rather than being git-cloned into `custom_nodes/`.

### Making changes to both

1. Fork your own branches of `comfy-cli` and `ComfyUI-Manager`, make changes.
2. Live-install `comfy-cli`:
   - `pip install -e /path/to/comfy-cli`
3. Live-install your fork of `ComfyUI-Manager` in editable mode:
   - `pip install -e /path/to/ComfyUI-Manager`
4. This makes the `cm-cli` entry point available and points it at your local source.

### Trying changes to both

1. Install both packages in editable mode as described above.
2. Go to a test dir and run:
   - `comfy --here install`
3. The `cm-cli` command will resolve to your locally installed editable package.

### Debugging both simultaneously

1. Follow instructions above to get working install with changes.
2. Add breakpoints directly to code: `import ipdb; ipdb.set_trace()`
3. Execute relevant `comfy-cli` command.

## Contact

If you have any questions or need further assistance,
[open an issue](https://github.com/Comfy-Org/comfy-cli/issues) or reach us on
[Discord](https://discord.com/invite/comfyorg).

Happy coding!
