# AGENTS.md

Onboarding notes for anyone (human or automated agent) making changes to
`comfy-cli` — a Python/[Typer](https://typer.tiangolo.com/) CLI for installing
and running ComfyUI. This is a quick-start; see [`DEV_README.md`](DEV_README.md)
for the fuller development guide.

## Setup

Requires Python >= 3.10 (CI targets 3.10, the minimum supported version). Install
the package with its dev extras in editable mode:

```bash
pip install -e '.[dev]'
```

This pulls in the runtime dependencies plus the dev toolchain (`ruff`, `pytest`,
`pytest-cov`, `pre-commit`, `jsonschema`).

## Verify your changes

Before opening a PR, run the same three checks CI enforces:

```bash
ruff check .            # lint
ruff format --check .   # formatting (CI runs `ruff format --diff`)
pytest                  # unit tests
```

All three must pass. Keep changes formatted with `ruff format` and imports sorted
(ruff's isort rules are enabled) so the format check stays green.

### pre-commit

The repo ships a [pre-commit](https://pre-commit.com/) config that runs ruff (lint
+ format) and a few file hygiene hooks on every commit. Install it once so commits
are auto-checked:

```bash
pre-commit install
```

## Branching

The default branch is **`main`** — branch off `main` and open PRs against it.

> **Gotcha:** a local checkout may already sit on a working branch (e.g.
> `agent-cli`), *not* `main`. Don't assume the current branch is the base. Create
> your branch (or worktree) off `origin/main` explicitly:
>
> ```bash
> git fetch origin
> git switch -c my-feature origin/main
> # or, without disturbing the current checkout:
> git worktree add ../comfy-cli-my-feature -b my-feature origin/main
> ```

## Conventions

- Use `typer` for command/argument handling and `rich` for console output.
- New commands are registered in `comfy_cli/cmdline.py`; subcommands live under
  `comfy_cli/command/<name>/`. See [`DEV_README.md`](DEV_README.md#adding-a-new-command)
  for the boilerplate.
- End-to-end tests are disabled by default; enable them with `TEST_E2E=true pytest
  tests/e2e/` (see [`docs/TESTING-e2e.md`](docs/TESTING-e2e.md)).
