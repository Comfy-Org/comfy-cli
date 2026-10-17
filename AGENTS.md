# AGENTS.md

Onboarding notes for anyone (human or automated agent) making changes to
`comfy-cli` — a Python/[Typer](https://typer.tiangolo.com/) CLI for installing
and running ComfyUI. This is a quick-start; see [`CONTRIBUTING.md`](CONTRIBUTING.md)
for the fuller development and contribution guide.

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
  `comfy_cli/command/<name>/`. See [`CONTRIBUTING.md`](CONTRIBUTING.md#adding-a-new-command)
  for the boilerplate.
- End-to-end tests are disabled by default; enable them with `TEST_E2E=true pytest
  tests/e2e/` (see [`docs/TESTING-e2e.md`](docs/TESTING-e2e.md)).

## Architecture Decision Records

Architectural decisions live in `docs/adr/`. Code changes must be consistent
with accepted ADRs. Proposed ADRs express design direction and should be
treated as guidance during review. See `.agents/checks/adr-compliance.md` for
the automated review rules.

Comfy repositories standardize the ADR behavior: canonical records, identifier
and status semantics, review expectations, and supersession history should mean
the same thing everywhere. Each repository implements validation and CI with
its native toolchain. In this Python repository that means
`scripts/check_adrs.py`, pytest, pre-commit, and GitHub Actions; do not copy a
TypeScript validator merely to make the implementation identical to another
repository.

Write an ADR when a decision establishes a durable boundary, invariant, data
ownership rule, compatibility contract, or dependency choice. Keep transient
implementation details in the PR instead. Capture the context, alternatives,
tradeoffs, and consequences that a future maintainer cannot infer from code.
Start from `docs/adr/TEMPLATE.md`. Its Context, Decision, Alternatives
considered, and Consequences sections are recommended review guidance, not a
schema enforced by the validator; concise records may adapt them when that
makes the decision clearer.

ADR files are named
`<DOMAIN>[-<SUBDOMAIN>]-<NNNN>-descriptive-title.md`: use at most two
broad-to-narrow tags, with the owning subsystem first, followed by the next
unused four-digit sequence number. Each ADR opens with
`# ADR-<IDENTIFIER>: Title`, a `Date: YYYY-MM-DD` line, and a `## Status`
section whose value is `Proposed`, `Accepted`, `Rejected`, `Deprecated`, or
`Superseded`. Run `python scripts/check_adrs.py`; CI and pre-commit enforce the
filename, metadata, identifier uniqueness, companion-directory naming,
non-legacy references, and ADR-link integrity. There is deliberately no
manually maintained index; list `docs/adr/` to discover records.
