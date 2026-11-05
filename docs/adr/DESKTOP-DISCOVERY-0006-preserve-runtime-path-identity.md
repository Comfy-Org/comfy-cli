# ADR-DESKTOP-DISCOVERY-0006: Preserve Runtime Path Identity

Date: 2026-10-25

## Status

Proposed

Relation: Builds on [ADR-AUTOMATION-0003](AUTOMATION-0003-one-versioned-machine-output-dialect.md).
The versioned envelope, additive field/error compatibility and separation of
human diagnostics from machine output remain unchanged.

## Context

Desktop separates its code checkout, data directory, interpreter and shared
model folders. Modern and legacy registries can coexist, and the last-launched
installation is not necessarily the process serving a workflow. Choosing
another installation silently can inventory or upload unrelated files while
reporting the wrong version.

## Decision

- Discover modern `comfyui-desktop-2`, `Comfy Desktop`, and supported legacy
  records. Keep each registry's model-root provenance with its installation;
  explicit caller/test registry overrides remain isolated. On Linux, read
  `settings.json` from the `XDG_CONFIG_HOME` folder of the same name as the
  `XDG_DATA_HOME` registry, where Desktop writes it.
- Describe the selection as a registry-derived candidate and expose its
  selection reason. Do not claim it identifies a running process. `which` and
  `env` report the candidate next to the selected workspace and never swap it
  in for install or launch.
- Selecting an installation from a running server's launch metadata is not
  part of this decision. Server metadata is not evidence of a shared
  filesystem, even for a loopback endpoint (which may be a forwarder); the
  command that first reads it records how it establishes that provenance.
- Surface missing directories with existence flags and warnings. A missing
  selected workspace leaves `which`'s exit code and `ok: true` envelope as they
  were under envelope/1; `exists: false` and a `workspace_not_found` warning
  report it. A Desktop path that cannot be checked (permission denied, a
  symlink loop, a `~` with no home directory to expand) is a warning, never a
  failed command. An adopted install whose data path cannot be read is marked
  unverified rather than reported as known. These warning codes stay in
  `warnings[]` and are not registered error codes: under ADR-AUTOMATION-0003
  registered codes are failure categories. Discovery is read-only: it creates
  no default workspace, installs nothing and grants no permission to
  inspect/upload every file under a shared root.
- Optional which/env discovery fields remain additive.

## Alternatives considered

Combining every existing install/model directory hides identity errors and
expands scope. Swapping the Desktop candidate in as the selected workspace
would change which checkout install and launch commands touch. These
approaches are rejected even when their paths happen to exist.

## Consequences

Discovery can support both installed Desktop generations while callers retain
explicit path/selection provenance. A caller that must know the running
process's roots still needs explicit paths or its own verified evidence.
Registry layouts remain external compatibility surfaces; unknown/malformed
records are diagnostics rather than evidence of availability.

## Validation

`test_desktop_install.py` covers registry isolation, model roots, the Linux
XDG split, unreadable records and paths, and serialization.
`test_which_desktop.py`, workspace-manager and Manager GUI tests cover public
reporting, missing roots and warning compatibility. Run those tests with the
worktree's package first on `PYTHONPATH`, plus Ruff lint/format,
`python scripts/check_adrs.py` and its native validator tests.

## Revisit triggers

Reopen if a registry-derived candidate is reported as the running process, if
an inferred path is reported as verified, if shared folders lose provenance,
or if new Desktop layouts require an additional documented discovery source. A
command that selects an installation from running-server metadata needs its
own decision on how it proves local process and filesystem provenance. A
change that allows discovery to mutate installations or grants upload
permission requires a separate decision.
