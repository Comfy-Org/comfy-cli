# ADR-DESKTOP-DISCOVERY-0006: Preserve Runtime Path Identity

Date: 2026-10-25

## Status

Proposed

Relation: Builds on [ADR-AUTOMATION-0003](AUTOMATION-0003-one-versioned-machine-output-dialect.md).
The versioned envelope, additive field/error compatibility and separation of
human diagnostics from machine output remain unchanged.

## Context

Desktop separates its code checkout, data directory, interpreter and shared
model folders. Modern and legacy registries can coexist, and an existing
last-launched installation is not necessarily the process serving a workflow.
Model-path YAML and launch flags can override the remembered data/input/output
roots. Choosing another installation silently can inventory or upload unrelated
files while reporting the wrong version.

## Decision

- Discover modern `comfyui-desktop-2`, `Comfy Desktop`, and supported legacy
  records. Keep each registry's model-root provenance with its installation;
  explicit caller/test registry overrides remain isolated. On Linux, read
  `settings.json` from the `XDG_CONFIG_HOME` folder of the same name as the
  `XDG_DATA_HOME` registry, where Desktop writes it.
- Without running-server evidence, describe selection as a registry-derived
  candidate and expose its selection reason. Do not claim it identifies the
  running process.
- Server launch metadata is not evidence of a shared filesystem, even for a
  loopback endpoint (which may be a forwarder). Before using its paths to select
  a local Desktop installation or read local configurations/interpreters/roots,
  require caller-established direct-endpoint and local-process provenance. Bind
  the request to its resolved address, bypass proxies, refuse redirects, match
  the address-specific socket owner across the request using PID/create time,
  and corroborate the actual Python script/argv/cwd. On Linux, require the same
  mount namespace and process root as the CLI; inability to read that evidence
  leaves provenance unknown. The pure discovery API defaults provenance to
  unknown. Remote, forwarded, container, inaccessible or ambiguous endpoints
  retain their reported catalog/metadata but require explicit local roots.
- With verified shared-filesystem provenance, require one recorded existing
  installation to match its absolute code/data path or the full resolved generated
  config path: that record's owning registry directory plus
  `instance-model-paths/<install_id>.yaml`. A copied filename/stem is insufficient;
  retain per-record ownership across coexisting registries and reject path
  separators and Windows drive/stream separators in IDs used for config identity. This preserves bundled/shared-code
  discovery without conflating registries. Internal ownership adds no public
  output fields. Zero or conflicting matches leave selection unresolved instead of combining
  unrelated installations.
- Derive effective code/data/model/input/output roots from that selected process.
  Resolve its model-path configuration the way ComfyUI loads it: the
  `extra_model_paths.yaml` beside the `main.py` the process runs (ComfyUI takes
  it from its script's real path, which for a bundled launch is not the
  recorded checkout) first, then each `--extra-model-paths-config`;
  `--models-directory` replaces the data dir's `models/`. A script path that is
  not absolute leaves that local file unresolved. Retain external paths
  explicitly.
  Report the running version when supplied. The output root is the explicit
  launch path or that process's data/output default, with its source recorded.
- Relative data/models/input/output paths cannot be resolved without the process's cwd:
  retain uncertainty rather than anchoring them to the agent's cwd. Relative or
  unreadable model config paths remain diagnostic evidence, never a claim that
  all required model folders were found.
- Surface missing directories with existence flags and warnings. A missing
  selected workspace leaves `which`'s exit code and `ok: true` envelope as they
  were under envelope/1; `exists: false` and a `workspace_not_found` warning
  report it. A Desktop path that cannot be checked (permission denied, a
  symlink loop, a `~` with no home directory to expand) is a warning, never a
  failed command. These warning codes stay in `warnings[]` and are not
  registered error codes: under ADR-AUTOMATION-0003 registered codes are
  failure categories. Discovery is read-only: it
  creates no default workspace, installs nothing and grants no permission to
  inspect/upload every file under a shared root.
- Optional which/env discovery fields remain additive. Runtime launch reporting
  contains known path flags only, not arbitrary argv values or credentials.
  File-upload scope and approvals belong to the calling operation; discovering
  an output directory does not authorize uploading its whole contents.

## Alternatives considered

Selecting the newest registry entry alone is insufficient when a caller has
actual process evidence. Combining every existing install/model directory hides
identity errors and expands scope. Guessing relative paths from the CLI's cwd
can point at unrelated files. These approaches are rejected even when their
paths happen to exist.

## Consequences

Discovery can support both installed Desktop generations while callers retain
explicit path/selection provenance. Ambiguous, stale or relative process metadata
can require a user's verified path/cwd instead of automatic continuation.
Registry layouts and server metadata remain external compatibility surfaces;
unknown/malformed records are diagnostics rather than evidence of availability.
An interpreter is not inferred solely from a remembered data directory when
verified instance/process information is available.

## Validation

`test_desktop_install.py` covers registry isolation, actual process matching,
model-config roots, explicit/default input/output paths, relative-cwd ambiguity
and serialization. `test_which_desktop.py`, workspace-manager and Manager GUI
tests cover public reporting, missing roots and warning compatibility. Run those
tests with the worktree's package first on `PYTHONPATH`, plus Ruff lint/format,
`python scripts/check_adrs.py` and its native validator tests.

## Revisit triggers

Reopen if one running process resolves to another installation, if an unresolved
relative path is reported as verified, if server metadata bypasses direct local
process/filesystem proof, if shared folders lose provenance, or if
new Desktop layouts/launch flags require an additional documented discovery
source. A change that allows discovery to mutate installations or grants upload
permission requires a separate decision.
