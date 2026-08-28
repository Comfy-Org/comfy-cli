# Changelog

All notable changes to `comfy-cli` are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

This file starts at **v1.11.0**, the first release with curated release notes.
Entries below are condensed from the published GitHub releases; each links to
the full notes. For **v1.10.5 and earlier**, the GitHub releases carry only
auto-generated "What's Changed" pull-request lists rather than curated notes, so
they are not transcribed here — see the
[releases page](https://github.com/Comfy-Org/comfy-cli/releases) for that
history.

## [Unreleased]

### Added

- `comfy workflow add-node` and an `add_node` op in `comfy workflow apply`
  refuse a class the catalog marks deprecated (`node_deprecated`), naming the
  live class with the same display name when there is one. Pass
  `--allow-deprecated` (or `"allow_deprecated": true` on the op) to add it
  anyway.
- `comfy nodes search` and `comfy nodes ls` hide deprecated classes by
  default; `--include-deprecated` shows them.

- `comfy-build`, the skill for building a custom ComfyUI environment on the
  developer platform, is now bundled with the CLI. `comfy skills show
  comfy-build` works, and an argument-free `comfy skills install` writes it on a
  machine with no network.
- `comfy build from-workflow --from <workflow.json> --name <name>` creates a
  build from a ComfyUI workflow, in the editing format or the API export.
- A workflow import prints its full report: the node classes nothing provides,
  the closest pack the registry named for each one, every model the graph loads
  (a workflow build carries none of them), the classes served by a partner API,
  and whether a ComfyUI version still has to be pinned.
- `CONTRIBUTING.md` (renamed from `DEV_README.md`) and this changelog.

### Changed

- `comfy skills install` no longer fetches any skill over the network.
  `comfy-build` was the only one it fetched, and it now ships in the wheel and
  is versioned with the CLI release, so the skill and the commands it describes
  can no longer drift apart.

- The builder client module is now `comfy_cli.builder_api` (was
  `comfy_cli.distribution_api`), and its methods say build and release
  (`create_build`, `list_releases`, ...), matching the builder's public API.
  The `distribution-definition/0` schema id is unchanged.
- `comfy build --json` payloads carry the builder's vocabulary: `buildId`,
  `releaseId`, and `builds` and `releases` arrays. The retired `distributionId`,
  `versionId`, `distributions` and `versions` keys are emitted alongside them
  with identical values, so a pinned script keeps parsing. The shipped schema
  filenames are unchanged.

### Deprecated

- `import comfy_cli.distribution_api` still works for one release and warns;
  import `comfy_cli.builder_api` instead.
- The `distributionId`, `versionId`, `distributions` and `versions` keys in
  `comfy build` `--json` output will be removed after one release; read
  `buildId`, `releaseId`, `builds` and `releases` instead. The six schemas that
  declare them mark each retired key deprecated in its description.

### Fixed

- The shipped `build_from_snapshot.json` schema now requires the `build` key
  the builder actually serves; it still required the pre-rename `distribution`
  key, so a valid `comfy build from-snapshot --json` payload failed validation.

## [1.16.0] - 2026-08-10

[Full notes](https://github.com/Comfy-Org/comfy-cli/releases/tag/v1.16.0) · 16 commits since v1.15.0. No breaking changes.

### Fixed

- Failed cloud jobs report their cause again: `jobs status` / `jobs watch` were
  reading field names `/api/jobs/<id>` does not serve, so every failure surfaced
  as an empty error. (#683)
- `jobs watch` streams live progress — it now attaches as the submitting client
  and understands the per-step `progress_state` message. (#693)
- Local `models search --text` matches token-wise and separator-insensitively,
  so `--text "sdxl base"` finds `sd_xl_base_1.0.safetensors`. (#684)
- `nodes path` constrains hops by source type and no longer claims exactness
  unconditionally. (#695)
- `comfy outdated` prefers the highest stable semver tag known to the local
  checkout over a mis-set GitHub `releases/latest` flag. (#694)
- `nvidia-smi` is resolved by absolute path in the CUDA probe, blocking Windows
  CWD planting. (#641)
- Registry failures are typed and carry a machine-readable error code. (#528)

## [1.15.0] - 2026-08-05

[Full notes](https://github.com/Comfy-Org/comfy-cli/releases/tag/v1.15.0) · 12 commits since v1.14.0. No breaking changes.

### Added

- `comfy stop --port <p>` — verified stop of a local ComfyUI this CLI did not
  launch. (#675)
- `comfy update --exit-on-fail` — a failed pack update exits non-zero. (#676)
- `error_code` on `jobs ls` rows. (#677)
- `text_outputs` in local `jobs status`. (#550)
- `partner_nodes_detected` telemetry, and `caller_kind` stamped on every event.
  (#647)

### Fixed

- Empty COMBO enums no longer skip validation — a loader with an empty option
  list (`UNETLoader`, `CLIPLoader` on an install with no models) was treated as
  unconstrained, so its missing model was never reported. (#680)
- An explicit `--port 0` is no longer swallowed by a `--host host:port` value.
  (#679)
- `validate` enforces promoted hard checks only on output-reachable nodes. (#565)
- Apple Silicon is detected under Rosetta 2 in the GPU probe. (#568)

## [1.14.0] - 2026-08-05

[Full notes](https://github.com/Comfy-Org/comfy-cli/releases/tag/v1.14.0) · 89 commits since v1.13.0. No breaking changes.

### Added

- `comfy system-stats` / `comfy free` — ComfyUI `/system_stats` and `/free`
  passthrough. (#626)
- `comfy workflow notes <path>` — read `Note` / `MarkdownNote` text out of a
  workflow, offline. (#611)
- `comfy node deps` — per-pack Python dependency report; `--registry <node-id>`
  pre-checks a not-yet-installed pack. (#610, #625)
- `comfy model download --background`, plus `download-status`, `downloads`, and
  `download-cancel`. (#607)
- `comfy update comfy --version <X>` — headless ComfyUI version switch/rollback
  with tag validation. (#606)
- `comfy templates check` — per-template runnable / missing / api-required
  verdict. (#557)
- `comfy upload --host/--port` — local target routing for uploads. (#648)
- `envelope/1` from `comfy launch` / `comfy stop` (#588) and `generate list` /
  `generate schema` (#621); envelope errors on every `model download` and
  `generate` failure path (#581, #601).
- `comfy run --allow-spend` gate on paid partner nodes; `validate` reports
  `partner_nodes` / `spends_credits`. (#591, #590)

### Fixed

- A dead local server is attributable to the job that killed it: `jobs status`
  falls back to the on-disk state file when the server is down (#602),
  `run --wait` names the `prompt_id` on disconnect (#605), the watcher records a
  terminal `server_died` (#604), and `jobs status` consults that record even
  when the server is back up (#674).
- `models search --text` walks every model folder, not just `checkpoints`. (#603)
- `comfy env` detects legacy ComfyUI-Manager clones and reconciles a stale
  `manager_gui_mode`. (#609)
- `comfy logs` resolves the right file, with a `user/comfyui.log` fallback and
  staleness metadata. (#608)
- Model downloads are atomic — streamed to a `.part` sibling and renamed on
  completion. (#666)
- `validate` / `nodes` read the same server `run` submits to instead of always
  consulting `127.0.0.1:8188`. (#667)
- `nodes search` is tokenized and order-independent. (#646)
- `jobs ls` state-file rows are scoped to the resolved `--where` target, with
  `--all` to opt out (#582); `jobs cancel` emits `prompt_not_found` for an
  unknown local id (#580).

### Security

- Server-supplied text is stripped of ANSI/control sequences and escaped before
  reaching Rich markup sinks. (#614, #627, #655)
- Absolute-path resolution for probe binaries blocks Windows CWD planting (#567);
  shared read cap on unbounded HTTP body reads (#654); authed urllib routed
  through a no-redirect opener (#530).

## [1.13.0] - 2026-07-28

[Full notes](https://github.com/Comfy-Org/comfy-cli/releases/tag/v1.13.0) · 41 commits since v1.12.0.

### Added

- `comfy cloud login` is agent-drivable — `--json` mode emits a machine-readable
  `login_url` event as soon as the authorize URL exists.
- `comfy run-template` — fetch a template, fill its params, spend-gate, and run
  to completion in one verb.
- `comfy outdated` — read-only version check for ComfyUI core and installed node
  packs.
- `comfy run --prompt` / `--set` — local text2img from the bundled default
  workflow.
- `comfy generate` spend gate: explicit consent before spending credits.
- `COMFY_LOCAL_URL` is honored for the local ComfyUI address; the `:8188`
  hardcode is gone.

### Changed

- `comfy generate` derives partner model enums from the active OpenAPI spec
  rather than a pinned list.
- `comfy validate` auto-converts UI-format workflows, matching `comfy run`.
- `get_job_status` migrated off the deprecated `/api/job/<id>/status` endpoint.
- Telemetry providers are built lazily so `comfy install` can upgrade
  `pydantic_core`; `mixpanel<5` pinned so `comfy install` cannot wedge
  `pydantic-core` on Windows.

### Fixed

- Downloads verify `Content-Length` and land atomically via `.part` rename; the
  extension taken from an untrusted `?filename=` param is sanitized.
- `_http_request` detects oversize responses instead of silently truncating.
- Widget-aware dynamic-combo expansion and name-aligned control-marker filtering
  in `workflow_to_api`.
- `validate` presence-checks required inputs, adds a no-outputs check, and
  hard-errors range violations.
- The OAuth session is refreshed during local partner-credential injection; the
  run WebSocket is closed on every exit path of local `--wait`.
- Telemetry network I/O is bounded so it cannot outlive the run envelope.
- Square brackets are escaped in Typer help strings so choice lists render.

## [1.12.0] - 2026-07-07

[Full notes](https://github.com/Comfy-Org/comfy-cli/releases/tag/v1.12.0)

### Added

- `comfy logs` — background ComfyUI logs are persisted to
  `<workspace>/user/comfyui_<port>.log` and readable via `comfy logs --tail N`.
  (#491)
- Local saved-workflow support: the saved-workflow verbs work with
  `--where local` via ComfyUI's `/userdata`. (#486)
- ROCm 7.2 support, set as the default. (#476)

### Changed

- `comfy download` copies on-disk local outputs instead of refusing them. (#485)
- Shared IPv6-aware `host:port` resolver for `comfy run` (#488); redirect-refusal
  and SSRF loopback guards consolidated into `comfy_cli/http.py` (#487, #482);
  the `where` default config read centralized via `where.resolve_default()`
  (#477).

### Fixed

- `comfy launch --background` no longer crashes on Python 3.14's removed
  implicit event loop. (#481)

## [1.11.1] - 2026-06-22

[Full notes](https://github.com/Comfy-Org/comfy-cli/releases/tag/v1.11.1)

The first PyPI-published cut of the agent-first line. `v1.11.0` was tagged but
did not publish to PyPI due to a release-trigger mismatch; `v1.11.1` is
identical code.

## [1.11.0] - 2026-06-22

[Full notes](https://github.com/Comfy-Org/comfy-cli/releases/tag/v1.11.0)

The agent-first release: an agent or a human can build, validate, run, and
review image/video/audio workflows on a local server or Comfy Cloud entirely
from the terminal. Additive and backward-compatible for interactive use.

### Added

- One machine contract — every command emits the same versioned JSON envelope,
  and every error carries a registered code plus an actionable hint (`--json`,
  automatic on non-TTY).
- A compile model for workflows: typed fragments wired by YAML blueprints via
  `comfy workflow compose`, with `comfy workflow decompose` as the inverse.
- Pre-flight validation (CQL): `comfy validate` checks a graph against the live
  server's `object_info` before you spend.
- Async-by-default execution with `comfy jobs`, including `comfy jobs wait
<id…>` to block on a whole batch.
- Projects — a `project/1` layout with content-addressed `assets push`, a run
  journal, and `--where` routing.
- `comfy preview` — image to thumbnail, video to contact sheet, audio to
  waveform.
- Bundled skills (`comfy skills`) that teach agents to operate, build, debug,
  and present the CLI.
- `comfy setup` — a guided onboarding wizard, surfaced from the welcome screen
  and a first-run nudge.

### Changed

- **Behavior change:** when output is piped or redirected (non-TTY), commands
  with structured output default to JSON instead of text. Pass `--no-json` or
  set `COMFY_OUTPUT=pretty` for the old piped text. Interactive terminal use is
  unchanged.
- `comfy install` / `comfy update` no longer assume the workspace interpreter
  ships pip; a pip-less uv-managed venv is bootstrapped automatically.

## Earlier releases

`v1.10.5` and earlier are published on the
[releases page](https://github.com/Comfy-Org/comfy-cli/releases) with
auto-generated pull-request lists.

[Unreleased]: https://github.com/Comfy-Org/comfy-cli/compare/v1.16.0...HEAD
[1.16.0]: https://github.com/Comfy-Org/comfy-cli/compare/v1.15.0...v1.16.0
[1.15.0]: https://github.com/Comfy-Org/comfy-cli/compare/v1.14.0...v1.15.0
[1.14.0]: https://github.com/Comfy-Org/comfy-cli/compare/v1.13.0...v1.14.0
[1.13.0]: https://github.com/Comfy-Org/comfy-cli/compare/v1.12.0...v1.13.0
[1.12.0]: https://github.com/Comfy-Org/comfy-cli/compare/v1.11.1...v1.12.0
[1.11.1]: https://github.com/Comfy-Org/comfy-cli/compare/v1.11.0...v1.11.1
[1.11.0]: https://github.com/Comfy-Org/comfy-cli/compare/v1.10.5...v1.11.0
