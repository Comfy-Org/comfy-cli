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

### Fixed

- `comfy … | head` exits 0 again, and `--json` keeps stderr clean, on typer
  >= 0.24. typer now runs on a vendored copy of click, so the broken-pipe guard
  no longer recognized the stdout wrapper click installs on EPIPE and reported a
  genuine failure instead. The wrapper is now matched by class name and defining
  module, which also stops the deprecated `click.utils.PacifyFlushWrapper`
  import from printing a warning onto stderr.
- `comfy --help-json` lists `choices` for enum options again, for the same
  reason: the vendored param types are not instances of the installed click's
  `Choice`, so the choice list was silently dropped.
- Promoted subgraph widgets are edited where the frontend reads them. The
  frontend (ADR 0009) keeps a promoted widget's value on the HOST instance
  (`widgets_values` positional over the widget-backed subgraph inputs) and
  runs that value over the interior default; `set-widget 57.width` used to
  follow the legacy `proxyWidgets` route into the interior node, so the edit
  neither ran nor showed on the canvas. `set-widget`, `set-slot` and `vary`
  now write the host value for `<instance>.<input>`, redirect an interior
  address that backs a promotion (`57/13.width`) to the same host value
  (`redirected_from` on the op), follow an outside link feeding the promoted
  input to its primitive (`PrimitiveInt`/`PrimitiveString*`/legacy
  `PrimitiveNode`, through `Reroute`s) and refuse — naming the driver — when
  a non-primitive node computes the value. Unpromoted interior widgets still
  write the definition. The op carries `promoted.value_index` and the
  materialized `host_widgets_values`.
- `comfy workflow connect` wires an outside node to a promoted widget
  (`PrimitiveInt.INT → 57.width`): the definition declares the input, so it
  is materialized on the instance (with the frontend's `widget` marker) and
  linked, type-checked against the declared input type.
- `comfy workflow slots` advertises promoted widgets at the instance address
  with the value the frontend runs (host value, else interior default), flags
  widgets fed by a link (`linked_from`), keeps unpromoted interior widgets
  reachable at `<instance>/<inner>.<widget>` (nested instances included), and
  no longer advertises the interior address behind a promotion.
- `convert_ui_to_api` (`comfy run`, `validate`) applies host-owned promoted
  values onto the expanded interior nodes — a post-migration template whose
  host prompt differs from the interior default (`audio_minimax_music_3`:
  interior caption `''`) was submitted with the interior value. Precedence
  matches the frontend: outside link, then host value, then interior.
- `comfy workflow connect` can wire a Load Image / Load Video / Load Audio
  into an auto-grow group nested under a dynamic combo (`GeminiNanoBanana2V2`
  `model.images`, `MinimaxHailuo03ReferenceNode` / `ByteDance2ReferenceNodeV2`
  `model.reference_images` / `reference_videos` / `reference_audios`, and the
  other 27 partner nodes shaped that way). Slots are resolved from the node's
  schema for its current selection — not from a pre-existing input — so an
  agent-built node grows `model.images.image_1`, `image_2`, … by name or by
  addressing the group base, a UI-built node reuses its free pre-created slot
  and then keeps growing, a UI-built top-level group (`BatchImagesNode`) can
  be base-addressed and grown past its free slot, a wrong element type is
  refused (`type mismatch`), and a group never grows past the schema's
  `names` length / `max`. `comfy nodes show` now lists every dynamic-combo
  option's sub-inputs and names each auto-grow group's element type, slot
  vocabulary and first keys to wire.
- The widget order no longer counts a dynamic combo's link-only sub-inputs
  (auto-grow groups, `GEMINI_INPUT_FILES`, …) as `widgets_values` slots.
  `add-node` wrote them as phantom `null` values and the published widget
  catalog named them, so every widget after the groups was one or more slots
  off: an agent-built Nano Banana 2 converted with `seed` in
  `response_modalities`, and through the CRDT doc host a UI-built MiniMax H3
  node's `seed` position mapped onto `model.reference_images`. `add-node`,
  the catalog and `set-widget` now share one walk, so a fresh node's layout
  is exactly what `set-widget` indexes and what the frontend serializes (41
  classes in the cloud catalog change width).
- The widget order (`comfy nodes widget-catalog`, `set-widget` indexing, the
  UI→API converter) now names every slot the frontend serializes: the
  `upload` button frontend extensions inject on media loaders (`LoadImage`,
  `LoadImageMask`, `LoadVideo`, `LoadAudio`, ...), the `audioUI` player on
  the audio family, the `PREVIEW_3D` `image` on `SaveGLB`/`Preview3D`, DOM
  widgets declared under an uppercase custom type (`Load3D.image`), and
  inputs whose `widgetType` overrides a link-shaped socket type
  (`LTXVEmptyLatentAudio.frame_rate`, the "Basic data handling" math nodes).
  Before, a workflow with any of these nodes carried more `widgets_values`
  than the catalog could name, so the cloud doc host refused to mint it
  (`createNodeMap(LoadImage): widgets_values has 2 entries but widget_order
  names only 1`) and `set-widget`/conversion read the values after such a
  slot one position off.
- `comfy generate <model>`, `comfy generate resume` and sync-mode creates now
  emit the `envelope/1` contract in `--output json` / `ndjson` modes instead
  of a bare partner blob: the partner payload is wrapped as `data.result`
  (verbatim) with `data.saved` listing `--download` artifacts, and the
  payload schema is registered as `comfy generate` → `generate_result.json`
  so `comfy discover` advertises it. Pretty mode with a tail `--json` keeps
  the legacy raw blob.

### Added

- `comfy workflow add-node` and an `add_node` op in `comfy workflow apply`
  refuse a class the catalog marks deprecated (`node_deprecated`), naming the
  live class with the same display name when there is one. Pass
  `--allow-deprecated` (or `"allow_deprecated": true` on the op) to add it
  anyway.
- `comfy nodes search` and `comfy nodes ls` hide deprecated classes by
  default; `--include-deprecated` shows them.

- `comfy knowledge pick` attaches each pick's model `fits` block (VRAM per
  variant, credit rate, max refs) when the bundle carries one, so a size or
  price constraint can be checked against a number rather than the caveat text.
- `comfy-build`, the skill for building a custom ComfyUI environment on the
  developer platform, is now bundled with the CLI. `comfy skills show
  comfy-build` works, and an argument-free `comfy skills install` writes it on a
  machine with no network.
- `comfy build init --from-workflow <workflow.json>` and `comfy build update
  --from-workflow <workflow.json>` read a ComfyUI workflow, in the editing
  format or the API export, into the local spec.
- `comfy build init --base-image <id>` and `comfy build update --base-image
  <id>` choose the curated base image a build is built on — the CUDA, Python
  and torch runtime — instead of leaving it to the catalog default.
  `comfy build refs base-images` lists the ids.
- A workflow import prints its full report: the node classes nothing provides,
  the closest pack the registry named for each one, every model the graph loads
  (a workflow import carries none of them), the classes served by a partner API,
  and whether a ComfyUI version still has to be pinned.
- `comfy build pull` names what it would change before changing it: the same
  definition diff `comfy build update` prints, echoed in the confirmation and
  carried in the `--json` payload as `summary` and `diff`. A fetched Build that
  omits `models` or `customNodes` drops the local entries, and the diff is where
  that is now visible. `comfy build pull --dry-run` prints the diff and writes
  nothing — with `--yes` the payload only arrives after the write, so this is
  how a non-interactive caller reads the diff before deciding.
- `CONTRIBUTING.md` (renamed from `DEV_README.md`) and this changelog.
- `comfy deploy` — run a Build release as a serverless endpoint: `up`, `status`,
  `ls`, `show`, `logs`, `events`, `scale`, `stop`, `start`, `delete`, `run`, and
  `refs compute`.

### Changed

- `comfy skills install` no longer fetches any skill over the network.
  `comfy-build` was the only one it fetched, and it now ships in the wheel and
  is versioned with the CLI release, so the skill and the commands it describes
  can no longer drift apart.
- The builder client module is now `comfy_cli.builder_api` (was
  `comfy_cli.distribution_api`), and its methods say build and release
  (`create_build`, `create_release`, `list_releases`, ...), matching the
  builder's public API. The `distribution-definition/0` schema id is unchanged.
- **Breaking:** `comfy build --json` payloads say `buildId`, `releaseId`,
  `builds` and `releases`, and nothing else: the retired `distributionId`,
  `versionId`, `distributions` and `versions` keys are gone from every payload,
  every error `details`, and every shipped schema rather than being carried
  alongside. Read the builder's own spelling.
- **Breaking:** `import comfy_cli.distribution_api` no longer works. The
  deprecation shim is removed together with the surface it shimmed; import
  `comfy_cli.builder_api`.
- **Breaking:** `comfy build` is restructured around a local `comfy-build.yaml`
  spec — `init`, `push`, `pull`, `status`, `ls`, `show`, `validate`, `update`,
  `delete`, plus `release`, `refs`, and `blob` subgroups. The `comfy distribution`
  alias and the `scan` / `create` / `version` / `artifact download` /
  `from-snapshot` / `from-workflow` commands it fronted are removed.
  `from-workflow` returns as the `comfy build init --from-workflow` and
  `comfy build update --from-workflow` options described under Added.
- **Breaking:** the read verbs are renamed, with no aliases left behind:
  `comfy build list` → `comfy build ls`, `comfy build get` → `comfy build show`,
  and `comfy build blob list` → `comfy build blob ls`. The reference lookups
  (`resolve`, `base-images`, `build-targets`, `model-dirs`) move under
  `comfy build refs`, and `comfy build blob upload` is removed — `comfy build push`
  uploads local models and nodes from the spec.
- **Breaking:** the `build_upload_unavailable` error code is retired. It was only
  ever raised by the removed `create` path, so nothing emits it and it no longer
  appears in `comfy discover`. This is the one exception to the append-only rule
  in `comfy_cli/schemas/error_codes.md`: the code is retired, never reused.
- **Breaking:** a `--json` run of `comfy build` or `comfy deploy` is never
  prompted, even from a terminal. A confirmation or missing required option now
  returns the matching refusal envelope — `*_needs_confirm`, `*_missing_input`,
  or `build_id_unknown` where a Build id could not be resolved — and exits 1,
  where it previously opened a TUI prompt on the same stream the envelope is
  written to. Pass `--yes` or the option itself to proceed non-interactively.
  Other command families still prompt under `--json`; they do not route through
  `comfy_cli.interaction`, and `--skip-prompt` remains the way to suppress them.
- **Breaking:** the global `--skip-prompt` now applies to `comfy build delete`,
  which previously ignored it. Combined with a non-agentic caller it accepts the
  delete confirmation, matching `build pull` and `build update`.
- **Breaking:** packaging a local custom node is all-or-nothing. Anything under
  `custom_nodes/<node>/` that cannot be read now fails `init` / `update` /
  `status` / `push` / `pull` with one `build_spec_invalid` envelope naming the
  node directory. Previously the two failure modes diverged and neither was
  usable: an unreadable **directory** was silently dropped from the archive
  whose digest becomes the node's committed `localDigest`, while an unreadable
  **file** escaped as an uncaught `PermissionError` — a traceback with no
  envelope at all, even under `--json`.
- Symlinks inside a custom node are still excluded from its archive, but are no
  longer excluded in silence: `init`, `update`, `push` and `pull` name them on
  stderr and carry them in a `skipped_symlinks` payload key, including on a
  `--dry-run` that writes nothing. `status` rescans but publishes no definition
  to point into, so for it the stderr warning is the whole report.

### Fixed

- `comfy build ls` and `comfy build release ls` show every row. The builder
  pages both reads, and the client took only the first page, so a workspace or
  a build past one page lost its tail — silently, with no error and nothing in
  the output to say rows were missing.
- The `--from-snapshot` path is no longer sent to analytics. It is a local
  filesystem path naming the user's home directory and their install layout,
  and it was shipped verbatim: the URL scrubber only strips credentials out of
  URLs and returns a bare path untouched, so the key has to be named in the
  redaction set, and the rename from `--from` had left it out.
- A misconfigured `COMFY_DEPLOY_URL` is reported as a `deploy_insecure_url`
  error instead of a traceback. The https guard signalled refusal with a bare
  `ValueError`, which no deploy command listed in its `except` tuple, so the
  failure escaped the command layer and `--json` printed no envelope at all.
  The message now names the setting actually in play rather than
  `COMFY_CLOUD_BASE_URL`.
- `comfy deploy ls` cannot hang on a defective pagination cursor. The loop
  walked whatever `nextCursor` came back until it was falsy, so a repeated
  cursor spun forever on an ever-growing list. A repeated cursor and a run past
  the page ceiling are both reported as `deploy_server_error` now.
- `comfy deploy run` validates a job output's `node_id`, `type` and `id` before
  it downloads the file rather than after. A malformed response used to leave
  files in `outputs/` that the result envelope then never accounted for,
  followed by exit 1 and no manifest.
- `comfy deploy show`, `status` and `stop` refuse a blank release id instead of
  matching on it. A lax local copy of the shared field validator accepted the
  empty string, so a release with a blank `id` adopted every deployment whose
  `releaseId` was also blank as belonging to that Build — which could point a
  lifecycle mutation at the wrong deployment.
- An asset upload's `Content-Length` always describes the bytes that follow it.
  The length came from a `stat()` at request-build time while the body was
  opened and read later, when urllib got round to consuming it; both now come
  from a single open handle, and the body is bounded to exactly the declared
  size.

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
