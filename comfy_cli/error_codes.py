"""Source of truth for the JSON envelope's ``error.code`` values.

Every code raised by ``renderer.error(code=…)`` must appear here. Two tests
enforce this both ways:

  - ``tests/comfy_cli/output/test_error_code_registry.py``:
      every raised code is registered
      every registered code is raised somewhere

That makes this module the canonical contract for agents. Agents fetch the
list via ``comfy discover`` and branch on the codes; if you rename, deprecate,
or remove one, you're breaking the contract and the tests fail before merge.

Codes are snake_case and match ``^[a-z][a-z0-9_]*$``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

CODE_PATTERN = re.compile(r"^[a-z][a-z0-9_]*$")


@dataclass(frozen=True)
class ErrorCode:
    code: str
    meaning: str
    hint: str | None = None


# ---------------------------------------------------------------------------
# The registry.
#
# Ordered roughly by subsystem so a reader can scan a logical neighborhood.
# Appendable; do not repurpose an existing code.
# ---------------------------------------------------------------------------

REGISTRY: tuple[ErrorCode, ...] = (
    # --- output / cancellation / lifecycle -----------------------------------
    ErrorCode(
        "cancelled",
        "User pressed Ctrl-C; in-flight work was torn down.",
    ),
    ErrorCode(
        "not_in_workspace",
        "Resolved no workspace where one was required (e.g. `comfy which`).",
        "run `comfy install`, or pass `--workspace`",
    ),
    # --- workflow loading ----------------------------------------------------
    ErrorCode(
        "workflow_not_found",
        "The `--workflow` path doesn't exist or isn't readable.",
        "check the path",
    ),
    ErrorCode(
        "workflow_invalid_json",
        "The file at `--workflow` failed JSON parsing.",
        "re-export the workflow from ComfyUI (File > Export (API))",
    ),
    ErrorCode(
        "workflow_not_api_format",
        "Loaded JSON isn't API-format and no converter is available.",
        "use ComfyUI's `File > Export (API)`",
    ),
    ErrorCode(
        "workflow_read_error",
        "Workflow file exists but isn't readable as UTF-8 text (OSError / UnicodeDecodeError).",
        "check file permissions and encoding",
    ),
    ErrorCode(
        "workflow_write_error",
        "`workflow get --out` could not write the fetched workflow to disk (OSError: permissions, "
        "missing parent dir, full disk, invalid path).",
        "check the --out path is writable and the disk has space",
    ),
    ErrorCode(
        "workflow_too_large",
        "A local ComfyUI `/userdata` response exceeded the in-memory read cap, so the CLI refused to "
        "truncate it into a corrupt/partial file. `details.limit_bytes` carries the cap.",
        "the saved workflow is unexpectedly large; inspect it directly on the server",
    ),
    ErrorCode(
        "workflow_content_not_json",
        "`workflow get` fetched content that isn't parseable JSON (non-UTF-8 bytes or a non-JSON body such "
        "as an HTML error page); the raw bytes were still written. Surfaced in `data.warnings[]`, not as an "
        "error envelope, so the command still succeeds.",
        "verify the id points at a real saved workflow, not a stray file, on the local server",
    ),
    # --- local server / WebSocket --------------------------------------------
    ErrorCode(
        "server_not_running",
        "Local ComfyUI server isn't reachable on host:port.",
        "run `comfy launch`",
    ),
    ErrorCode(
        "connection_error",
        "Could not connect to the ComfyUI server.",
        "check the server is running and the host:port is correct",
    ),
    ErrorCode(
        "ws_disconnected",
        "WebSocket dropped and reconnect failed mid-execution.",
        "check the server is still running; re-run the command",
    ),
    ErrorCode(
        "ws_timeout",
        "WebSocket idle past `--timeout` while waiting for the server.",
        "re-run with a larger `--timeout` (e.g. `--timeout 300`)",
    ),
    ErrorCode(
        "prompt_rejected",
        "Server returned 400. `details.node_errors` carries the per-node errors.",
        "inspect `details.node_errors` and fix the workflow",
    ),
    ErrorCode(
        "client_error",
        "Server rejected the request with an HTTP 4xx that isn't a validation failure "
        "(401/403/429/…). `details.status` and `details.body` carry the response.",
        "check `details.body` for the server's message",
    ),
    ErrorCode(
        "server_error",
        "Server returned an HTTP 5xx while submitting the workflow. "
        "`details.status` and `details.body` carry the response.",
        "check the ComfyUI server logs",
    ),
    ErrorCode(
        "invalid_response",
        "Server returned HTTP 2xx but the body was unparseable or lacked a `prompt_id`.",
        "check that the host:port really is a ComfyUI server",
    ),
    ErrorCode(
        "object_info_unavailable",
        "`/object_info` returned an HTTP error, or an HTTP 200 with an unparseable body. "
        "`details.status` and `details.body` carry the response.",
        "check the ComfyUI server logs; restart the server",
    ),
    ErrorCode(
        "prompt_not_found",
        "Asked about a prompt_id the server doesn't know.",
        "`comfy jobs ls` to find a valid prompt_id",
    ),
    ErrorCode(
        "partner_node_requires_credential",
        "Workflow uses a partner-API node (category `partner/*` — Veo, Kling, BFL, Gemini, etc.) "
        "but no `api_key_comfy_org` credential is available. Local submit would succeed at /prompt "
        "and then fail opaquely at execute time with `Unauthorized: Please login first`.",
        "re-submit with `--where cloud` (the CLI auto-injects the credential there), or run "
        "`comfy auth set comfy-cloud-api-key --key …` so the local submit path can inject it too",
    ),
    ErrorCode(
        "workflow_empty",
        "Workflow JSON is an empty object (no nodes).",
        "add at least one node to the workflow",
    ),
    ErrorCode(
        "default_workflow_unavailable",
        "`comfy run --prompt`/`--set` could not load the bundled default text2img graph "
        "(missing or corrupt package data). A packaging fault, not user input.",
        "reinstall comfy-cli",
    ),
    ErrorCode(
        "conversion_error",
        "UI-format workflow could not be converted to API format.",
        "export your workflow from ComfyUI via 'File > Export (API)' and retry",
    ),
    ErrorCode(
        "conversion_crash",
        "UI-format workflow conversion crashed unexpectedly.",
        "export your workflow from ComfyUI via 'File > Export (API)' and retry",
    ),
    ErrorCode(
        "template_not_found",
        "The requested workflow template was not found.",
        "check the template name and try again",
    ),
    ErrorCode(
        "gallery_load_failed",
        "Failed to load the workflow gallery.",
        "check your network connection and try again",
    ),
    ErrorCode(
        "gallery_fetch_failed",
        "Failed to fetch gallery data from the remote server.",
        "check your network connection and try again",
    ),
    ErrorCode(
        "workflow_unknown_nodes",
        "Workflow references class_type(s) not present in the server's object_info. "
        "`details.unknown_nodes` lists each with close_matches.",
        "fix the class_type names; install missing custom nodes",
    ),
    # --- routing / cloud / auth ---------------------------------------------
    ErrorCode(
        "where_invalid",
        "`--where` value was neither `local` nor `cloud`.",
        "use `--where local` or `--where cloud`",
    ),
    ErrorCode(
        "cloud_not_configured",
        "`--where cloud` requested without a stored session.",
        "run `comfy cloud login`",
    ),
    ErrorCode(
        "cloud_unauthorized",
        "Cloud rejected the bearer token (missing / expired / invalid).",
        "run `comfy cloud login`",
    ),
    ErrorCode(
        "cloud_http_error",
        "Cloud returned a non-2xx HTTP error. `details.status` carries the code.",
        "check `details.body` for the server's message",
    ),
    ErrorCode(
        "cloud_timeout",
        "Cloud wait_for_completion exceeded `--timeout`.",
        "raise `--timeout`, or `comfy jobs watch <id> --where cloud`",
    ),
    ErrorCode(
        "partial_execution",
        "The cloud reported `completed` but returned outputs for fewer output "
        "nodes than were submitted — branch(es) were pruned server-side (likely "
        "failed validation). Surfaced as a non-fatal warning in `data.warnings`.",
        "inspect the pruned branch's inputs; validate with `comfy --json validate` "
        "against `--where cloud` before re-running",
    ),
    # --- models / templates introspection ------------------------------------
    ErrorCode(
        "invalid_argument",
        "An argument intended for a URL path failed safe-path validation.",
        "use only alphanumerics, `_`, `-`, or `.` in path-segment arguments",
    ),
    ErrorCode(
        "folder_not_found",
        "Cloud or local server returned 404 for the requested model folder.",
        "list available folders via `comfy models list-folders`",
    ),
    ErrorCode(
        "model_not_found",
        "No asset matched the requested model name exactly. `details.close_matches` lists substring hits.",
        "use `comfy models search --text <substring>` to find candidates",
    ),
    ErrorCode(
        "models_show_local_unsupported",
        "`comfy models show` needs the cloud asset catalog; local servers don't have one.",
        "for local filename listing use `comfy models list-folder <folder>`",
    ),
    ErrorCode(
        "template_fetch_failed",
        "Fetching the per-template workflow JSON from `Comfy-Org/workflow_templates` failed.",
        "check network; if 404, the gallery and templates dir are out of sync — report upstream",
    ),
    ErrorCode(
        "template_workflow_invalid_json",
        "Upstream `templates/<name>.json` was not parseable JSON.",
        "report at https://github.com/Comfy-Org/workflow_templates/issues",
    ),
    ErrorCode(
        "cancel_failed",
        "`comfy jobs cancel` could not reach the local server to cancel the prompt.",
        "check the server is still running on the host/port",
    ),
    # --- auth (provider keys + cloud session intertwined) --------------------
    ErrorCode(
        "auth_invalid_key",
        "Missing or empty `--key` on `comfy auth set`.",
        "pass `--key <KEY>`",
    ),
    ErrorCode(
        "auth_not_found",
        "Tried to remove a provider with no stored key.",
        "`comfy auth list` to see what's stored",
    ),
    ErrorCode(
        "auth_not_signed_in",
        "Action requires a Comfy Cloud session.",
        "run `comfy cloud login`",
    ),
    ErrorCode(
        "auth_use_login_for_cloud",
        "`comfy auth set comfy-cloud` is no longer the cloud auth path.",
        "use `comfy cloud login`",
    ),
    ErrorCode(
        "auth_use_logout_for_cloud",
        "`comfy auth remove comfy-cloud` is no longer the cloud signout path.",
        "use `comfy cloud logout`",
    ),
    # --- oauth ---------------------------------------------------------------
    ErrorCode(
        "oauth_register_failed",
        "Dynamic client registration (RFC 7591) failed.",
        "check that the cloud server is reachable",
    ),
    ErrorCode(
        "oauth_authorize_failed",
        "OAuth authorization step failed (user denied, state mismatch, etc.).",
        "re-run `comfy cloud login`",
    ),
    ErrorCode(
        "oauth_token_failed",
        "OAuth token exchange failed.",
        "re-run `comfy cloud login` to start a fresh authorization",
    ),
    ErrorCode(
        "oauth_refresh_failed",
        "OAuth token refresh failed.",
        "run `comfy cloud login` to sign in again",
    ),
    ErrorCode(
        "oauth_cancelled",
        "OAuth flow was cancelled by the user.",
        "re-run `comfy cloud login` to retry sign-in",
    ),
    ErrorCode(
        "oauth_timeout",
        "Timed out waiting for browser callback during OAuth login.",
        "re-run `comfy cloud login` and complete the sign-in in your browser",
    ),
    # --- watcher / background jobs -------------------------------------------
    ErrorCode(
        "watcher_crashed",
        "Background watcher process is no longer running; job state is stale.",
        "re-submit the workflow, or check `comfy jobs status <id>` against the server",
    ),
    ErrorCode(
        "watcher_timeout",
        "Background watcher gave up after max runtime without a terminal status.",
        "the job may still be running — check `comfy jobs status <id>`, or re-watch with a longer `--timeout`",
    ),
    ErrorCode(
        "watcher_poll_error",
        "Background watcher encountered a transient error polling the server.",
        "transient — the job is likely still running; re-run `comfy jobs watch <id>`",
    ),
    ErrorCode(
        "unknown_status_stall",
        "Cloud reported a status the CLI does not recognize and it did not change within the stall window.",
        "check `comfy jobs status <id> --where cloud`; report the status so it can be mapped",
    ),
    ErrorCode(
        "no_prompt_ids",
        "`jobs wait` was given no prompt_ids to wait on.",
        "pass one or more prompt_ids, or `--all` to wait on every locally-tracked job",
    ),
    ErrorCode(
        "wait_timeout",
        "`jobs wait` gave up before every job reached a terminal state.",
        "the jobs may still be running — raise `--timeout`, or check `comfy jobs status <id>`",
    ),
    ErrorCode(
        "execution_error",
        "ComfyUI reported an execution error for the workflow.",
        "inspect the error details or re-run with `--wait --verbose`",
    ),
    ErrorCode(
        "transient_auth",
        "An API node's server-side session token expired mid-execution "
        '("Unauthorized: Please login first to use this node"). Transient — not a local credential problem.',
        "resubmit the same workflow — it succeeds on retry; `comfy cloud login` will not help",
    ),
    # --- background server logs ----------------------------------------------
    ErrorCode(
        "no_log_file",
        "`comfy logs` found no captured ComfyUI log — the server was never launched "
        "via `comfy launch --background`, or it was launched externally.",
        "start ComfyUI with `comfy launch` so its output is captured",
    ),
    ErrorCode(
        "log_read_failed",
        "`comfy logs` located the logfile but could not read it — it was removed or its "
        "permissions changed between the existence check and the read (TOCTOU window).",
        "check the file still exists and is readable, then retry",
    ),
    # --- general argument / mode errors --------------------------------------
    ErrorCode(
        "missing_argument",
        "Required argument(s) not provided.",
        "run the command with `--help` to see its required arguments",
    ),
    ErrorCode(
        "json_incompatible",
        "Requested feature is not available in JSON output mode.",
        "drop `--json` (or pass `--no-json`) for this command",
    ),
    # --- skills --------------------------------------------------------------
    ErrorCode(
        "unknown_skill",
        "Requested skill is not in the bundled set.",
        "run `comfy skills list` to see available skills",
    ),
    ErrorCode(
        "skill_invalid",
        "A skill path failed format validation (missing SKILL.md, frontmatter name/description, or name/dir mismatch).",
        "a skill dir must contain SKILL.md with `name:`/`description:` frontmatter; run `comfy skills validate <path>`",
    ),
    # --- workflow editor -----------------------------------------------------
    ErrorCode(
        "workflow_not_frontend_format",
        "Workflow editing requires the UI export (with `nodes[]` / `links[]`); "
        "got API-format. Auto-convert isn't wired yet.",
        "in ComfyUI, use the regular save (File > Save Workflow) — the API export is for `comfy run`, not for editing",
    ),
    ErrorCode(
        "workflow_slot_invalid",
        "A slot override failed validation (bad shape, unknown address, etc.).",
        "see `details` — addresses follow `<instance_id>.<input_name>`",
    ),
    # --- workflow fragments / compose ---------------------------------------
    ErrorCode(
        "fragment_invalid",
        "A workflow fragment file failed schema validation "
        "(bad `_fragment` header, missing fields, dangling `binds`, malformed interior node).",
        "see `details.path` and the message — run `comfy workflow fragment validate <path>` to re-check",
    ),
    ErrorCode(
        "fragment_lib_not_found",
        "The fragment library directory doesn't exist.",
        "create `./fragments/` (default) or pass `--lib <dir>`",
    ),
    ErrorCode(
        "blueprint_not_found",
        "The compose blueprint YAML file doesn't exist.",
        "check the path",
    ),
    ErrorCode(
        "blueprint_invalid_yaml",
        "The blueprint file isn't valid YAML.",
        "lint with `yamllint` or fix the syntax",
    ),
    ErrorCode(
        "blueprint_invalid",
        "The blueprint semantically fails to compose: missing required input/param, "
        "unknown input/param key, duplicate alias, or unresolvable cross-step reference.",
        "see `details.step_alias` and the message",
    ),
    ErrorCode(
        "blueprint_yaml_unavailable",
        "PyYAML is not installed — `comfy workflow compose` needs it to read blueprints.",
        "pip install pyyaml",
    ),
    ErrorCode(
        "compose_io_error",
        "Reading the blueprint or writing the composed workflow failed with an OSError "
        "(permissions, missing parent dir, disk full, unreadable encoding).",
        "check the path is readable/writable and the disk has space",
    ),
    ErrorCode(
        "workflow_conversion_failed",
        "`comfy workflow decompose` could not flatten a frontend-format workflow to "
        "API format (malformed graph, or object_info that doesn't match the nodes).",
        "re-export from ComfyUI, or pass a matching --input object_info.json",
    ),
    ErrorCode(
        "decompose_io_error",
        "Writing the projected fragment file failed with an OSError (permissions, missing parent dir, disk full).",
        "check the --out/--lib path is writable and the disk has space",
    ),
    # --- preview -------------------------------------------------------------
    ErrorCode(
        "preview_input_not_found",
        "The file passed to `comfy preview` doesn't exist or isn't readable.",
        "check the path",
    ),
    ErrorCode(
        "ffmpeg_unavailable",
        "`comfy preview` needs ffmpeg + ffprobe on PATH and they weren't found.",
        "install ffmpeg (e.g. `brew install ffmpeg` / `apt install ffmpeg`)",
    ),
    ErrorCode(
        "preview_unsupported_media",
        "The file has no image/video/audio stream to preview.",
        "pass an image, video, or audio file",
    ),
    ErrorCode(
        "preview_failed",
        "ffprobe/ffmpeg failed to probe the file or render the preview image.",
        "check the file isn't corrupt; try a different --grid/--width",
    ),
    # --- CQL / object_info ---------------------------------------------------
    ErrorCode(
        "cql_no_graph",
        "No object_info source available (no local server, no `--input`).",
        "pass `--input <path>`, or start the server with `comfy launch`",
    ),
    ErrorCode(
        "object_info_stale",
        "Live object_info fetch failed; the response was served from a cached copy that may be out of date. "
        "`details.source` has the host key; `details.reason` has the fetch error. "
        "Surfaced in `data.warnings[]` (not as an error envelope) so the command still succeeds.",
        "re-run once the server/session is reachable to get a fresh schema",
    ),
    ErrorCode(
        "description_ignored",
        "`comfy workflow save --where local --description` was given a description, but the local "
        "file-backed `/userdata` store has nowhere to keep it. Surfaced in `data.warnings[]` "
        "(not as an error envelope) so the save still succeeds.",
        "descriptions are a Comfy Cloud feature; drop `--description` on the local path",
    ),
    ErrorCode(
        "cql_query_invalid",
        "Grammar query failed to parse or evaluate.",
        "check the grammar; `comfy nodes ls --help` has examples",
    ),
    ErrorCode(
        "node_not_found",
        "Requested node class isn't in the loaded environment.",
        "see `details.close_matches` or run `comfy nodes search`",
    ),
    # --- file transfer (upload / download) -----------------------------------
    ErrorCode(
        "upload_failed",
        "HTTP error during file upload to the server's input directory.",
        "check the file exists and the server is reachable",
    ),
    ErrorCode(
        "download_failed",
        "HTTP error while downloading an output file.",
        "check that the job completed successfully and the server is reachable",
    ),
    ErrorCode(
        "download_no_outputs",
        "The job has no output files (yet).",
        "wait for the job to complete before downloading",
    ),
    ErrorCode(
        "download_no_prompt",
        "No prompt_id was provided to the download command.",
        "pass a prompt_id argument, or pipe from `comfy --json run --wait`",
    ),
    ErrorCode(
        "download_job_not_found",
        "The prompt_id wasn't found in state files or the server API.",
        "check the prompt_id and ensure the job has completed",
    ),
    ErrorCode(
        "setup_missing_where",
        "--non-interactive requires --where (local or cloud).",
        "comfy setup --non-interactive --where cloud --api-key sk-...",
    ),
    ErrorCode(
        "setup_no_auth",
        "Cloud requires authentication in non-interactive mode.",
        "pass --api-key sk-... or run `comfy cloud login` first",
    ),
    # --- project (project/1 convention) ---------------------------------------
    ErrorCode(
        "project_already_exists",
        "`comfy project init` ran in a directory already governed by a comfy.yaml project (`details.root`).",
        "use the existing project, or init outside it",
    ),
    ErrorCode(
        "project_not_found",
        "No comfy.yaml (schema project/1) governs the current directory.",
        "run: comfy project init",
    ),
    ErrorCode(
        "asset_not_pushed",
        "A blueprint references `$asset.<name>` with no matching entry in .comfy/assets.lock.json "
        "(or the file is missing under assets/).",
        "run: comfy assets push",
    ),
    ErrorCode(
        "asset_stale",
        "A referenced asset changed on disk after its last push — its sha256 no longer matches the lock.",
        "run: comfy assets push",
    ),
    ErrorCode(
        "var_not_defined",
        "A blueprint references `$var.<name>` with no matching entry under `vars:` in the project's comfy.yaml.",
        "add the name under `vars:` in <root>/comfy.yaml, then re-compose",
    ),
    # --- generate / emit -----------------------------------------------------
    ErrorCode(
        "emit_workflow_failed",
        "`generate --emit-workflow` could not build the partner-node workflow.",
        "check the model name and that all required inputs are provided",
    ),
    # --- feedback ------------------------------------------------------------
    ErrorCode(
        "feedback_message_required",
        "`comfy feedback` was run in JSON/non-interactive mode without an inline message.",
        'comfy feedback "your feedback here"',
    ),
)


_BY_CODE: dict[str, ErrorCode] = {ec.code: ec for ec in REGISTRY}


def is_registered(code: str) -> bool:
    return code in _BY_CODE


def get(code: str) -> ErrorCode | None:
    return _BY_CODE.get(code)


def all_codes() -> list[str]:
    return [ec.code for ec in REGISTRY]


def as_discover_rows() -> list[dict[str, str | None]]:
    """The shape ``comfy discover`` emits under ``data.error_codes``."""
    return [{"code": ec.code, "meaning": ec.meaning, "hint": ec.hint} for ec in REGISTRY]
