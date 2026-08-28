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
    # --- launch / stop lifecycle ---------------------------------------------
    ErrorCode(
        "server_already_running",
        "`comfy launch --background` found a background ComfyUI already running.",
        "run `comfy stop` before launching another background service",
    ),
    ErrorCode(
        "port_invalid",
        "`comfy launch --background` got a non-integer `--port`. `details.port` carries the offending value.",
        "pass an integer `--port` (e.g. `--port 8188`)",
    ),
    ErrorCode(
        "port_in_use",
        "`comfy launch --background` found the target port already in use. `details.port` carries the port.",
        "stop the process on that port or pass a different `--port`",
    ),
    ErrorCode(
        "launch_failed",
        "ComfyUI failed to launch (background monitor saw no success line) or a "
        "foreground launch exited non-zero. `details` carries the log / returncode.",
        "check the error log for the underlying failure",
    ),
    ErrorCode(
        "no_background_server",
        "`comfy stop` found no background ComfyUI recorded as running.",
        "run `comfy launch --background` first",
    ),
    ErrorCode(
        "stop_failed",
        "`comfy stop` could not kill the recorded background ComfyUI process. `details.pid` carries the process id.",
        "kill the process manually if it is still running",
    ),
    ErrorCode(
        "port_not_listening",
        "`comfy stop --port <p>` found no process LISTENing on that port. `details.port` carries the port.",
        "check the port, or run `comfy stop` to stop the server this CLI started",
    ),
    ErrorCode(
        "unverified_process",
        "`comfy stop --port <p>` found a listener it could not positively identify as ComfyUI, so it "
        "refused to stop it. `details` carries the pid, whatever cmdline was readable, and the reason.",
        "confirm what is on that port and stop it yourself if it really is ComfyUI",
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
        "workflow_unparseable",
        "A cloud `/api/workflows` call returned a non-empty 200 body that couldn't be decoded as JSON "
        "(non-UTF-8 bytes or a non-JSON body such as an HTML proxy/error page). Distinct from an empty "
        "body (legitimately no data): the malformed body is surfaced as a hard error rather than a "
        "misleading empty list / null id. `details.operation` carries the verb.",
        "the server sent a malformed body; retry, and report it if it persists",
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
        "Asked about a prompt_id the server doesn't know. `comfy jobs status` only reports this once the "
        "local state file has been checked too — and only a file that names this same prompt on this same "
        "target counts, so a cloud job or another local instance's job is never the answer here (when it "
        "is a cloud one, `hint` redirects to `--where cloud`). If that file holds a terminal verdict (e.g. "
        "the job died with an earlier server) AND the live server confirmed it has no record, the verdict "
        "is returned as a normal result instead. When a matching file exists but that pair does not hold — "
        "the record is non-terminal, or `/queue` and `/history` did not answer — `details` carries "
        "`last_known_status`, `submitted_at`, `updated_at`, `workflow`, and `server_confirmed_no_record` "
        "(false means the absence is unverified, so it is not the job's outcome).",
        "`comfy jobs ls` to find a valid prompt_id",
    ),
    ErrorCode(
        "partner_node_requires_credential",
        "Workflow uses a partner-API node (category `partner/*` — Veo, Kling, BFL, Gemini, etc.) "
        "but no `api_key_comfy_org` credential is available. Local submit would succeed at /prompt "
        "and then fail opaquely at execute time with `Unauthorized: Please login first`.",
        "run: comfy cloud login (or set COMFY_API_KEY in the environment, or persist a key with "
        "`comfy cloud set-key --key …` so the local submit path can inject it too; cloud runs "
        "auto-inject via --where cloud)",
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
        "no_checkpoint_available",
        "`comfy run --prompt`/`--set` (bundled default text2img) needs a checkpoint, but the "
        "target positively enumerated ZERO installed checkpoints. Only raised when object_info "
        "was fetched and its checkpoint list is empty — never when object_info couldn't be fetched "
        "(that path fails open and submits).",
        "install a checkpoint (local: `comfy model download --url <checkpoint-url>`; cloud: run a "
        "published gallery template, which provisions models), then re-run — or `--set "
        "checkpoint=<name>` once one is available",
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
        "gallery_cache_write_failed",
        "The gallery index was fetched but could not be written to the local cache. "
        "Only `comfy templates refresh` raises this — for `templates ls/show/fetch` "
        "a cache-write failure is non-fatal, since the data is already in hand.",
        "check permissions and free space on the cache directory",
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
        "host_port_invalid",
        "`--host`/`--port` (or a combined `--host host:port`) failed validation before any server "
        "contact; the process exits 2 (usage error). The rejected value is usually a flag, but the "
        "same check also covers the host recorded in `config.background`, so a corrupted background "
        "record trips it with no bad flag passed. Click also writes its usual usage message to "
        "stderr — this envelope exists so JSON/NDJSON consumers still get a parseable final line.",
        "pass `--host <hostname-or-ip>` and `--port 1-65535`; if you passed neither, the saved "
        "background server is bad — clear it with `comfy stop`",
    ),
    ErrorCode(
        "host_flag_cloud",
        "`--host`/`--port` were combined with an effective `cloud` target. They address a local "
        "ComfyUI only; the cloud address comes from the signed-in account. `details` carries the "
        "offending host/port, the resolved `where`, and the `where_source` that produced it "
        "(`flag`, `env`, `project`, `config`, or `auto`) — the target may never have been "
        "explicitly requested.",
        "pass `--where local` to aim at a local server; to reach a different cloud address set "
        "`COMFY_CLOUD_BASE_URL` or run `comfy cloud set-base-url`",
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
        "cloud_billing_unavailable",
        "`comfy cloud status` could not read `/api/billing/status`, so there is no tier or "
        "subscription state to report. Distinct from `cloud_unauthorized` (a rejected "
        "credential) and from the command's own degraded rows: the workspace, features and "
        "plans calls each drop a single row when they fail, and only this one is fatal.",
        "retry shortly; if it persists, contact Comfy support",
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
        "An argument intended for a URL path or for a filesystem path component failed "
        "safe-path validation — e.g. a `comfy model download` filename (from `--filename` or "
        "from the CivitAI API response) that carries a path separator, a drive letter or `..` "
        "and would write outside the workspace.",
        "a path-segment argument must be a single segment: non-empty, not `.` or `..`, and free "
        "of `/` and `\\`; for `model download`, choose the destination directory with "
        "`--relative-path` instead",
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
        "cloud_only_command",
        "The command requires Comfy Cloud (e.g. `comfy assets library`); there is no local equivalent.",
        "sign in with `comfy cloud login` and re-run with `--where cloud`",
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
        "template_ambiguous",
        "`comfy templates get --where …` matched more than one template; `get` resolves exactly one. "
        "`details.candidates` carries up to 10 matches (name + title/type/tags/models) and "
        "`details.matched` the full count.",
        "add another --where filter (e.g. name=<substring>) to narrow to a single template",
    ),
    ErrorCode(
        "template_filter_invalid",
        "A `comfy templates get --where` filter was malformed (not key=value), used an unknown key, "
        "or no filter was given at all. Valid keys: type, category, tag, model, provider, name — "
        "identical semantics to the `templates ls` flags.",
        "pass repeatable `--where key=value` pairs, e.g. `--where type=video --where tag=API`",
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
        "server_died",
        "The local ComfyUI server became unreachable (or restarted without the job) while it "
        "was in flight — the server likely crashed or was killed (e.g. an out-of-memory allocation). "
        "Raised by the background watcher and by a foreground (`--wait`) run; recorded on the job state file.",
        "check the ComfyUI server log (it may have been OOM-killed), then `comfy launch` and re-submit; "
        "the prompt_id is in `comfy jobs status <id>`",
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
    ErrorCode(
        "select_no_match",
        "A `--select <expr>` projection was malformed or matched nothing in the payload. The command "
        "fails open — `ok` stays true and exit code stays 0 — and `data` carries a bounded key "
        "inventory of the full payload (`data.inventory`: top-level keys, value types, sizes, one "
        "nested level of keys) plus this advisory in `data.warnings[]`.",
        "read `data.inventory` for the payload's real keys, then re-run with a corrected --select; "
        "grammar: dot path `a.b.c`, array index `a.0.b`, wildcard `items.#.name`, comma multi-select "
        "`name,inputs`",
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
        "workflow_print_unsupported",
        "`comfy workflow print` refused: the workflow contains something it cannot render faithfully "
        "(legacy group node, duplicate node id, link to a missing node/slot, non-integer link slot, "
        "link cycle, unknown `--format`). `details.reasons` lists every reason.",
        "fix the listed reasons, or read the graph with `comfy workflow slots` / `comfy workflow ls-nodes`",
    ),
    ErrorCode(
        "workflow_slot_invalid",
        "A slot override failed validation (bad shape, unknown address, etc.).",
        "see `details` — addresses follow `<instance_id>.<input_name>`; "
        "`comfy workflow print <file>` shows every node with its id and widget names",
    ),
    ErrorCode(
        "workflow_edit_invalid",
        "A structured edit (add-node/connect/set-widget/delete-node) failed: "
        "unknown class_type, missing node, bad slot/widget name, or malformed address.",
        "run `comfy workflow print <file>` to see every node, edge and widget value with its id in one read "
        "(`comfy workflow slots <file>` for exact `<node_id>.<input>` addresses, `comfy nodes search` for class names)",
    ),
    ErrorCode(
        "workflow_clear_not_batchable",
        "A batch (`workflow apply` / `workflow foreach`) contained a `clear` op. `clear` wipes the whole "
        "graph and is standalone-only (docs/op-vocabulary-v1.md: batchable = no), so the batch was "
        "rejected atomically — nothing was applied.",
        "run the standalone `comfy workflow clear <file>` first, then apply the remaining ops as a batch",
    ),
    ErrorCode(
        "workflow_reset_doc_not_batchable",
        "A batch (`workflow apply` / `workflow foreach`) contained a `reset_doc` op. `reset_doc` resets the "
        "whole document to the empty baseline and erases its replay history, so it is standalone-only "
        "(docs/op-vocabulary-v1.md: batchable = no) and the batch was rejected atomically — nothing was applied.",
        "run the standalone `comfy workflow reset-doc <file> --confirm` first, then apply the remaining ops as a batch",
    ),
    ErrorCode(
        "workflow_reset_doc_unconfirmed",
        "`comfy workflow reset-doc` was called without `--confirm`. The command fails closed: it erases every "
        "node AND the document's replay history, which no later op can undo.",
        "re-run with `--confirm` if that is really what you want — otherwise `comfy workflow clear <file>` "
        "empties the graph while keeping the document's history",
    ),
    ErrorCode(
        "normalized_value",
        "Warning (not fatal): a set-widget value wasn't an exact COMBO option, so "
        "the nearest matching option was used. Surfaced in the op's `warnings`.",
        "see the warning's `from`/`to`; pass an exact option to avoid the fuzzy match",
    ),
    ErrorCode(
        "ui_only_node_skipped",
        "Warning (not fatal): `workflow capture` skipped a UI-only node (Note/MarkdownNote/"
        "Reroute/GetNode/SetNode/PrimitiveNode) — those never reach the API and `apply` "
        "refuses to mint them. Data flow through the node was spliced to the real source.",
        "expected for annotated workflows; the recipe rebuilds the executable graph, not canvas decoration",
    ),
    ErrorCode(
        "ui_only_link_dropped",
        "Warning (not fatal): a captured link traced back through UI-only nodes to no real "
        "source (e.g. a dangling Reroute), so `workflow capture` dropped it.",
        "check the named node/input; wire it to a real source before capturing if the link matters",
    ),
    ErrorCode(
        "primitive_feed_unrepresentable",
        "Warning (not fatal): a PrimitiveNode feeds an input that is not a widget on the "
        "target node, so `workflow capture` could not express its value as a set_widget.",
        "set the target input from a real node or widget before capturing",
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
        "ffmpeg_untrusted",
        "ffmpeg/ffprobe were found, but only inside the directory you ran from, "
        "so they were refused rather than executed — the shape of a planted binary.",
        "run `comfy preview` from a directory that does not contain an ffmpeg/ffprobe binary",
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
    ErrorCode(
        "node_deprecated",
        "`workflow add-node` (or an `add_node` op in a batch) named a class the catalog marks deprecated. "
        "Nothing was added. `details.replacement` names the live class with the same display name when "
        "one exists.",
        "add `details.replacement` instead, or pass --allow-deprecated "
        '(`"allow_deprecated": true` on the op) when the user asked for that exact node',
    ),
    ErrorCode(
        "path_bounds_invalid",
        "`comfy nodes path` was given `--max-depth` or `--max-paths` below 1. Such a bound admits no "
        "path at all, so the search is refused rather than returning an empty result that would read "
        "as a proof that no route exists.",
        "use `--max-depth 6 --max-paths 10` (or any bound >= 1)",
    ),
    ErrorCode(
        "expand_miss",
        "`comfy nodes search --expand-top N` matched a node class but could not resolve its full schema "
        "from the catalog. Surfaced as a per-hit error entry inside `data.expanded[]` (not as an error "
        "envelope) — the search itself still succeeds and the other hits still expand.",
        "the hit itself is still valid; inspect it directly with `comfy nodes show <class_type>`",
    ),
    # --- file transfer (upload / download) -----------------------------------
    ErrorCode(
        "upload_failed",
        "HTTP error during file upload to the server's input directory.",
        "check the file exists and the server is reachable",
    ),
    ErrorCode(
        "download_failed",
        "A download failed. Either an HTTP error while fetching a job's output file, or "
        "`comfy model download` failing to fetch the model (transfer error, Hugging Face "
        "download error, or an unresolvable CivitAI model/version). `details.url` carries the "
        "source URL; `details.stage` is `resolve` when the failure was metadata lookup, not transfer.",
        "check that the source URL is reachable and the job completed successfully",
    ),
    ErrorCode(
        "model_file_exists",
        "`comfy model download` refused to overwrite an existing file at the target path "
        "(`details.path`). The download was NOT performed — the command fails rather than "
        "exiting 0, so a caller can't mistake the skip for a completed download.",
        "pass `--filename` to save under a different name, or remove the existing file",
    ),
    ErrorCode(
        "model_download_in_flight",
        "`comfy model download` refused to start because a live download is already writing to "
        "the same destination (`details.path`). `details.download_id` names that download, "
        "`details.status` is its current status, and `details.kind` is `background` (a detached "
        "worker) or `foreground` (a `comfy model download` running in another terminal). The httpx "
        "downloader streams into a `.part` sibling, so the destination stays absent until the "
        "transfer completes — without this check two submissions would both run and the later one "
        "would silently overwrite the earlier; under `--downloader aria2`, which writes straight to "
        "the destination, they would interleave into the same file.",
        "track it with `comfy model download-status <id>`; a background download can be stopped "
        "with `comfy model download-cancel <id>`, a foreground one with Ctrl-C in its own terminal",
    ),
    ErrorCode(
        "model_download_foreground_cancel",
        "`comfy model download-cancel` refused to cancel a download that is running in the "
        "foreground of another terminal (`details.id`, `details.pid`). A background download runs in "
        "its own session, so cancelling it signals only the worker's process group; a foreground "
        "download's recorded pid is the user's own CLI process, which shares the terminal's "
        "foreground process group — signalling that group would kill the surrounding shell job "
        "rather than just the transfer. Once the foreground process is gone its record reconciles "
        "to `failed` and `download-cancel` will sweep the partial file as usual.",
        "interrupt it with Ctrl-C in the terminal running it",
    ),
    ErrorCode(
        "hf_unauthorized",
        "Hugging Face returned 401 for the model URL and no Hugging Face API token is configured "
        "(gated or private repo).",
        "set the token via `comfy model download --set-hf-api-token <token>` or the `HF_API_TOKEN` "
        "environment variable",
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
    # --- background model downloads (`model download --background`) ----------
    ErrorCode(
        "download_not_found",
        "No download state file matches the given download id. Both `--background` submissions and plain "
        "foreground `comfy model download` runs write one, so an id that resolves to neither was never "
        "written, or its record has already been pruned.",
        "list the known downloads with `comfy model downloads`",
    ),
    ErrorCode(
        "download_state_unwritable",
        "The `<workspace>/.comfy-downloads` state directory could not be written.",
        "check the workspace is writable, or run without --background",
    ),
    ErrorCode(
        "download_worker_spawn_failed",
        "The detached background download worker could not be started.",
        "run without --background to download in the foreground",
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
        "generate_target_required",
        "`comfy generate` was invoked with a flag token where its first positional argument (the "
        "partner model alias) belongs — e.g. `comfy generate --prompt=x`. `generate` is a "
        "cloud/partner verb that spends credits; it always needs a model alias first.",
        'name a model alias first (`comfy generate flux-pro --prompt "…"`, `comfy generate list` to '
        "browse them), or use `comfy run-template` for local text-to-image",
    ),
    ErrorCode(
        "generate_unknown_model",
        "The model alias/id passed to `comfy generate` (or `generate schema` / `generate resume`) is "
        "not in the partner-endpoint catalog.",
        "run `comfy generate list` to see available models; `comfy generate refresh` re-fetches the catalog",
    ),
    ErrorCode(
        "generate_bad_args",
        "`comfy generate` could not parse its arguments: a missing/malformed flag value, a missing "
        "required model parameter, a bad subcommand usage, or a resume of a non-polling model.",
        "run `comfy generate schema <model>` for the parameter list, or `comfy generate --help` for usage",
    ),
    ErrorCode(
        "generate_timeout_invalid",
        "`comfy generate --timeout` was given a value that isn't a number.",
        "pass seconds as a number, e.g. `--timeout 300`",
    ),
    ErrorCode(
        "generate_api_error",
        "The partner-proxy API rejected the call or returned an unusable response (auth failure, "
        "non-2xx status, non-JSON body). `details.status` / `details.body` carry the response when "
        "the failure was an HTTP status.",
        "check `comfy cloud login` / COMFY_API_KEY and the reported status; retry if it was a 5xx",
    ),
    ErrorCode(
        "generate_network_error",
        "A transport-level failure (DNS, TLS, connect, read timeout) while talking to the partner "
        "proxy — the request may never have reached it.",
        "check network connectivity and retry; raise `--timeout` if the model is slow",
    ),
    ErrorCode(
        "generate_job_failed",
        "The partner job reached a terminal non-succeeded state (failed/cancelled). "
        "`details.response` carries the raw partner response.",
        "check `details.response` for the partner's reason; fix the inputs and re-run, or "
        "`comfy generate resume <model> <job_id>` if the job may still settle",
    ),
    ErrorCode(
        "generate_spec_invalid",
        "`comfy generate refresh` fetched an OpenAPI document that failed validation, so it was "
        "refused rather than cached over the working catalog.",
        "check COMFY_API_BASE_URL points at the Comfy API; the existing cached catalog is still usable",
    ),
    ErrorCode(
        "emit_workflow_failed",
        "`generate --emit-workflow` could not build the partner-node workflow.",
        "check the model name and that all required inputs are provided",
    ),
    # --- custom node registry ------------------------------------------------
    ErrorCode(
        "node_publish_failed",
        "Publishing a custom-node version to the registry failed: either a "
        "client-side validation gap (missing publisher id / project name in "
        "pyproject.toml) or a non-2xx from the registry. `details.status` and "
        "`details.body` carry the response when it was an HTTP failure.",
        "check `details.body`; ensure `[tool.comfy] publisher_id` and `[project] name` are set, and the token is valid",
    ),
    ErrorCode(
        "registry_install_failed",
        "`comfy node registry-install` fetching a custom node from the registry failed with a non-2xx. "
        "`details.status` and `details.body` carry the response.",
        "check the node id and version exist in the registry (`comfy node registry-list`); check `details.body`",
    ),
    ErrorCode(
        "spend_consent_required",
        "A credit-spending command hit its spend gate with no consent, so it failed closed — "
        "nothing was submitted and no credits were spent. `comfy run-template` raises this when a "
        "template uses partner-API (paid) nodes and `--allow-spend` is absent or the interactive "
        "confirmation was declined (`details.partner_nodes` / `details.gallery_signals` carry the "
        "evidence); `comfy generate` raises it when a credit-spending call runs non-interactively "
        "(`--json` / no TTY) with no consent.",
        "consent to the spend and re-run — `comfy run-template --allow-spend`, or "
        "`comfy generate --yes` (persist with `comfy generate consent always`)",
    ),
    # --- update / version switch --------------------------------------------
    ErrorCode(
        "update_version_target_invalid",
        "`comfy update --version` was combined with a target other than `comfy`.",
        "run `comfy update comfy --version <version>`",
    ),
    ErrorCode(
        "update_custom_nodes_failed",
        "`comfy update all --exit-on-fail` ran `cm-cli update all` and it exited non-zero. "
        "The update is not atomic, so some packs may have updated before the failure; "
        "`details.cm_cli_returncode` carries cm-cli's raw status (the process exit code is "
        "normalized — signals become 128+N, and 2 becomes 1 so it can't be confused with a "
        "CLI usage error).",
        "read the cm-cli output above for the failing pack, then re-run `comfy update all`",
    ),
    ErrorCode(
        "version_switch_unknown_version",
        "`comfy update comfy --version X` could not resolve X to a ComfyUI tag; the workspace was left untouched.",
        "run `git tag --list 'v*'` in your ComfyUI workspace to see every available version",
    ),
    ErrorCode(
        "version_switch_git_unavailable",
        "`comfy update comfy --version X` could not use git at all — it is absent from PATH, or the "
        "only match resolved into the directory you ran from and was refused. Reported separately so "
        "it isn't misread as the requested version not existing.",
        "a version switch needs a usable git — install it, or run from a directory that has no git binary in it",
    ),
    ErrorCode(
        "version_switch_dirty_tree",
        "`comfy update comfy --version X --no-stash` found uncommitted changes and refused to switch.",
        "commit or stash your changes, or re-run without --no-stash to stash them automatically",
    ),
    ErrorCode(
        "version_switch_failed",
        "A git operation during `comfy update comfy --version X` failed; any stash that was created is preserved.",
        "resolve the git error in your ComfyUI workspace, then re-run",
    ),
    ErrorCode(
        "version_switch_deps_failed",
        "The version switch checked out successfully but reinstalling requirements.txt failed.",
        "re-run the same command once the cause is fixed; it is idempotent and safe to repeat",
    ),
    # --- feedback ------------------------------------------------------------
    ErrorCode(
        "feedback_message_required",
        "`comfy feedback` was run in JSON/non-interactive mode without an inline message.",
        'comfy feedback "your feedback here"',
    ),
    # --- custom node install (`comfy node install`) ---------------------------
    ErrorCode(
        "node_install_failed",
        "`comfy node install --exit-on-fail` failed: `cm-cli install` exited non-zero "
        '(`details.failed_stage` == "cm-cli", raw status in `details.cm_cli_returncode`) or, '
        "with --fast-deps, the follow-up dependency install failed after the packs installed "
        '(`details.failed_stage` == "dependency-install", raw status in `details.returncode`). '
        "The process exit code is normalized — signals become 128+N, and any status whose low "
        "byte is 0 or 2 becomes 1 so it can't read as success or a CLI usage error.",
        "read the output above for the failing pack or dependency, then re-run `comfy node install --exit-on-fail`",
    ),
    # --- custom node dependency report (`comfy node deps`) --------------------
    ErrorCode(
        "installed_versions_unavailable",
        "`comfy node deps` could not read the workspace venv's installed packages (`pip list --format=json` "
        "failed, timed out, or returned unparseable output), so every parseable requirement is reported with "
        '`status: "unknown"`. Surfaced in `data.warnings[]` (not as an error envelope) so the declared '
        "requirements are still reported.",
        "check the workspace venv has pip (`comfy env`), then re-run",
    ),
    ErrorCode(
        "pack_read_error",
        "A pack's `requirements.txt` existed but could not be read (permissions, I/O). That pack's row omits "
        "the unreadable file's requirements. Surfaced in `data.warnings[]` (not as an error envelope) so the "
        "rest of the report still succeeds.",
        "check the file's permissions under `custom_nodes/<pack>/`",
    ),
    ErrorCode(
        "registry_unavailable",
        "`comfy node deps --registry <node-id>` could not reach the Comfy registry (network failure, timeout, "
        "or a non-200 response), so that candidate's row carries `declared: null` plus a per-entry `warning`. "
        "Surfaced in `data.warnings[]` (not as an error envelope): every other pack still reports normally.",
        "check network access to api.comfy.org, then re-run (add `--refresh` to bypass the 1h cache)",
    ),
    ErrorCode(
        "registry_invalid_node_id",
        "A `comfy node deps --registry <node-id>` value was blank or contained characters outside "
        "`[A-Za-z0-9._-]`, so it was rejected without a network call. Registry ids never contain `/`, `?` "
        "or `#`; interpolated into the lookup URL those would retarget the request at a different path "
        "(including the side-effecting install endpoint) or inject a query string. "
        "Surfaced in `data.warnings[]`: the other `--registry` ids still report normally.",
        "pass the pack's registry id as shown by `comfy node registry-list` (e.g. `comfyui-example`), "
        "not a URL, an `owner/repo` path, or a local directory name",
    ),
    ErrorCode(
        "registry_node_not_found",
        "`comfy node deps --registry <node-id>` reached the Comfy registry, which reported no such node "
        "(HTTP 404) — a misspelled id, or a pack that was never published to the registry. Distinct from "
        "`registry_unavailable`: retrying or `--refresh` will never resolve it. Surfaced in `data.warnings[]`.",
        "check the id with `comfy node registry-list`; an unpublished pack has no registry metadata, so "
        "install it and re-run `comfy node deps` to read its requirements from disk instead",
    ),
    ErrorCode(
        "registry_partial_dependency_metadata",
        "`comfy node deps --registry <node-id>` got a dependency list from the registry containing "
        "non-string entries (e.g. `null`), which were dropped. The row's `declared` list is therefore "
        "incomplete — a dropped entry that would have conflicted is not reported. Surfaced in `data.warnings[]`.",
        "treat that row as partial; read the pack's own `requirements.txt` upstream to confirm the full set",
    ),
    ErrorCode(
        "registry_no_dependency_metadata",
        "`comfy node deps --registry <node-id>` reached the registry, but it published no dependency metadata "
        "for that pack's latest version, so the row carries `declared: null` rather than an empty list — the "
        'API cannot distinguish "declares nothing" from "field absent". Surfaced in `data.warnings[]`.',
        "the pack publisher must publish a version declaring its dependencies; nothing to fix locally",
    ),
    # --- build (the serverless builder) --------------------------------------
    ErrorCode(
        "build_spec_invalid",
        "A build spec or legacy scan definition could not be read, has an unsupported schema, or is invalid. "
        "`details.path` carries the path when one is available.",
        "fix the named field, or regenerate the file",
    ),
    ErrorCode(
        "build_spec_write_error",
        "`comfy build` could not write the build spec or legacy scan definition. `details` carries the "
        "path and the underlying OS error.",
        "check the directory exists and is writable",
    ),
    ErrorCode(
        "build_spec_not_found",
        "A `comfy build` command found no spec at the path `PATH` resolved to — `<dir>/comfy-build.yaml` for "
        "a directory, or the file itself for a `.yaml`/`.json` `PATH`. `details.path` carries the exact "
        "absolute path probed.",
        "create a comfy-build.yaml at that path, or pass the PATH that holds the spec",
    ),
    ErrorCode(
        "build_models_dir_missing",
        "`comfy build scan` could not find a models/ directory to scan. `details.path` carries the "
        "resolved path. Either no workspace is selected or the given `--models-dir` doesn't exist.",
        "run from a ComfyUI workspace, or pass `--models-dir <path>` pointing at your models/ folder",
    ),
    ErrorCode(
        "build_output_write_error",
        "`comfy build scan --output <path>` could not write the definition file. `details` carries the "
        "path and the underlying OS error.",
        "check the directory exists and is writable",
    ),
    ErrorCode(
        "build_definition_invalid",
        "`comfy build create --from <path>` could not read the definition file, or it isn't a "
        "build definition (missing/invalid JSON or no `models` key). `details.path` carries the path.",
        "pass a file produced by `comfy build scan -o <path>`",
    ),
    ErrorCode(
        "build_workflow_invalid",
        "`comfy build from-workflow --from <path>` could not read the workflow file, or it is not a JSON "
        "object. `details.path` carries the path.",
        "pass a workflow saved from ComfyUI (either the editing format or the API export)",
    ),
    ErrorCode(
        "build_not_signed_in",
        "`comfy build create --execute` found no usable Cloud JWT — the builder authenticates with the "
        "OAuth session token, and there isn't a valid one.",
        "run `comfy cloud login` first",
    ),
    ErrorCode(
        "build_upload_unavailable",
        "`comfy build create --execute` couldn't produce a private-blob upload: the model bytes weren't "
        "found (pass `--models-dir`), or a local (non-git) custom node needs packaging, which isn't supported yet.",
        "pass `--models-dir <path>` so model bytes can be located; use git-based custom nodes for now",
    ),
    ErrorCode(
        "build_builder_error",
        "A `comfy build` builder call got an error from the builder API (HTTP error, network failure, or "
        "an unexpected response shape). `details`/message carry the underlying error.",
        "check the builder URL and your access; retry, and inspect the message for the specific failure",
    ),
    ErrorCode(
        "build_not_enabled",
        "The builder returned 403 FEATURE_NOT_ENABLED: the developer platform is in limited beta and the "
        "signed-in account is not enabled for it yet.",
        "the developer platform is in limited beta; request access, then sign in with an enabled account",
    ),
    ErrorCode(
        "build_missing_comfy_version",
        "`comfy build create` was given a definition with no `baseComfyVersion`. The builder can create "
        "the build but cannot cut a release without a pinned ComfyUI version, so create fails fast here "
        "instead of surfacing a raw builder 400 after doing work. `details.path` carries the definition path.",
        "re-scan with `--comfy-version <ref>` (a tag, branch, or commit), or add a `baseComfyVersion` to the definition JSON",
    ),
    ErrorCode(
        "build_registry_pin_missing",
        "`comfy build create --execute` sent the definition to the builder's snapshot importer, which "
        "could not vouch for one or more custom node pins and returned a definition without them. Creating "
        "anyway would build an image quietly missing what the user asked for, so create stops. The advisories "
        "printed above name why each pack could not be carried.",
        "edit the definition to name a published registry version, or remove the pack",
    ),
    ErrorCode(
        "build_delete_needs_confirm",
        "`comfy build delete` was run without `--yes` in a non-interactive context (JSON output, an agent, "
        "or a pipe) where there is no TTY to confirm on. Delete is refused rather than blocking on a prompt.",
        "pass `--yes` to confirm the delete when running non-interactively",
    ),
    # --- knowledge -----------------------------------------------------------
    ErrorCode(
        "knowledge_unavailable",
        "No knowledge bundle could be loaded (no COMFY_KNOWLEDGE_FILE, no cache, no reachable COMFY_KNOWLEDGE_URL).",
        "set COMFY_KNOWLEDGE_FILE to a knowledge.json or COMFY_KNOWLEDGE_URL to fetch one; run `comfy knowledge status`",
    ),
    ErrorCode(
        "knowledge_unknown_model",
        "`comfy knowledge resolve` found no row for the alias or id. `details.close_matches` lists near names.",
        "try one of `details.close_matches`",
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
