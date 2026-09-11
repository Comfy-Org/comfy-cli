# `comfy run --json`: Machine-Readable Output (NDJSON)

> **Dialect change (breaking).** The legacy `{"event": …, "schema_version": 1,
> "error": {"kind": …}}` dialect was removed; `comfy run --json` now emits the
> CLI-wide renderer stream. Every event line carries `schema: "event/1"` and a
> `type` discriminator, and the stream is terminated by a single envelope line
> (`schema: "envelope/1"`, `type: "envelope"`) carrying `ok`/`data`/`error`.
> Consumers discriminate the final line by `type: "envelope"`. Error
> categorisation moved from `error.kind` to the registered `error.code` values
> (`comfy --json discover` lists them all). The full legacy→current mapping is
> at the [bottom of this document](#legacy-dialect-mapping).

This document specifies the output contract of `comfy run --json`. The intent
is to give agents and automation a stable, parseable view of a workflow
execution — independent of the human-readable Rich-formatted output that
`comfy run` emits by default.

`comfy run --json` is exactly `comfy --json-stream run`: the command-local
flag switches the process-wide renderer into NDJSON streaming mode. The same
event names are used by `comfy jobs watch` in stream mode, so the run stream
and the watch stream speak one dialect.

## `jobs watch` attaches as the submitting session

A local ComfyUI server does **not** broadcast execution events: it addresses
`executing` / `executed` / `progress_state` / `execution_cached` /
`execution_success` to the websocket session that submitted the prompt. So
`comfy jobs watch <prompt_id>` resolves that prompt's `client_id` — from the job
state file `comfy run` wrote, else `/queue`, else `/history` — and reconnects
under it; the terminal envelope reports which id it used (`data.client_id`) and
whether it was the real submitter (`data.attached`). Use `--client-id` to force a
specific one. Both keys are present on *every* `jobs watch` terminal envelope: a
watch of an already-finished prompt short-circuits without opening a socket, and
reports `client_id: null` / `attached: false`. Reconnecting under an existing id
is ComfyUI's own session-resume path, so a submitter that is *still* holding that
socket (an open browser tab, a blocking `comfy run`) stops receiving events until
it reconnects.

`data.completed_nodes` on the terminal envelope does not depend on the stream: it
is the union of what the watch observed and what `/history` records for the
prompt, so it is populated even for a watch that attached after the job ended.

The per-node events `jobs watch` streams carry the **same shape** as the `comfy
run` ones — one event per node, keyed by `node`. ComfyUI's websocket delivers
`execution_cached` as a single message listing every cached node, and `jobs
watch` fans that out into one `execution_cached` event per node, exactly as
`comfy run` does, so a consumer written against the run dialect counts cached
nodes correctly on either stream. `jobs watch` has no workflow graph in hand,
so it omits the optional `title` / `class_type` fields that `comfy run`
attaches to `executing`, `execution_cached` and `executed`; it also does not
carry the `outputs` array on `executed`, reporting each artifact as its own
`output` event instead. *(Changed since 1.16.0, whose `jobs watch` emitted one
`execution_cached` carrying a `nodes` array — a run-dialect consumer counting
cached nodes per event undercounted every cached node beyond the first. If you
wrote a watch consumer against 1.16.0 and read `ev["nodes"]` on this event,
read `ev["node"]` instead, once per event. The `nodes` array form is still
accepted by the published event schema so a stream captured from 1.16.0 keeps
validating, but nothing emits it.)*

## Overview

When `--json` is passed, `comfy run` switches into a strict
machine-readable mode:

- **stdout** carries exclusively **NDJSON** (newline-delimited JSON): one
  JSON object per line, each terminated by `\n`. No ANSI, no progress bar,
  no headings. Each line is written and flushed to stdout as soon as the
  underlying event is produced; agents may rely on read-as-emitted timing —
  there is no batching (the `progress` event is throttled to ~10 Hz per node).
- The stream is UTF-8. Non-ASCII characters in string fields are emitted
  as-is (`json.dumps(..., ensure_ascii=False)`).
- **stderr** carries human-readable side messages plus anything the CLI
  cannot route through the JSON contract: framework-level Python errors,
  uncaught exceptions, library warnings. Agents should not parse stderr;
  they may discard it or capture it for diagnostics.
- **Exit code** is `0` when the final envelope has `ok: true`, `130` when
  the run was cancelled (`error.code: "cancelled"` — Ctrl-C or a
  server-side interrupt), and `1` for every other failure. Fine-grained
  error categorisation is carried in `error.code`, not in the exit code.

In `--json` mode, `--verbose` has no effect: agents receive the full event
stream regardless.

**Workflow input format.** `--workflow` accepts both the ComfyUI **API
format** (the canonical `{node_id: {class_type, inputs, ...}}` graph
produced by "Save (API Format)") and the **exported UI format** (the
`{nodes: [...], links: [...]}` shape produced by "Save"). UI workflows
are converted to API format client-side via `/object_info` before
queuing; conversion is signalled by a [`converted`](#converted) event
emitted before [`queued`](#queued). API-format input does not produce a
`converted` event.

All duration fields in this contract are floats representing seconds.
Numeric count fields (e.g., `progress.completed` / `total`) are JSON
`number` and may be int or float depending on the underlying node.

## Stream shape

Every line on stdout is a JSON object with two universal fields:

| Field    | Type | Description                                                       |
| -------- | ---- | ----------------------------------------------------------------- |
| `schema` | str  | Contract version: `"event/1"` on events, `"envelope/1"` on the final line |
| `type`   | str  | Discriminator. Agents must dispatch on this field.                |

The stream always ends with exactly one line of `type: "envelope"`:

```json
{"schema": "envelope/1", "type": "envelope", "ok": true, "command": "run", "version": "1.6.1", "where": "local", "data": {...}, "error": null}
```

| Field     | Type         | Description                                                     |
| --------- | ------------ | --------------------------------------------------------------- |
| `ok`      | bool         | `true` on success, `false` on failure                           |
| `command` | str          | The subcommand (`"run"`)                                        |
| `version` | str          | comfy-cli version                                               |
| `where`   | str \| null  | Target this invocation was routed to: `"local"` or `"cloud"`. Set as soon as routing resolves, so client-side failures that never reach that backend still carry it (e.g. `workflow_not_found`). `null` for commands that don't route, and for errors raised *before* routing resolves (e.g. `where_invalid`). |
| `data`    | dict \| null | Result payload on success (see [Success envelope](#success-envelope)) |
| `error`   | dict \| null | Error object on failure (see [Error object](#error-object))     |

### Stream archetypes

| Outcome                   | Stream                                                                       |
| ------------------------- | ---------------------------------------------------------------------------- |
| Success (`--wait`)        | `[converted]? + prompt_preview + queued + [node events]* + envelope(ok)`     |
| `--no-wait` queued (default) | `[converted]? + prompt_preview + queued + envelope(ok, data.status="queued")` |
| `--print-prompt`          | `[converted]? + prompt_preview + envelope(ok, data.status="preview")`        |
| Failure mid-execution     | `[converted]? + prompt_preview + queued + [node events]* + envelope(error)`  |
| Failure at validation, consent, or submission | `[converted]? + prompt_preview + envelope(error)`         |
| Failure before the graph is parsed | `[converted]? + envelope(error)`                                    |

Where `[node events]*` is zero or more interleaved `execution_cached`,
`executing`, `progress`, `executed`, and `output` events. `[X]?` means X
may or may not appear. An error envelope can replace any non-terminal
line, ending the stream early.

The last two rows split on **whether the CLI has a parsed graph in hand
yet**, because `prompt_preview` is emitted as soon as it does — before any
check that could still refuse the run:

- **Before the graph is parsed** — the workflow file is missing, unreadable,
  or not JSON; a UI→API conversion failed (`conversion_error`,
  `conversion_crash`, `cql_no_graph`); the graph is empty
  (`workflow_empty`) or not in API format (`workflow_not_api_format`); no
  local server is running (`server_not_running`). These emit a bare error
  envelope. `converted` can precede it in exactly one case: conversion
  succeeded but its output still failed API-format classification.
- **After it is parsed** — the CQL pre-flight (`workflow_unknown_nodes` and
  friends), the `spend_consent_required` consent gate, cloud authentication
  (`cloud_unauthorized`), and the submit call itself all run *after*
  `prompt_preview`. A run refused by any of them therefore emits
  `prompt_preview` first and *then* the error envelope — the previewed graph
  is what the CLI *would* have submitted, not a promise that it did.

This ordering is identical on both targets. Note that `prompt_preview`
carries the full workflow graph, so `--json-stream` output should be treated
as sensitive if custom nodes embed local paths or credential-like widget
values in it. Events are emitted in `--json-stream` mode only — neither
pretty nor plain `--json` mode ever writes a `prompt_preview` line.

These archetypes hold for **both** `--where local` and `--where cloud`, with
one exception: `comfy run --where cloud` produces no `[node events]*` — use
`comfy jobs watch --where cloud` for in-flight cloud progress (see
[Per-target differences](#per-target-differences)). Everything else — the
`converted` / `prompt_preview` / `queued` prefix, the single terminal
envelope, and the exit-code mapping — is identical on both targets.

## Per-target differences

`comfy run` has two execution targets — `--where local` (the default: an HTTP
submit plus a WebSocket session against a ComfyUI server you run) and
`--where cloud` (an HTTPS submit plus polling against Comfy Cloud). They emit
the **same event dialect** and the same envelope framing, but the targets are
not the same machine and a few things genuinely cannot match. The complete
list of differences:

| Aspect | `--where local` | `--where cloud` |
| ------ | --------------- | --------------- |
| Per-node events (`executing`, `execution_cached`, `progress`, `executed`, `output`, `execution_error`) | Emitted, streamed live from the server WebSocket | Not emitted by `comfy run`. The cloud API is polled for a terminal record, so a `--wait` run goes straight from `queued` to the final envelope. For in-flight cloud progress, watch the job instead: `comfy --json-stream jobs watch <prompt_id> --where cloud` emits a coarse `state` event per status transition plus an `output` event per artifact |
| `queued.validation_warnings` | May be non-empty: the server can accept a prompt (HTTP 200) while reporting per-node issues | Always `[]` — the cloud rejects any submit carrying `node_errors` outright, as a `prompt_rejected` error envelope |
| `queued.base_url` | Absent | Present — the cloud endpoint the prompt was submitted to |
| Envelope `where` | `"local"` | `"cloud"` |
| Envelope target fields | `data.host` (str), `data.port` (int) | `data.base_url` (str) |
| Envelope `data.cached_node_ids` / `data.executed_node_ids` | Present on `--wait` | Absent — they are derived from the per-node event stream, which the cloud has none of |
| Envelope `data.warnings` | Absent | Present on `--wait` success: an array of non-fatal warning objects (currently only `partial_execution`, see below). `[]` when there are none |
| `prompt_rejected` `details` | `status` (400) and `node_errors` | `node_errors` only — the cloud reports rejected nodes on an otherwise-2xx submit, so there is no 4xx status to report. The `node_errors` value has the same [array-of-records shape](#node_errors-shape) on both targets |
| Error codes | The local set below | The workflow/pre-flight codes and the node-failure codes (`execution_error`, `transient_auth`, `prompt_rejected`, `cancelled`, `spend_consent_required`) plus the cloud-only codes below. The local-server- and WebSocket-specific codes cannot occur: `server_not_running`, `object_info_unavailable`, `connection_error`, `ws_timeout`, `ws_disconnected`, `invalid_response`, `client_error`, `server_error`, `partner_node_requires_credential` (the cloud injects the caller's credential itself) |

Everything not in that table is the same on both targets, including
`converted`, `prompt_preview`, the `queued` field set, `data.status`,
`data.prompt_id` / `client_id` / `outputs` / `outputs_by_node` /
`outputs_by_item` / `state_file` / `watcher_spawned` / `elapsed_seconds`, and
the `--print-prompt` and `--no-wait` stream shapes.

### Cloud-only error codes

| `code`               | Triggered when                                                     | `details`                       | Exit |
| -------------------- | ------------------------------------------------------------------ | -------------------------------- | ---- |
| `cloud_unauthorized` | No usable cloud session, or the session was rejected — run `comfy cloud login` | —                     | 1 |
| `cloud_http_error`   | The cloud API returned a non-2xx response on submit or while polling | `status` (int), `body` (str) on submit; `status`, `prompt_id` while polling | 1 |
| `cloud_timeout`      | The cloud job produced no progress for `--timeout` seconds          | `prompt_id` (str)               | 1 |
| `cql_no_graph`       | A UI-format workflow needs the cloud `object_info` snapshot to be lowered to API format, and it could not be loaded — run `comfy nodes refresh --where cloud` | — | 1 |

All of these are registered in `comfy_cli/error_codes.py` and listed by
`comfy --json discover`, exactly like the local codes.

### `partial_execution` warning

The cloud prunes workflow branches that fail server-side validation and still
reports the job as completed. On a `--wait` success the CLI diffs the output
nodes it submitted against the ones that returned outputs and, when some are
missing, appends a warning object to `data.warnings` rather than passing the
run off as a clean success:

```json
{"code": "partial_execution", "message": "submitted 2 output node(s) but the cloud returned outputs for only 1; 1 branch(es) were pruned server-side (likely failed validation) and produced nothing", "submitted_output_nodes": 2, "returned_output_nodes": 1}
```

The envelope is still `ok: true` (exit `0`) — the job did run. Agents that
need all-or-nothing semantics should check `data.warnings` is empty.

## Event reference

| `type`             | When                                                 |
| ------------------ | ---------------------------------------------------- |
| `converted`        | UI-format workflow was client-side converted         |
| `prompt_preview`   | The API-format workflow graph about to be submitted  |
| `queued`           | Server accepted the prompt (HTTP 200 on `/prompt`)   |
| `execution_cached` | Node hit the execution cache and was skipped         |
| `executing`        | Node started execution                               |
| `progress`         | In-flight progress update for the running node       |
| `executed`         | Node finished and reported its outputs               |
| `output`           | One file-like output became available (`url`)        |
| `execution_error`  | Server reported a node exception (error envelope follows) |
| `login_url`        | `comfy cloud login`: OAuth authorize URL, before the browser-callback wait |

Agents must ignore events whose `type` they do not recognise — new event
kinds may be added in a backward-compatible manner. Agents must ignore
unknown fields on known events for the same reason.

A handful of fields carry values from a server-defined open set rather
than a fixed enumeration: `class_type`, `category`, `type` (output folder),
and `exception_type`. Agents must accept and pass through unknown values
without keying behaviour on specific strings.

Every per-node event also carries a `title` field — the human-readable
label to show in a per-node UI: **`_meta.title` if present, else
`class_type`, else the node id**.

### `converted`

Emitted once if the input workflow was in UI format and was converted to
API format client-side.

```json
{"schema": "event/1", "type": "converted", "node_count": 2}
```

### `prompt_preview`

Emitted once the workflow has been successfully loaded, parsed, and (if
UI-format) converted — i.e., in every stream except the **Failure
pre-flight** archetype. Carries the API-format workflow graph the CLI is
about to POST to `/prompt` — the same dict that would land in the
request's `prompt` field. It does **not** include `client_id` or
`extra_data` (so any `--api-key` value never appears here).

Under `--print-prompt` this is the only event: the CLI emits it plus the
final envelope and exits 0 without queuing.

```json
{"schema": "event/1", "type": "prompt_preview", "prompt": {"1": {"class_type": "EmptyLatentImage", "inputs": {"width": 512, "height": 512, "batch_size": 1}}}}
```

### `queued`

Emitted after the submit request returns success — `POST /prompt` returning
200 locally, the equivalent cloud submit under `--where cloud` — **and** after
the job's state file has been persisted.

```json
{
  "schema": "event/1",
  "type": "queued",
  "prompt_id": "9b1c…",
  "client_id": "fe2a…",
  "validation_warnings": [],
  "nodes": [
    {"node_id": "1", "class_type": "GeminiNanoBanana2", "title": "Nano Banana 2"},
    {"node_id": "2", "class_type": "SaveImage", "title": "Save Image"}
  ]
}
```

| Field                 | Type          | Description                                            |
| --------------------- | ------------- | ------------------------------------------------------ |
| `prompt_id`           | str           | Server-assigned prompt UUID                            |
| `client_id`           | str           | Client-generated UUID (sent with `/prompt`)            |
| `validation_warnings` | array of dict | Per-node validation issues ComfyUI reported alongside a successful queue (some output chains validated, others didn't). Same record shape as `prompt_rejected`'s `details.node_errors` (see [shape](#node_errors-shape)). Empty (`[]`) in the common case. |
| `nodes`               | array of dict | Manifest of every node in the submitted (post-conversion) workflow: `node_id` (str), `class_type` (str), `title` (str). Lets piped consumers render a per-node UI without the workflow file. |
| `base_url`            | str           | **Cloud only.** The cloud endpoint the prompt was submitted to. Absent on `--where local`. |

`queued` is emitted **after** the submit call returns successfully **and after
the job's state file has been persisted** — on the async (`--no-wait`) paths,
after the background watcher spawn attempt too. On both targets it therefore
means more than "the server has the prompt": the durable record exists, so a
consumer may run `comfy jobs status <prompt_id>` — or expect the job in
`comfy jobs ls` — the moment it reads this line. A run whose submit fails emits
its error envelope with no `queued` line at all.

### `executing`

Emitted when a node starts execution. Two consecutive `executing` events
with different `node` values are normal (intermediate compute nodes whose
outputs aren't published to the client never fire `executed`); agents that
track a "current node" should treat a new `executing` as implicitly
closing the previous one.

```json
{"schema": "event/1", "type": "executing", "node": "2", "title": "Save Image", "class_type": "SaveImage", "prompt_id": "9b1c…"}
```

### `execution_cached`

One event per node whose outputs were retrieved from the execution cache
(from ComfyUI's `execution_cached` websocket message, which lists every
cached node in one frame and is fanned out here). Same fields as
`executing`. A cached output-bearing node (e.g., a cached `SaveImage`)
may emit both `execution_cached` AND `executed`.

```json
{"schema": "event/1", "type": "execution_cached", "node": "1", "title": "Latent", "class_type": "EmptyLatentImage", "prompt_id": "9b1c…"}
```

`comfy jobs watch` emits the same per-node event without `title` /
`class_type` (see [`jobs watch` attaches as the submitting
session](#jobs-watch-attaches-as-the-submitting-session)). It never emits a
list-shaped `nodes` field on this event.

### `progress`

Per-step progress for samplers, video encoders, and any node that calls
`ProgressBar.update_absolute(...)`. Throttled to ~10 events/second per
node. Same field names as the `comfy jobs watch` stream.

```json
{"schema": "event/1", "type": "progress", "node": "1", "completed": 14, "total": 30, "prompt_id": "9b1c…"}
```

Some custom nodes may emit `completed > total` near the end of execution;
agents rendering a progress bar should clamp.

### `executed`

Emitted when the server reports node completion via its `executed`
websocket message. **Not guaranteed for every executed node** —
intermediate compute nodes that don't surface output to the client may
finish without it.

```json
{
  "schema": "event/1",
  "type": "executed",
  "node": "2",
  "title": "Save Image",
  "class_type": "SaveImage",
  "outputs": [
    {
      "category": "images",
      "node_id": "2",
      "class_type": "SaveImage",
      "title": "Save Image",
      "filename": "banana_test_00001_.png",
      "subfolder": "",
      "type": "output",
      "url": "http://127.0.0.1:8188/view?filename=banana_test_00001_.png&subfolder=&type=output"
    }
  ],
  "prompt_id": "9b1c…"
}
```

`outputs` is populated by iterating each key in ComfyUI's
`executed.output` dict and emitting any item that matches the
file-record shape (a dict containing a `filename` key). Items that are
not file-record-shaped (strings, booleans, mixed lists from nodes that
publish non-file data like text predictions or animation flags) are
silently skipped. See [Output object](#output-object).

### `output`

One event per newly seen file-like output, emitted right after the
`executed` event that produced it. `url` is the contractual way to fetch
the bytes (it points at ComfyUI's `/view` endpoint); on a local loopback
run with a resolvable workspace it may instead be an absolute local file
path — treat any non-`http(s)` value as a filesystem path.

```json
{"schema": "event/1", "type": "output", "url": "http://127.0.0.1:8188/view?filename=banana_test_00001_.png&subfolder=&type=output", "prompt_id": "9b1c…"}
```

### `execution_error`

Diagnostic event carrying the raw server payload when a node raises
during execution. The terminal error envelope (`error.code:
"execution_error"`) follows immediately; agents can dispatch on the
envelope alone.

```json
{"schema": "event/1", "type": "execution_error", "prompt_id": "9b1c…", "details": {"node_id": "1", "exception_message": "API key invalid", "...": "..."}}
```

### `login_url`

Emitted by `comfy cloud login` under `--json`/`--json-stream` — this event is
part of the sign-in stream, not the `run` stream. It carries the OAuth
authorize URL as soon as it is built, and is flushed **before** the command
blocks (up to `timeout_s` seconds) waiting for the loopback browser callback,
so a parent process driving login headlessly can open (or forward) `url` in
time. If the callback never arrives, the terminal envelope is an
`oauth_timeout` error; on success it is the login envelope (`data.action:
"login"`, `data.session` with tokens redacted).

```json
{"schema": "event/1", "type": "login_url", "url": "https://api.comfy.org/oauth/authorize?...", "timeout_s": 300}
```

## Success envelope

On `--wait` success, `data` carries:

| Field               | Type            | Description                                            |
| ------------------- | --------------- | ------------------------------------------------------ |
| `workflow`          | str             | Absolute path of the submitted workflow file           |
| `status`            | str             | `"completed"` (`"queued"` without `--wait`; `"preview"` under `--print-prompt`) |
| `prompt_id`         | str             | Server-assigned prompt UUID                            |
| `client_id`         | str             | Client-generated UUID                                  |
| `outputs`           | array of str    | URL (or local path) per file-like output, deduplicated |
| `outputs_by_node`   | dict            | The same outputs grouped by the node id that produced them |
| `outputs_by_item`   | dict            | The same outputs grouped by `compose` foreach item; `{}` when the workflow carried no item map |
| `cached_node_ids`   | array of str    | **Local only.** Node IDs the server reported as cached  |
| `executed_node_ids` | array of str    | **Local only.** Node IDs the executor *ran* — the union of every node that appeared in an `executing` or `executed` event, including intermediate compute nodes |
| `warnings`          | array of dict   | **Cloud only.** Non-fatal warnings about the completed run — see [`partial_execution`](#partial_execution-warning). `[]` when there are none |
| `elapsed_seconds`   | float \| null   | Wall-clock duration (null when not waiting)            |
| `host` / `port`     | str / int       | **Local only.** Target server                          |
| `base_url`          | str             | **Cloud only.** Target cloud endpoint                  |
| `state_file`        | str \| null     | Path of the job state file (poll with `comfy jobs status`) |

`cached_node_ids` and `executed_node_ids` may overlap: a cached
output-bearing node emits both `execution_cached` and `executed`. Agents
wanting "ran fresh, not from cache" should compute
`set(executed_node_ids) - set(cached_node_ids)`. Both are derived from the
per-node event stream and so are local-only — see
[Per-target differences](#per-target-differences).

Without `--wait` (the default), the stream ends at the `queued` envelope
(`data.status: "queued"`, `data.watcher_spawned: bool`) and a detached
watcher keeps the state file updated; follow up with
`comfy jobs watch <prompt_id>` or `comfy jobs status <prompt_id>`. This is
the same on both targets, `watcher_spawned` included (add `--where cloud` to
the follow-up commands for a cloud job). The async envelope carries no
`outputs_by_node` / `outputs_by_item` / `warnings` — nothing has run yet.

### Known limit: `--wait` has no background watcher

`--wait` writes the job's state file at submit and finalizes it in the
foreground — it does **not** spawn the detached watcher the async path does
(`data.watcher_spawned` is an async-path field only). Every ordinary outcome is
still recorded: a node failure, a cancel, and a lost connection to a dying
server all run through the foreground handlers, which write `error.code`
(the classified execution verdict or `execution_error`; `cancelled`;
`server_died`) before exiting.

A `--wait` process that is **killed from outside** — a caller-imposed timeout
(`SIGKILL`/`SIGTERM` on the process group), a terminal going away, the OS
reaping the CLI alongside the server — runs no handler, so the state file is
left at its submit-time `running`. To cover that, `--wait` (local and cloud)
stamps its own `watcher_pid` (+ start time) on the submit-time record: the next
`jobs ls` finds a non-terminal record whose recorded pid is dead, and its
stale-watcher reap finalizes the job as `error` with `error_code:
"watcher_crashed"` — the same treatment a crashed background watcher gets — and
`jobs ls --orphaned` lists it. The stamp is dropped again on the exits where
the job may genuinely still be alive on the server, so the reap can't claim a
crash the CLI hasn't established: when local `--wait` gives up on its *own*
`--timeout` (`ws_timeout`), and when cloud `--wait` dies on a network error
that escapes its handlers (a DNS failure or connection reset while polling, as
opposed to the handled `cloud_timeout` / `cloud_unauthorized` /
`cloud_http_error` exits, which record a terminal verdict of their own). In
both cases the record is left non-terminal with no pid, the reap never touches
it, and `comfy jobs status <prompt_id>` can still consult the server for the
real outcome.

The reap itself never overwrites a verdict that landed first: it re-reads each
record under that record's lock before rewriting it, so a `--wait` run
finishing normally in the same instant keeps its `completed` status and its
outputs.

What the reap cannot tell you is *why* the process died, or what happened to the
job afterwards — `watcher_crashed` records the watcher's death, not the job's
outcome. **For runs that may outlive the caller's patience, submit without
`--wait`** — the async path spawns a watcher that survives the parent (its own
session/process group), and it is the watcher that writes `server_died` when the
server disappears mid-job. `comfy jobs ls` then reports both the status and the
`error_code`.

A watcher is deliberately *not* spawned on `--wait`: it would put a second,
independent writer on the state file the foreground already finalizes, add a
second server connection per foreground run, and leave a background process
behind after a synchronous command returns — a real regression on the common
path in exchange for a case the async path already covers.

## `comfy validate --json` envelope

`comfy validate --workflow <file> --json` checks a workflow without submitting
it and emits a single envelope (no event stream). On a valid workflow `ok` is
`true` and `data` carries:

| Field                  | Type          | Description                                                                                     |
| ---------------------- | ------------- | ----------------------------------------------------------------------------------------------- |
| `workflow`             | str           | Absolute path of the validated workflow file                                                    |
| `valid`                | bool          | Whether the graph passed validation (also the envelope `ok`)                                     |
| `error_count`          | number        | Number of entries in `errors`                                                                    |
| `warning_count`        | number        | Number of entries in `warnings`                                                                  |
| `errors`               | array of dict | Per-node validation errors (empty when `valid`)                                                  |
| `warnings`             | array of dict | Non-fatal validation warnings                                                                    |
| `partner_nodes`        | array of str  | Sorted class_types in the workflow that are partner-API (paid) nodes. **Always present**; `[]` when none |
| `spends_credits`       | bool          | `true` iff `partner_nodes` is non-empty — a convenience flag for "will this workflow spend Comfy credits?" |
| `converted_from_ui`    | bool          | Present and `true` only when the input was a UI export that was lowered to API format before validating |
| `converted_node_count` | number        | Present only alongside `converted_from_ui`: node count of the converted graph                    |

`partner_nodes` / `spends_credits` are **informational only** — they never
change validate's exit code (validate stays advisory; the credit-spend gate
lives in `comfy run`). Detection is the same authoritative `api_node: true`
flag (with a `partner/...` category fallback) that `comfy run` uses, run over
the loaded `object_info`; in offline `--input` mode where the `object_info`
lacks `api_node` flags the list is simply empty (fail-open). When
`spends_credits` is `true`, pretty (non-`--json`) mode also prints a yellow
`⚠ uses partner-API (paid) nodes …` line after the verdict.

Invalid workflows emit `ok: false` with the same `data` fields (`valid: false`,
a populated `errors` array, `partner_nodes`/`spends_credits` still present) and
exit code `1`. Structural failures (missing file, non-object JSON, an
unconvertible UI export) emit an [error object](#error-object) instead.

## Error object

Every failure envelope carries:

| Field     | Type         | Description                                                                |
| --------- | ------------ | -------------------------------------------------------------------------- |
| `code`    | str          | Registered discriminator — see [Error codes](#error-codes); the full registry is in `comfy --json discover` under `data.error_codes` |
| `message` | str          | Human-readable summary. For display only — dispatch on `code`              |
| `hint`    | str \| null  | Suggested next action                                                      |
| `details` | dict \| null | Per-code structured extras (documented below)                              |

## Error codes

Codes raised by `comfy run` against a local server, with their `details`
payloads. All of them are registered in `comfy_cli/error_codes.py` (the
registry test enforces this) and surfaced by `comfy discover`. For
`--where cloud`, see [Cloud-only error codes](#cloud-only-error-codes) — it
lists what the cloud adds and which of the codes below cannot occur there.

| `code`                    | Triggered when                                                                  | `details`                                          | Exit |
| ------------------------- | ------------------------------------------------------------------------------- | -------------------------------------------------- | ---- |
| `workflow_not_found`      | `--workflow` path does not exist                                                | —                                                  | 1 |
| `workflow_invalid_json`   | Workflow file is not valid JSON                                                 | —                                                  | 1 |
| `workflow_read_error`     | Workflow file exists but isn't readable as text (`OSError`, `UnicodeDecodeError`) | —                                                | 1 |
| `workflow_not_api_format` | File parses but is neither UI nor API format                                    | —                                                  | 1 |
| `workflow_empty`          | Workflow has no executable nodes (UI conversion produced `{}`, or API workflow is `{}`) | —                                          | 1 |
| `conversion_error`        | UI→API converter raised `WorkflowConversionError`                               | —                                                  | 1 |
| `conversion_crash`        | UI→API converter raised an unexpected exception                                 | `exception_type` (str)                             | 1 |
| `server_not_running`      | Pre-flight probe found no ComfyUI on host:port                                  | `host` (str), `port` (int)                         | 1 |
| `object_info_unavailable` | `/object_info` returned an HTTP error, or HTTP 200 with an unparseable body     | `status` (int), `body` (str)                       | 1 |
| `connection_error`        | Server unreachable mid-flow: `URLError`, `TimeoutError`, or other `OSError` (including on `/object_info`) | —                       | 1 |
| `workflow_unknown_nodes`  | Pre-submit validation found unknown class_types / shape mismatches              | `errors` (array), `warnings` (array)               | 1 |
| `partner_node_requires_credential` | Workflow uses a partner-API node and no `api_key_comfy_org` credential is available | `partner_nodes` (array of str, capped at 20 entries × 64 chars each), `partner_node_count` (int, the exact total — read this, not `len(partner_nodes)`), `host`, `port` | 1 |
| `spend_consent_required`  | Workflow embeds partner-API (paid) nodes and `--allow-spend` was not passed (machine mode) or interactive consent was declined; re-run with `--allow-spend`. Free (non-partner) workflows are unaffected. | `partner_nodes` (array of str, capped at 20 entries × 64 chars each), `partner_node_count` (int, the exact total — read this, not `len(partner_nodes)`); local path also carries `host`, `port`, the cloud path carries `where: "cloud"` | 1 |
| `prompt_rejected`         | Server returned HTTP 400 with `node_errors`                                     | `status` (400), `node_errors` (array — [shape](#node_errors-shape)) | 1 |
| `client_error`            | Server returned another HTTP 4xx response                                       | `status` (int, 4xx), `body` (str)                  | 1 |
| `server_error`            | Server returned an HTTP 5xx response                                            | `status` (int, 5xx), `body` (str)                  | 1 |
| `invalid_response`        | Server returned HTTP 2xx but body was unparseable or lacked `prompt_id`         | `status` (int, 2xx)                                | 1 |
| `ws_timeout`              | WebSocket `recv` idle past `--timeout`                                          | `timeout` (int, seconds)                           | 1 |
| `ws_disconnected`         | WebSocket connection dropped mid-execution                                      | —                                                  | 1 |
| `cancelled`               | Run was interrupted — client `SIGINT` (Ctrl-C) or the server's `execution_interrupted` (e.g. `/interrupt`) | —                       | 130 |
| `execution_error`         | A node raised during execution (server emitted `execution_error`)               | `node_id` (str), `class_type` (str), `title` (str), `exception_type` (str), `traceback` (str) | 1 |
| `transient_auth`          | The `execution_error` cause was an API node's server-side session token expiring mid-execution — transient, so resubmitting the same workflow succeeds. Local credentials are fine; `comfy cloud login` does not help | Same fields as `execution_error` | 1 |

### `exception_type` field

Provided for diagnostic and observability purposes (e.g., metrics
bucketing). **Open set** — the format is whatever ComfyUI sends,
typically the bare class name for builtins (`RuntimeError`,
`ValueError`) and a dotted module path for non-builtins. May be `""`
when the server omits it. Don't key retry or routing logic on it; use
`error.code` for coarse dispatch.

### `traceback` field

A single multi-line string carrying the formatted stack frames as
reported by ComfyUI (joined from the server's `traceback.format_tb()`
output). It does NOT include the `"Traceback (most recent call last):"`
header or the final `"ExceptionType: message"` summary line. May be
empty (`""`).

### `node_errors` shape

Used for `prompt_rejected`'s `details.node_errors` and for
`queued.validation_warnings`. The value is an array of self-contained
records, one per affected node. Each record carries `node_id` (str —
same identifier as in the per-node events) plus the per-node fields
ComfyUI emits:

```json
"node_errors": [
  {
    "node_id": "1",
    "errors": [
      {
        "type": "value_not_in_list",
        "message": "Value not in list",
        "details": "resolution: '5K' not in ['1K','2K','4K']",
        "extra_info": {"input_name": "resolution", "received_value": "5K"}
      }
    ],
    "dependent_outputs": ["2"],
    "class_type": "GeminiNanoBanana2"
  }
]
```

The inner per-node fields are defined by ComfyUI's `validate_prompt()`
and may evolve with ComfyUI versions — agents should ignore unknown
fields. The CLI guarantees only that the outer value is an array of
dicts, each carrying a `node_id` (str).

If a server reports a bare value instead of the per-node dict above (e.g.
`{"1": "missing input"}`), the CLI does not drop the record — it wraps the
value as `{"node_id": "1", "errors": ["missing input"]}` so the count in the
message always matches the array and the diagnostic survives. `errors` items
are objects in the normal case; treat a non-object item as an opaque message.
`node_id` is always taken from the payload's own map key, so a server-supplied
`node_id` field inside a record cannot override it.

## Output object

Entries of `executed.outputs`:

| Field        | Type | Description                                                                       |
| ------------ | ---- | ---------------------------------------------------------------------------------- |
| `category`   | str  | Output category as keyed by ComfyUI's `executed.output` dict. **Open set** (`images`, `audio`, `3d`, `latents`, …) |
| `node_id`    | str  | Node that produced the output                                                      |
| `class_type` | str  | Node class name                                                                    |
| `title`      | str  | Display label                                                                      |
| `filename`   | str  | Raw filename as reported by the server                                             |
| `subfolder`  | str  | Subfolder within the output root. Defaults to `""`                                 |
| `type`       | str  | ComfyUI output folder discriminator. **Open set** (`output`, `temp`, `input`). Defaults to `"output"` |
| `url`        | str  | `http(s)://<host>:<port>/view?...` URL — fetch this to get the bytes               |

## Process-level termination

The CLI may be terminated by the operating system or a parent process
(SIGKILL, SIGTERM, OOM-kill, segmentation fault). In these cases the
envelope may never be emitted and the stream may be truncated. Agents
should treat the run as failed when **both**:
- the process exit code is non-zero, and
- the last line on stdout is not a `type: "envelope"` line, or stdout is
  empty.

Stderr may contain a Python traceback in these cases.

## Examples

### Successful run (UI-format input, `--wait`)

```json
{"schema":"event/1","type":"converted","node_count":2}
{"schema":"event/1","type":"prompt_preview","prompt":{"1":{"class_type":"GeminiNanoBanana2","inputs":{"prompt":"a banana"},"_meta":{"title":"Nano Banana 2"}},"2":{"class_type":"SaveImage","inputs":{"filename_prefix":"banana_test","images":["1",0]},"_meta":{"title":"Save Image"}}}}
{"schema":"event/1","type":"queued","prompt_id":"9b1c…","client_id":"fe2a…","validation_warnings":[],"nodes":[{"node_id":"1","class_type":"GeminiNanoBanana2","title":"Nano Banana 2"},{"node_id":"2","class_type":"SaveImage","title":"Save Image"}]}
{"schema":"event/1","type":"executing","node":"1","title":"Nano Banana 2","class_type":"GeminiNanoBanana2","prompt_id":"9b1c…"}
{"schema":"event/1","type":"progress","node":"1","completed":4,"total":4,"prompt_id":"9b1c…"}
{"schema":"event/1","type":"executing","node":"2","title":"Save Image","class_type":"SaveImage","prompt_id":"9b1c…"}
{"schema":"event/1","type":"executed","node":"2","title":"Save Image","class_type":"SaveImage","outputs":[{"category":"images","node_id":"2","class_type":"SaveImage","title":"Save Image","filename":"banana_test_00001_.png","subfolder":"","type":"output","url":"http://127.0.0.1:8188/view?filename=banana_test_00001_.png&subfolder=&type=output"}],"prompt_id":"9b1c…"}
{"schema":"event/1","type":"output","url":"http://127.0.0.1:8188/view?filename=banana_test_00001_.png&subfolder=&type=output","prompt_id":"9b1c…"}
{"schema":"envelope/1","type":"envelope","ok":true,"command":"run","version":"1.6.1","where":"local","data":{"workflow":"/path/wf.json","status":"completed","prompt_id":"9b1c…","client_id":"fe2a…","outputs":["http://127.0.0.1:8188/view?filename=banana_test_00001_.png&subfolder=&type=output"],"cached_node_ids":[],"executed_node_ids":["1","2"],"elapsed_seconds":8.342,"host":"127.0.0.1","port":8188,"state_file":"…"},"error":null}
```

Exit code: `0`.

### Successful cloud run (`--where cloud --wait`)

Same prefix as the local stream; no per-node events, because the cloud path
polls for a terminal record instead of streaming a WebSocket session.

```json
{"schema":"event/1","type":"prompt_preview","prompt":{"1":{"class_type":"GeminiNanoBanana2","inputs":{"prompt":"a banana"}},"2":{"class_type":"SaveImage","inputs":{"filename_prefix":"banana_test","images":["1",0]}}}}
{"schema":"event/1","type":"queued","prompt_id":"9b1c…","client_id":"fe2a…","validation_warnings":[],"nodes":[{"node_id":"1","class_type":"GeminiNanoBanana2","title":"GeminiNanoBanana2"},{"node_id":"2","class_type":"SaveImage","title":"SaveImage"}],"base_url":"https://api.comfy.org"}
{"schema":"envelope/1","type":"envelope","ok":true,"command":"run","version":"1.6.1","where":"cloud","data":{"workflow":"/path/wf.json","status":"completed","prompt_id":"9b1c…","client_id":"fe2a…","outputs":["https://…/banana_test_00001_.png"],"outputs_by_node":{"2":["https://…/banana_test_00001_.png"]},"outputs_by_item":{},"warnings":[],"elapsed_seconds":21.7,"base_url":"https://api.comfy.org","state_file":"…"},"error":null}
```

Exit code: `0`.

### Failure: workflow file missing

```json
{"schema":"envelope/1","type":"envelope","ok":false,"command":"run","version":"1.6.1","where":"local","data":null,"error":{"code":"workflow_not_found","message":"Specified workflow file not found: /tmp/missing.json","hint":"check the path; pass the API-format JSON exported from ComfyUI","details":null}}
```

Exit code: `1`.

### Failure: server returned validation errors

```json
{"schema":"event/1","type":"prompt_preview","prompt":{"…":"…"}}
{"schema":"envelope/1","type":"envelope","ok":false,"command":"run","version":"1.6.1","where":"local","data":null,"error":{"code":"prompt_rejected","message":"Workflow has 1 validation error(s)","hint":"inspect `details.node_errors` and fix the workflow","details":{"status":400,"node_errors":[{"node_id":"1","errors":[{"type":"value_not_in_list","message":"Value not in list","details":"resolution: '5K' not in ['1K','2K','4K']"}],"dependent_outputs":["2"],"class_type":"GeminiNanoBanana2"}]}}}
```

Exit code: `1`.

### Failure: node raised during execution

```json
{"schema":"event/1","type":"prompt_preview","prompt":{"…":"…"}}
{"schema":"event/1","type":"queued","prompt_id":"9b1c…","client_id":"fe2a…","validation_warnings":[],"nodes":[{"…":"…"}]}
{"schema":"event/1","type":"executing","node":"1","title":"Nano Banana 2","class_type":"GeminiNanoBanana2","prompt_id":"9b1c…"}
{"schema":"event/1","type":"execution_error","prompt_id":"9b1c…","details":{"node_id":"1","exception_message":"API key invalid","…":"…"}}
{"schema":"envelope/1","type":"envelope","ok":false,"command":"run","version":"1.6.1","where":"local","data":null,"error":{"code":"execution_error","message":"API key invalid","hint":"inspect the per-node fields in details; re-run with `--wait --verbose`","details":{"node_id":"1","class_type":"GeminiNanoBanana2","title":"Nano Banana 2","exception_type":"RuntimeError","traceback":"  File \"/path/to/node.py\", line 42, in execute\n    raise RuntimeError(\"API key invalid\")\n"}}}
```

Exit code: `1`.

### Cancellation (Ctrl-C or server interrupt)

```json
{"schema":"envelope/1","type":"envelope","ok":false,"command":"run","version":"1.6.1","where":"local","data":null,"error":{"code":"cancelled","message":"Cancelled by user","hint":null,"details":null}}
```

Exit code: `130`.

## Validating `data` against the shipped schemas

The per-command schemas live in `comfy_cli/schemas/`. Most are self-contained,
but the five discovery schemas — `templates.json`, `nodes.json`, `models.json`,
`generate_list.json`, `generate_schema.json` — reference a shared one:

```json
"knowledge": { "$ref": "knowledge_block.json" }
```

Each schema carries an `$id` under `https://comfy.org/schemas/`. That namespace
is an identifier, not a location; nothing is served there. A validator built on
one schema file alone therefore cannot resolve the reference, and validating a
`templates ls` payload that carries a `knowledge` block fails with
`Unresolvable: knowledge_block.json` rather than a validation error.

Build a registry from the whole directory first:

```python
import json
from pathlib import Path

import jsonschema
from referencing import Registry, Resource

SCHEMAS = Path("comfy_cli/schemas")  # or wherever you vendored them
registry = Registry().with_resources(
    (s["$id"], Resource.from_contents(s))
    for s in (json.loads(p.read_text()) for p in SCHEMAS.glob("*.json"))
    if "$id" in s
)

schema = json.loads((SCHEMAS / "templates.json").read_text())
jsonschema.Draft202012Validator(schema, registry=registry).validate(envelope["data"])
```

Any validator works the same way: load every `*.json` in the directory into
whatever the library calls its store, keyed by `$id`. Ship the directory as a
unit — a single schema file copied out on its own is not self-sufficient.

## Stability

### What is stable

- The two-schema framing: `event/1` lines + a final `envelope/1` /
  `type: "envelope"` line. Bump rule: additive optional fields = no bump;
  rename/remove/retype a field or changed exit semantics = bump.
- The set of event `type`s listed above and the field names within them.
- The set of `error.code` values listed above, their registration in the
  error-code registry, and the per-code `details` documented for each.
- The exit code mapping: `0` on `ok: true`, `130` on `cancelled`, `1` on
  every other failure.
- The stdout/stderr separation: stdout carries only NDJSON.

### What may change in a non-breaking way

- New event types being added (ignore unknown `type` values).
- New `error.code` values being added (default-handle unknown codes).
- New optional fields being added to existing events or to `data`
  (ignore unknown fields).

## Legacy dialect mapping

For consumers migrating from the pre-`event/1` `comfy run --json` dialect
(removed; was `{"event": …, "schema_version": 1}` with `error.kind`):

| Legacy event     | Current                                                      |
| ---------------- | ------------------------------------------------------------ |
| `converted`      | `type: "converted"`                                          |
| `prompt_preview` | `type: "prompt_preview"`                                     |
| `queued`         | `type: "queued"`                                             |
| `node_executing` | `type: "executing"` (`node_id` → `node`)                     |
| `node_cached`    | `type: "execution_cached"` (`node_id` → `node`)              |
| `node_progress`  | `type: "progress"` (`value`/`max` → `completed`/`total`; `class_type`/`title` dropped — read them from the `executing` event) |
| `node_executed`  | `type: "executed"` (+ one `output` event per file)           |
| `completed`      | envelope `ok: true` (`outputs` → `data.outputs` as URLs; `cached_node_ids` / `executed_node_ids` preserved in `data`) |
| `failed`         | envelope `ok: false` with `error.code`                       |

| Legacy `error.kind`        | Current `error.code`        | Exit |
| -------------------------- | --------------------------- | ---- |
| `workflow_not_found`       | `workflow_not_found`        | 1 |
| `workflow_invalid_json`    | `workflow_invalid_json`     | 1 |
| `workflow_read_error`      | `workflow_read_error`       | 1 |
| `workflow_format_invalid`  | `workflow_not_api_format`   | 1 |
| `workflow_empty`           | `workflow_empty`            | 1 |
| `conversion_error`         | `conversion_error`          | 1 |
| `conversion_crash`         | `conversion_crash`          | 1 |
| `connection_error` (probe) | `server_not_running`        | 1 |
| `connection_error` (network) | `connection_error`        | 1 |
| `object_info_unavailable`  | `object_info_unavailable`   | 1 |
| `validation_error`         | `prompt_rejected`           | 1 |
| `client_error`             | `client_error`              | 1 |
| `server_error`             | `server_error`              | 1 |
| `invalid_response`         | `invalid_response`          | 1 |
| `timeout`                  | `ws_timeout`                | 1 |
| `connection_lost`          | `ws_disconnected`           | 1 |
| `execution_interrupted`    | `cancelled`                 | 130 (was 1) |
| `execution_error`          | `execution_error`           | 1 |

Other framing changes: `schema_version: 1` → `schema: "event/1"`;
per-kind error extras moved under `error.details`; stdout is now UTF-8
(was ASCII-escaped); `--print-prompt` and `--no-wait` streams now end
with an explicit `ok: true` envelope instead of ending at
`prompt_preview` / `queued`.
