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
| `where`   | str \| null  | `"local"` or `"cloud"`                                          |
| `data`    | dict \| null | Result payload on success (see [Success envelope](#success-envelope)) |
| `error`   | dict \| null | Error object on failure (see [Error object](#error-object))     |

### Stream archetypes

| Outcome                   | Stream                                                                       |
| ------------------------- | ---------------------------------------------------------------------------- |
| Success (`--wait`)        | `[converted]? + prompt_preview + queued + [node events]* + envelope(ok)`     |
| `--no-wait` queued (default) | `[converted]? + prompt_preview + queued + envelope(ok, data.status="queued")` |
| `--print-prompt`          | `[converted]? + prompt_preview + envelope(ok, data.status="preview")`        |
| Failure mid-execution     | `[converted]? + prompt_preview + queued + [node events]* + envelope(error)`  |
| Failure during submission | `[converted]? + prompt_preview + envelope(error)`                            |
| Failure pre-flight        | `envelope(error)`                                                            |

Where `[node events]*` is zero or more interleaved `execution_cached`,
`executing`, `progress`, `executed`, and `output` events. `[X]?` means X
may or may not appear. An error envelope can replace any non-terminal
line, ending the stream early.

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

Emitted after `POST /prompt` returns 200.

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
(from ComfyUI's `execution_cached` websocket message). Same fields as
`executing`. A cached output-bearing node (e.g., a cached `SaveImage`)
may emit both `execution_cached` AND `executed`.

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

## Success envelope

On `--wait` success, `data` carries:

| Field               | Type            | Description                                            |
| ------------------- | --------------- | ------------------------------------------------------ |
| `workflow`          | str             | Absolute path of the submitted workflow file           |
| `status`            | str             | `"completed"` (`"queued"` without `--wait`; `"preview"` under `--print-prompt`) |
| `prompt_id`         | str             | Server-assigned prompt UUID                            |
| `client_id`         | str             | Client-generated UUID                                  |
| `outputs`           | array of str    | URL (or local path) per file-like output, deduplicated |
| `cached_node_ids`   | array of str    | Node IDs the server reported as cached                 |
| `executed_node_ids` | array of str    | Node IDs the executor *ran* — the union of every node that appeared in an `executing` or `executed` event, including intermediate compute nodes |
| `elapsed_seconds`   | float \| null   | Wall-clock duration (null when not waiting)            |
| `host` / `port`     | str / int       | Target server                                          |
| `state_file`        | str \| null     | Path of the job state file (poll with `comfy jobs status`) |

`cached_node_ids` and `executed_node_ids` may overlap: a cached
output-bearing node emits both `execution_cached` and `executed`. Agents
wanting "ran fresh, not from cache" should compute
`set(executed_node_ids) - set(cached_node_ids)`.

Without `--wait` (the default), the stream ends at the `queued` envelope
(`data.status: "queued"`, `data.watcher_spawned: bool`) and a detached
watcher keeps the state file updated; follow up with
`comfy jobs watch <prompt_id>` or `comfy jobs status <prompt_id>`.

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
registry test enforces this) and surfaced by `comfy discover`.

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
| `partner_node_requires_credential` | Workflow uses a partner-API node and no `api_key_comfy_org` credential is available | `partner_nodes` (array of str), `host`, `port` | 1 |
| `prompt_rejected`         | Server returned HTTP 400 with `node_errors`                                     | `status` (400), `node_errors` (array — [shape](#node_errors-shape)) | 1 |
| `client_error`            | Server returned another HTTP 4xx response                                       | `status` (int, 4xx), `body` (str)                  | 1 |
| `server_error`            | Server returned an HTTP 5xx response                                            | `status` (int, 5xx), `body` (str)                  | 1 |
| `invalid_response`        | Server returned HTTP 2xx but body was unparseable or lacked `prompt_id`         | `status` (int, 2xx)                                | 1 |
| `ws_timeout`              | WebSocket `recv` idle past `--timeout`                                          | `timeout` (int, seconds)                           | 1 |
| `ws_disconnected`         | WebSocket connection dropped mid-execution                                      | —                                                  | 1 |
| `cancelled`               | Run was interrupted — client `SIGINT` (Ctrl-C) or the server's `execution_interrupted` (e.g. `/interrupt`) | —                       | 130 |
| `execution_error`         | A node raised during execution (server emitted `execution_error`)               | `node_id` (str), `class_type` (str), `title` (str), `exception_type` (str), `traceback` (str) | 1 |

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
