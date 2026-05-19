# `comfy run --json`: Machine-Readable Output (NDJSON)

This document specifies the output contract of `comfy run --json`. The intent
is to give agents and automation a stable, parseable view of a workflow
execution — independent of the human-readable Rich-formatted output that
`comfy run` emits by default.

## Overview

When `--json` is passed, `comfy run` switches into a strict
machine-readable mode:

- **stdout** carries exclusively **NDJSON** (newline-delimited JSON): one
  JSON object per line, each terminated by `\n`. No ANSI, no progress bar,
  no headings. Each line is written and flushed to stdout as soon as the
  underlying event is produced; agents may rely on read-as-emitted timing —
  there is no batching.
- The stream is 7-bit ASCII clean. Non-ASCII characters in string fields
  are emitted as `\uXXXX` JSON escapes (equivalent to Python's
  `json.dumps(..., ensure_ascii=True)`).
- **stderr** is reserved for things the CLI cannot route through the JSON
  contract: framework-level Python errors, uncaught exceptions, library
  warnings. Agents should not parse stderr; they may discard it or
  capture it for diagnostics.
- **Exit code** is `0` when the terminal event is `completed`,
  `queued` (`--no-wait` mode), or `prompt_preview` (`--print-prompt`
  mode); `1` on a `failed` terminal event. Error categorisation is
  carried in the `error.kind` field of the `failed` event, not in the
  exit code (see [Stability](#stability-and-exit-codes)).

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
Numeric count fields (e.g., `node_progress.value` / `max`) are JSON
`number` and may be int or float depending on the underlying node.

This contract targets ComfyUI servers reached via `--host` / `--port`
(typically local). Cloud-specific URL routing (e.g., `/api` prefix,
endpoint renames like `/history` → `/history_v2`) is not exercised in
v1 and may not work without additional flag wiring — agents talking to
Comfy Cloud should expect to use their own client code for now.

## Stream shape

Every line on stdout is a JSON object containing a non-empty string
`event` field acting as a discriminator. Agents must dispatch on this
field. In normal `--json` execution the stream ends with a terminal
event (`completed` or `failed`). In `--no-wait` mode the stream ends
at `queued`, and the agent is responsible for polling
`/history/{prompt_id}` to observe completion. In `--print-prompt`
mode the stream ends at `prompt_preview` (no execution happens, so
`completed`/`failed` would be category-error events about something
that was never started). See [Process-level termination](#process-level-termination)
for the edge case where the CLI is killed before emitting its terminal
event.

### Stream archetypes

The stream takes one of these shapes depending on the workflow format and
outcome:

| Outcome                           | Stream                                                              |
| --------------------------------- | ------------------------------------------------------------------- |
| Success                           | `[converted]? + prompt_preview + queued + [node_*]* + completed`    |
| `--no-wait` queued                | `[converted]? + prompt_preview + queued`                            |
| `--print-prompt`                  | `[converted]? + prompt_preview` (terminal)                          |
| Failure mid-execution             | `[converted]? + prompt_preview + queued + [node_*]* + failed`       |
| Failure during submission         | `[converted]? + prompt_preview + failed`                            |
| Failure pre-flight                | `failed`                                                            |

Where `[node_*]*` is zero or more interleaved `node_cached`,
`node_executing`, `node_progress`, and `node_executed` events. `[X]?`
means X may or may not appear.

### Early-exit rule

`failed` can replace any non-terminal event, terminating the stream
early. The archetypes table above is the canonical view of what shapes
of stream agents will actually see; the early-exit rule is the universal
quantifier behind it.

### Schema version

Every event in v1 streams carries `schema_version: 1`. The field is a
monotonically increasing integer; minor versions are not used. Future
schema versions will emit `schema_version: 2`, etc., on every event.
Agents may read the version from any line. Agents needing
feature-presence detection beyond the integer version should
feature-detect by field presence rather than version comparison.

## Event reference

| Event             | When                                              | Terminal |
| ----------------- | ------------------------------------------------- | -------- |
| `converted`       | UI-format workflow was client-side converted      |          |
| `prompt_preview`  | The API-format workflow graph about to be submitted | (✓ in `--print-prompt`) |
| `queued`          | Server accepted the prompt (HTTP 200 on `/prompt`)| (✓ in `--no-wait`) |
| `node_cached`     | Node hit the execution cache and was skipped      |          |
| `node_executing`  | Node started execution                            |          |
| `node_progress`   | In-flight progress update for the running node    |          |
| `node_executed`   | Node finished and reported its outputs            |          |
| `completed`       | Workflow finished successfully                    | ✓        |
| `failed`          | Workflow could not complete                       | ✓        |

Agents must ignore events whose `event` value they do not recognise —
new event kinds may be added in a backward-compatible manner. Agents
must ignore unknown fields on known events for the same reason.

A handful of fields carry values from a server-defined open set rather
than a fixed enumeration: `class_type`, `category`, `type`, and
`exception_type`. Each is flagged **Open set** at one canonical
description below; the same treatment applies wherever that field
appears across events. Agents must accept and pass through unknown
values without keying behaviour on specific strings.

Every event that names a single node also carries a `title` field —
the human-readable label to show in a per-node UI. The contract for
`title` is the same wherever it appears: **`_meta.title` if present,
else `class_type`, else `node_id`**. Per-event field tables list it
simply as "display label" rather than repeating the chain.

### `converted`

Emitted once if the input workflow was in UI format and was converted to
API format client-side. Not emitted when the input was already in API
format.

```json
{"event": "converted", "schema_version": 1, "node_count": 2}
```

| Field            | Type | Description                                    |
| ---------------- | ---- | ---------------------------------------------- |
| `event`          | str  | `"converted"`                                  |
| `schema_version` | int  | `1`                                            |
| `node_count`     | int  | Number of nodes in the converted graph         |

### `prompt_preview`

Emitted in `--json` mode once the workflow has been successfully
loaded, parsed, and (if UI-format) converted — i.e., in every stream
except the **Failure pre-flight** archetype (a `failed`-only stream
where the CLI bails out before it has a workflow to preview: file
errors, parse errors, server-probe / `/object_info` failures, UI
conversion failures, etc.). Fires immediately after the optional
`converted` event and immediately before `queued`. Carries the API-format workflow graph the CLI is
about to POST to `/prompt` — the same dict that would land in the
request's `prompt` field. Gives agents a complete audit trail of what
was submitted, useful for debugging conversions, logging, and
post-mortem analysis, without needing a separate run.

Under `--print-prompt` this event is also the **terminal** event:
the CLI emits it and exits 0 without queuing. In normal flow it's
informational and execution continues with `queued`.

```json
{"event": "prompt_preview", "schema_version": 1, "prompt": {"1": {"class_type": "EmptyLatentImage", "inputs": {"width": 512, "height": 512, "batch_size": 1}}}}
```

| Field            | Type | Description                                                            |
| ---------------- | ---- | ---------------------------------------------------------------------- |
| `event`          | str  | `"prompt_preview"`                                                     |
| `schema_version` | int  | `1`                                                                    |
| `prompt`         | dict | The API-format workflow graph keyed by node id. Same shape as the `prompt` field POSTed to `/prompt`. Does NOT include `client_id` or `extra_data` (those are runtime fields, not part of the workflow — so any `--api-key` value never appears here). |

For UI-format input under `--print-prompt`, `/object_info` must still
be reachable (the converter consults it). For API-format input under
`--print-prompt`, no server requests are made and the command works
fully offline. In normal (non-`--print-prompt`) flow the server is
always contacted regardless, since execution follows.

### `queued`

Emitted after `POST /prompt` returns 200. Carries the server's prompt
handle, which can be used to correlate against `/history/{prompt_id}`.

```json
{
  "event": "queued",
  "schema_version": 1,
  "prompt_id": "9b1c…",
  "client_id": "fe2a…",
  "validation_warnings": [],
  "nodes": [
    {"node_id": "1", "class_type": "GeminiNanoBanana2", "title": "Nano Banana 2"},
    {"node_id": "2", "class_type": "SaveImage", "title": "Save Image"}
  ]
}
```

| Field                 | Type           | Description                                                  |
| --------------------- | -------------- | ------------------------------------------------------------ |
| `event`               | str            | `"queued"`                                                   |
| `schema_version`      | int            | `1`                                                          |
| `prompt_id`           | str            | Server-assigned prompt UUID                                  |
| `client_id`           | str            | Client-generated UUID (sent with `/prompt`)                  |
| `validation_warnings` | array of dict  | List of per-node validation issues that ComfyUI reported alongside a successful queue (some output chains validated, others didn't). Same record shape as `validation_error.node_errors` (see below). Empty (`[]`) in the common case. |
| `nodes`               | array of dict  | Manifest of every node in the submitted (post-conversion) workflow. Each entry has `node_id` (str), `class_type` (str), and `title` (str — display label, see canonical `title` rule). Lets piped consumers (who don't have the workflow file at hand) render a per-node UI immediately without waiting for `completed`. |

The `validation_warnings` field exists specifically for the case where
ComfyUI's `validate_prompt` returns success because at least one output
chain validated, but other nodes failed validation and will not run.
This field is not a general warnings channel; other warning surfaces
would require separate spec decisions.

### `node_cached`

Emitted up front as a group, before any `node_executing` in the same
run, listing nodes whose outputs were retrieved from the execution
cache. Comes from ComfyUI's `execution_cached` websocket message.

Note: for cached nodes that had prior UI output (e.g., a cached
`SaveImage`), ComfyUI emits both `execution_cached` AND a per-node
`executed` payload with the cached UI dict. The CLI surfaces both: a
single node may produce both `node_cached` and `node_executed` events.
See [completed](#completed) for the resulting semantics of
`cached_node_ids` / `executed_node_ids`.

```json
{"event": "node_cached", "schema_version": 1, "node_id": "1", "class_type": "GeminiNanoBanana2", "title": "Nano Banana 2"}
```

| Field            | Type | Description                                 |
| ---------------- | ---- | ------------------------------------------- |
| `event`          | str  | `"node_cached"`                             |
| `schema_version` | int  | `1`                                         |
| `node_id`        | str  | Node key in the workflow dict               |
| `class_type`     | str  | Node class name. **Open set** — current ComfyUI versions emit names like `KSampler`, `SaveImage`, plus arbitrary custom-node class names; agents must accept and pass through unknown values without keying behaviour on specific strings. |
| `title`          | str  | display label (see canonical `title` rule above) |

### `node_executing`

Emitted when a node starts execution. Subsequent `node_progress` events
(if any) refer to this node until either a different `node_executing`
arrives or the node's `node_executed` event is emitted.

Two consecutive `node_executing` events with different `node_id` values
are normal: the first node finished but didn't surface a result the
server forwarded as `executed` (this happens for intermediate compute
nodes whose outputs aren't published to the client). Agents that track
a "current node" should treat a new `node_executing` as implicitly
closing the previous one.

```json
{"event": "node_executing", "schema_version": 1, "node_id": "2", "class_type": "SaveImage", "title": "Save Image"}
```

Fields are identical to `node_cached`.

### `node_progress`

Per-step progress for samplers, video encoders, and any node that calls
`ProgressBar.update_absolute(...)`. Shares the `node_id` of the most
recently-emitted `node_executing` event.

```json
{"event": "node_progress", "schema_version": 1, "node_id": "1", "class_type": "GeminiNanoBanana2", "title": "Nano Banana 2", "value": 14, "max": 30}
```

| Field            | Type   | Description                                                 |
| ---------------- | ------ | ----------------------------------------------------------- |
| `event`          | str    | `"node_progress"`                                           |
| `schema_version` | int    | `1`                                                         |
| `node_id`        | str    | Node currently running                                      |
| `class_type`     | str    | Node class name (duplicated from the workflow so stateless consumers don't need to track the prior `node_executing`) |
| `title`          | str    | display label (see canonical `title` rule above)                 |
| `value`          | number | Current progress. Typically int (step count); some custom nodes emit float (fractional progress). Defaults to `0` when the server omits the field. |
| `max`            | number | Total progress; same caveat as `value`. Defaults to `0` when the server omits the field. |

Some custom nodes may emit `value > max` near the end of execution.
Agents rendering a progress bar should clamp `value` to `max`.

### `node_executed`

Emitted when the server reports node completion via its `executed`
websocket message. **Not guaranteed for every executed node** —
intermediate compute nodes that don't surface output to the client may
finish without emitting this event. Output nodes (like `SaveImage`) and
some custom partner nodes that publish previews reliably emit it. A
cached output-bearing node also emits `node_executed` (in addition to
`node_cached`).

```json
{
  "event": "node_executed",
  "schema_version": 1,
  "node_id": "2",
  "class_type": "SaveImage",
  "title": "Save Image",
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
  ]
}
```

| Field            | Type             | Description                                 |
| ---------------- | ---------------- | ------------------------------------------- |
| `event`          | str              | `"node_executed"`                           |
| `schema_version` | int              | `1`                                         |
| `node_id`        | str              | Node key                                    |
| `class_type`     | str              | Node class name                             |
| `title`          | str              | display label (see canonical `title` rule above) |
| `outputs`        | array of Output  | File-like outputs (empty if none)           |

`outputs` is populated by iterating each key in ComfyUI's
`executed.output` dict and emitting any item that matches the
file-record shape (a dict containing a `filename` key). Items that are
not file-record-shaped (strings, booleans, mixed lists from nodes that
publish non-file data like text predictions or animation flags) are
silently skipped. See [Output object](#output-object) for the entry
shape.

### `completed`

Terminal event on success. Carries identifiers, timing, the aggregated
output list, and node-execution metadata.

```json
{
  "event": "completed",
  "schema_version": 1,
  "prompt_id": "9b1c…",
  "client_id": "fe2a…",
  "elapsed_seconds": 8.342,
  "outputs": [],
  "cached_node_ids": ["1"],
  "executed_node_ids": ["2"]
}
```

| Field               | Type            | Description                                                  |
| ------------------- | --------------- | ------------------------------------------------------------ |
| `event`             | str             | `"completed"`                                                |
| `schema_version`    | int             | `1`                                                          |
| `prompt_id`         | str             | Server-assigned prompt UUID                                  |
| `client_id`         | str             | Client-generated UUID                                        |
| `elapsed_seconds`   | float           | Wall-clock duration from start of `comfy run` (same clock as `failed.elapsed_seconds`) |
| `outputs`           | array of Output | All file-like outputs across all nodes (empty if none)       |
| `cached_node_ids`   | array of str    | Node IDs the server reported as cached (via `execution_cached`) |
| `executed_node_ids` | array of str    | Node IDs the executor *ran* — the union of every `node_id` that appeared in a `node_executing` or `node_executed` event. Named for what the executor did (run a node), broader than the leaf-only `node_executed` event: includes intermediate compute nodes (CheckpointLoaderSimple, KSampler, etc.) that don't surface output to the client. |

`cached_node_ids` and `executed_node_ids` are independent signals about
what the server reported. **They may overlap**: a cached output-bearing
node emits both `execution_cached` and `executed`, so it appears in
both lists. Agents wanting "ran fresh, not from cache" should compute
`set(executed_node_ids) - set(cached_node_ids)`.

### `failed`

Terminal event on any failure. The `error.kind` discriminator is the
documented stable enum (see [Error object](#error-object) and
[Error kinds](#error-kinds)).

```json
{
  "event": "failed",
  "schema_version": 1,
  "prompt_id": "9b1c…",
  "client_id": "fe2a…",
  "elapsed_seconds": 1.23,
  "error": {
    "kind": "execution_error",
    "message": "API key invalid",
    "node_id": "1",
    "class_type": "GeminiNanoBanana2",
    "title": "Nano Banana 2",
    "exception_type": "RuntimeError",
    "traceback": "  File \"/path/to/node.py\", line 42, in execute\n    raise RuntimeError(\"API key invalid\")\n"
  }
}
```

| Field             | Type        | Description                                                                |
| ----------------- | ----------- | -------------------------------------------------------------------------- |
| `event`           | str         | `"failed"`                                                                 |
| `schema_version`  | int         | `1`                                                                        |
| `prompt_id`       | str \| null | `null` when failure occurred before `/prompt` was accepted                 |
| `client_id`       | str \| null | `null` when failure occurred before `WorkflowExecution` was constructed    |
| `elapsed_seconds` | float       | Wall-clock duration from start of `comfy run`                              |
| `error`           | Error       | See [Error object](#error-object)                                          |

If `prompt_id` is non-null, `client_id` is also non-null (a `prompt_id`
cannot be assigned without a `client_id`).

## Output object

```json
{
  "category": "images",
  "node_id": "2",
  "class_type": "SaveImage",
  "title": "Save Image",
  "filename": "banana_test_00001_.png",
  "subfolder": "",
  "type": "output",
  "url": "http://127.0.0.1:8188/view?filename=..."
}
```

| Field        | Type        | Description                                                                                              |
| ------------ | ----------- | -------------------------------------------------------------------------------------------------------- |
| `category`   | str         | Output category as keyed by ComfyUI's `executed.output` dict. **Open set.** Current ComfyUI versions emit values like `images`, `audio`, `3d`, `latents`; agents must accept and pass through unknown values. |
| `node_id`    | str         | Node that produced the output                                                                            |
| `class_type` | str         | Node class name                                                                                          |
| `title`      | str         | display label (see canonical `title` rule above)                                                              |
| `filename`   | str         | Raw filename as reported by the server                                                                   |
| `subfolder`  | str         | Subfolder within the output folder's root. Defaults to `""` when the server omits or empties the field.   |
| `type`       | str         | ComfyUI output folder discriminator. **Open set.** Current ComfyUI versions emit `output`, `temp`, `input`; agents must accept and pass through unknown values. Defaults to `"output"` when the server omits or empties the field. |
| `url`        | str         | `http(s)://<host>:<port>/view?...` URL — always present, fetch this to get the bytes |

### Fetching output bytes

The `url` field is the only contractual way to retrieve an output's
bytes. It points at ComfyUI's `/view` endpoint and works whether the
agent is on the same machine as ComfyUI, on a different host, or
talking to Cloud. For the local case, a loopback HTTP fetch from a
ComfyUI on the same box is cheap — the agent's HTTP client reads
through the kernel loopback in the same way it'd read a local file.

An earlier draft of v1 also emitted a `local_path` field for the
same-machine case; it was removed because resolving ComfyUI's actual
output directory reliably (across manual launches, alternate install
paths, multi-install machines, bind-mounted volumes) wasn't feasible.
Agents should rely on `url` exclusively.

## Error object

Every `failed` event carries an `error` object with these universal
fields, plus per-kind extras documented in [Error kinds](#error-kinds).

| Field     | Type | Description                                                                                  |
| --------- | ---- | -------------------------------------------------------------------------------------------- |
| `kind`    | str  | Discriminator; one of the values in [Error kinds](#error-kinds)                              |
| `message` | str  | Human-readable summary. For display only — agents should dispatch on `kind`, not on `message`|

The set of per-kind extra fields is the documented minimum. New
optional extra fields may be added in non-breaking releases; existing
fields will not be removed or renamed without a `schema_version` bump.

## Error kinds

Per-kind extra fields. Universal fields (`kind`, `message`) are
documented in [Error object](#error-object).

Each `error.kind` is a stable string. New kinds may be added in
backward-compatible releases; existing kinds will not be renamed or
removed without a schema version bump.

| `kind`                    | Triggered when                                                                                | Extra fields                                       |
| ------------------------- | --------------------------------------------------------------------------------------------- | -------------------------------------------------- |
| `workflow_not_found`      | `--workflow` path does not exist                                                              | —                                                  |
| `workflow_invalid_json`   | Workflow file is not valid JSON                                                               | —                                                  |
| `workflow_read_error`     | Workflow file exists but isn't readable as text (`OSError`, `UnicodeDecodeError`)             | —                                                  |
| `workflow_format_invalid` | File parses but is neither UI nor API format                                                  | —                                                  |
| `workflow_empty`          | Workflow has no executable nodes (UI conversion produced `{}`, or API workflow is `{}`)       | —                                                  |
| `conversion_error`        | UI→API converter raised `WorkflowConversionError`                                             | —                                                  |
| `conversion_crash`        | UI→API converter raised an unexpected exception                                               | `exception_type` (str)                             |
| `object_info_unavailable` | `/object_info` returned an HTTP error, or an HTTP 200 with an unparseable body                | `status_code` (int), `body` (str)                  |
| `connection_error`        | ComfyUI server unreachable: `URLError`, `TimeoutError`, or other `OSError` while contacting it (including on `/object_info`) | —                                |
| `validation_error`        | Server returned HTTP 400 with `node_errors`                                                   | `node_errors` (array of dict; see [shape](#validation_errornode_errors-shape)) |
| `client_error`            | Server returned an HTTP 4xx response (not validation)                                         | `status_code` (int, 4xx), `body` (str)             |
| `server_error`            | Server returned an HTTP 5xx response                                                          | `status_code` (int, 5xx), `body` (str)             |
| `invalid_response`        | Server returned HTTP 2xx but body was unparseable or lacked `prompt_id`                       | `status_code` (int, 2xx), `body` (str)             |
| `timeout`                 | WebSocket `recv` timed out                                                                    | `timeout_seconds` (float)                          |
| `connection_lost`         | WebSocket connection dropped mid-execution                                                    | —                                                  |
| `execution_interrupted`   | Workflow was interrupted — either by the server (`execution_interrupted` WS, e.g., via `/interrupt`) or by the client process receiving `SIGINT` (Ctrl-C) | —              |
| `execution_error`         | A node raised during execution (server emitted `execution_error`)                             | `node_id` (str), `class_type` (str), `title` (str — display label, see canonical `title` rule), `exception_type` (str), `traceback` (str) |

### `exception_type` field

`exception_type` is provided for diagnostic and observability purposes
(e.g., metrics bucketing). **Open set** — the format is whatever
ComfyUI sends, typically the bare class name for builtins
(`RuntimeError`, `ValueError`) and a dotted module path for non-
builtins (`comfy.model_management.InterruptProcessingException`). May
be `""` when the server omits it. Agents should not key retry or
routing logic on `exception_type`; use `error.kind` for coarse
dispatch and `error.message` for human display.

### `traceback` field

`traceback` is a single multi-line string carrying the formatted stack
frames as reported by ComfyUI (joined from the server's
`traceback.format_tb()` output). It does NOT include the
`"Traceback (most recent call last):"` header or the final
`"ExceptionType: message"` summary line — agents reconstructing a
Python-style display can do so themselves from `exception_type`,
`error.message`, and `traceback`. May be empty (`""`) when the server's
formatted stack is empty.

After `json.loads()`, the `traceback` string contains real newline
characters (the JSON wire-format `\n` escapes are decoded).

### `validation_error.node_errors` shape

The same shape is used for `queued.validation_warnings`. The value is
an array of self-contained records, one per affected node. Each record
carries `node_id` (str — same identifier as appears in `node_*` events)
plus the per-node fields ComfyUI emits. Example shape:

```json
"node_errors": [
  {
    "node_id": "1",
    "errors": [
      {
        "type": "value_not_in_list",
        "message": "Value not in list",
        "details": "resolution: '5K' not in ['1K','2K','4K']",
        "extra_info": {
          "input_name": "resolution",
          "received_value": "5K"
        }
      }
    ],
    "dependent_outputs": ["2"],
    "class_type": "GeminiNanoBanana2"
  }
]
```

The inner per-node fields are defined by ComfyUI's `validate_prompt()`
in [`server.py`](https://github.com/comfyanonymous/ComfyUI/blob/master/server.py)
and may evolve with ComfyUI versions. **Agents should ignore unknown
fields.** The CLI guarantees only:
- the outer value is an array of dicts, each carrying a `node_id` (str), and
- under current ComfyUI versions, each record additionally carries
  `errors`, `dependent_outputs`, and `class_type`.

The record order matches ComfyUI's response order and is not guaranteed
to be sorted; consumers that need a specific order should sort
themselves.

## Process-level termination

The CLI may be terminated by the operating system or a parent process
(SIGKILL, SIGTERM, SIGINT, OOM-kill, segmentation fault). In these
cases, no terminal event is emitted and the stream may be truncated.

Agents should treat the run as failed when **both**:
- the process exit code is non-zero, and
- the last line on stdout is not one of the documented terminal events
  (`completed`, `failed`, `queued` under `--no-wait`, or `prompt_preview`
  under `--print-prompt`), or stdout is empty.

Stderr may contain a Python traceback in these cases.

## Examples

Class type names in these examples (`SaveImage`, `GeminiNanoBanana2`,
etc.) are illustrative — they reflect specific ComfyUI/partner nodes.
Agents should not hardcode behavior on specific `class_type` strings;
the contract guarantees the *shape* of these fields, not their content.

Every line, including the terminal event (`completed` / `failed` /
`queued` under `--no-wait` / `prompt_preview` under `--print-prompt`),
ends with `\n`. Agents using line iteration (`for line in stdout`) are fine;
agents using `splitlines()` or `split("\n")` should filter empty
trailing entries.

### Successful run (UI-format input)

```json
{"event":"converted","schema_version":1,"node_count":2}
{"event":"prompt_preview","schema_version":1,"prompt":{"1":{"class_type":"GeminiNanoBanana2","inputs":{"prompt":"a banana","width":2048,"height":2048},"_meta":{"title":"Nano Banana 2"}},"2":{"class_type":"SaveImage","inputs":{"filename_prefix":"banana_test","images":["1",0]},"_meta":{"title":"Save Image"}}}}
{"event":"queued","schema_version":1,"prompt_id":"9b1c…","client_id":"fe2a…","validation_warnings":[],"nodes":[{"node_id":"1","class_type":"GeminiNanoBanana2","title":"Nano Banana 2"},{"node_id":"2","class_type":"SaveImage","title":"Save Image"}]}
{"event":"node_executing","schema_version":1,"node_id":"1","class_type":"GeminiNanoBanana2","title":"Nano Banana 2"}
{"event":"node_progress","schema_version":1,"node_id":"1","class_type":"GeminiNanoBanana2","title":"Nano Banana 2","value":1,"max":4}
{"event":"node_progress","schema_version":1,"node_id":"1","class_type":"GeminiNanoBanana2","title":"Nano Banana 2","value":4,"max":4}
{"event":"node_executing","schema_version":1,"node_id":"2","class_type":"SaveImage","title":"Save Image"}
{"event":"node_executed","schema_version":1,"node_id":"2","class_type":"SaveImage","title":"Save Image","outputs":[{"category":"images","node_id":"2","class_type":"SaveImage","title":"Save Image","filename":"banana_test_00001_.png","subfolder":"","type":"output","url":"http://127.0.0.1:8188/view?filename=banana_test_00001_.png&subfolder=&type=output"}]}
{"event":"completed","schema_version":1,"prompt_id":"9b1c…","client_id":"fe2a…","elapsed_seconds":8.342,"outputs":[{"category":"images","node_id":"2","class_type":"SaveImage","title":"Save Image","filename":"banana_test_00001_.png","subfolder":"","type":"output","url":"http://127.0.0.1:8188/view?filename=banana_test_00001_.png&subfolder=&type=output"}],"cached_node_ids":[],"executed_node_ids":["1","2"]}
```

Exit code: `0`.

Note: node 1 (`GeminiNanoBanana2`) does not emit a `node_executed`
event in this example — it's an intermediate compute node whose result
is forwarded via tensors rather than surfaced as a file output, so the
server doesn't send an `executed` ws message for it.

### `--no-wait` (API-format input)

```json
{"event":"prompt_preview","schema_version":1,"prompt":{"1":{"class_type":"GeminiNanoBanana2","inputs":{"prompt":"a banana","width":2048,"height":2048},"_meta":{"title":"Nano Banana 2"}},"2":{"class_type":"SaveImage","inputs":{"filename_prefix":"banana_test","images":["1",0]},"_meta":{"title":"Save Image"}}}}
{"event":"queued","schema_version":1,"prompt_id":"9b1c…","client_id":"fe2a…","validation_warnings":[],"nodes":[{"node_id":"1","class_type":"GeminiNanoBanana2","title":"Nano Banana 2"},{"node_id":"2","class_type":"SaveImage","title":"Save Image"}]}
```

Exit code: `0`. The agent is responsible for polling
`/history/{prompt_id}` to observe completion.

### Failure: workflow file missing

```json
{"event":"failed","schema_version":1,"prompt_id":null,"client_id":null,"elapsed_seconds":0.001,"error":{"kind":"workflow_not_found","message":"Workflow file not found: /tmp/missing.json"}}
```

Exit code: `1`.

### Failure: server returned validation errors

```json
{"event":"converted","schema_version":1,"node_count":2}
{"event":"prompt_preview","schema_version":1,"prompt":{"1":{"class_type":"GeminiNanoBanana2","inputs":{"prompt":"a banana","resolution":"5K"},"_meta":{"title":"Nano Banana 2"}},"2":{"class_type":"SaveImage","inputs":{"filename_prefix":"banana_test","images":["1",0]},"_meta":{"title":"Save Image"}}}}
{"event":"failed","schema_version":1,"prompt_id":null,"client_id":"fe2a…","elapsed_seconds":0.45,"error":{"kind":"validation_error","message":"Value not in list","node_errors":[{"node_id":"1","errors":[{"type":"value_not_in_list","message":"Value not in list","details":"resolution: '5K' not in ['1K','2K','4K']","extra_info":{"input_name":"resolution","received_value":"5K"}}],"dependent_outputs":["2"],"class_type":"GeminiNanoBanana2"}]}}
```

Exit code: `1`.

### Failure: node raised during execution

```json
{"event":"prompt_preview","schema_version":1,"prompt":{"1":{"class_type":"GeminiNanoBanana2","inputs":{"prompt":"a banana","width":2048,"height":2048},"_meta":{"title":"Nano Banana 2"}},"2":{"class_type":"SaveImage","inputs":{"filename_prefix":"banana_test","images":["1",0]},"_meta":{"title":"Save Image"}}}}
{"event":"queued","schema_version":1,"prompt_id":"9b1c…","client_id":"fe2a…","validation_warnings":[],"nodes":[{"node_id":"1","class_type":"GeminiNanoBanana2","title":"Nano Banana 2"},{"node_id":"2","class_type":"SaveImage","title":"Save Image"}]}
{"event":"node_executing","schema_version":1,"node_id":"1","class_type":"GeminiNanoBanana2","title":"Nano Banana 2"}
{"event":"failed","schema_version":1,"prompt_id":"9b1c…","client_id":"fe2a…","elapsed_seconds":2.1,"error":{"kind":"execution_error","message":"API key invalid","node_id":"1","class_type":"GeminiNanoBanana2","title":"Nano Banana 2","exception_type":"RuntimeError","traceback":"  File \"/path/to/node.py\", line 42, in execute\n    raise RuntimeError(\"API key invalid\")\n"}}
```

Exit code: `1`.

### Failure: websocket timeout

```json
{"event":"prompt_preview","schema_version":1,"prompt":{"1":{"class_type":"GeminiNanoBanana2","inputs":{"prompt":"a banana","width":2048,"height":2048},"_meta":{"title":"Nano Banana 2"}},"2":{"class_type":"SaveImage","inputs":{"filename_prefix":"banana_test","images":["1",0]},"_meta":{"title":"Save Image"}}}}
{"event":"queued","schema_version":1,"prompt_id":"9b1c…","client_id":"fe2a…","validation_warnings":[],"nodes":[{"node_id":"1","class_type":"GeminiNanoBanana2","title":"Nano Banana 2"},{"node_id":"2","class_type":"SaveImage","title":"Save Image"}]}
{"event":"node_executing","schema_version":1,"node_id":"1","class_type":"GeminiNanoBanana2","title":"Nano Banana 2"}
{"event":"failed","schema_version":1,"prompt_id":"9b1c…","client_id":"fe2a…","elapsed_seconds":30.0,"error":{"kind":"timeout","message":"WebSocket timed out after 30s waiting for server response","timeout_seconds":30.0}}
```

Exit code: `1`.

### Failure: workflow interrupted

```json
{"event":"prompt_preview","schema_version":1,"prompt":{"1":{"class_type":"GeminiNanoBanana2","inputs":{"prompt":"a banana","width":2048,"height":2048},"_meta":{"title":"Nano Banana 2"}},"2":{"class_type":"SaveImage","inputs":{"filename_prefix":"banana_test","images":["1",0]},"_meta":{"title":"Save Image"}}}}
{"event":"queued","schema_version":1,"prompt_id":"9b1c…","client_id":"fe2a…","validation_warnings":[],"nodes":[{"node_id":"1","class_type":"GeminiNanoBanana2","title":"Nano Banana 2"},{"node_id":"2","class_type":"SaveImage","title":"Save Image"}]}
{"event":"node_executing","schema_version":1,"node_id":"1","class_type":"GeminiNanoBanana2","title":"Nano Banana 2"}
{"event":"failed","schema_version":1,"prompt_id":"9b1c…","client_id":"fe2a…","elapsed_seconds":3.2,"error":{"kind":"execution_interrupted","message":"Workflow execution was interrupted"}}
```

Exit code: `1`.

## Stability and exit codes

### What is stable

For the v1 contract documented here:
- The set of event names listed above and the field names within them.
- The set of `error.kind` values listed above and the per-kind extra
  fields documented for each.
- The exit code mapping: `0` when the terminal event is `completed`,
  `queued` (under `--no-wait`), or `prompt_preview` (under
  `--print-prompt`); `1` on `failed`.
- The stdout/stderr separation: stdout carries only NDJSON (no ANSI,
  no human-readable progress bar, no headings); stderr is reserved
  for framework-level Python errors, uncaught exceptions, and library
  warnings — agents should not parse it.
- The 7-bit ASCII encoding of stdout (non-ASCII characters in string
  fields are emitted as `\uXXXX` JSON escapes, equivalent to
  `json.dumps(..., ensure_ascii=True)`).
- The `schema_version: 1` field on every event of v1 streams.

### What may change in a non-breaking way

- New event types being added (agents must ignore unknown `event` values).
- New `error.kind` values being added (agents must default-handle unknown
  kinds).
- New optional fields being added to existing events (agents must ignore
  unknown fields).

New events that would alter the meaning of existing events when
ignored (for example, a per-node skip event whose absence would make
`executed_node_ids` incomplete) require a `schema_version` bump rather
than being treated as an additive change.

### Why exit codes are not granular

The 0/1 mapping (defined in "What is stable" above) intentionally
trades resolution for stability. `error.kind` is the expressive,
extensible discriminator — agents dispatch on it; the exit code is
just a coarse "did we succeed?" signal. Granular exit codes can be
introduced later for non-`--json` callers in a separate,
evidence-driven change without breaking the JSON contract.
