# Arm-B bench runner — CLI micro-edit **proxy** (BE-2309 / BE-2302)

> ⚠️ **This is a spike-sanctioned PROXY, not a product feature and not the real thing.**
> Arm B of the BE-2302 A/B benchmark is meant to measure *Kishore's CLI-agent prototype*,
> which is not yet accessible. Until it is, this runner stands in for it by driving Claude
> over the `comfy workflow` micro-edit substrate that already lives in this repo. Every
> emitted NDJSON row is stamped `"proxy": true` so no downstream reader mistakes proxy
> numbers for the real prototype's. See **Swapping in the real prototype** below.

## What it is

`run_arm_b.py` is a minimal agent loop (Claude Opus `claude-opus-4-8`, raw Anthropic
Messages API) that exposes exactly the arm-B toolset BE-2302 defines — "4–5 generic
micro-edit tools operating on a temp JSON workflow on disk":

| Arm-B role       | Tool       | comfy-cli command / layer it wraps                          |
| ---------------- | ---------- | ----------------------------------------------------------- |
| read             | `slots`    | `comfy workflow slots` (`command/workflow.py`)              |
| read (validate)  | `cql`      | the `comfy_cli/cql/` node/type catalog (`Graph`, `engine.py`) |
| edit             | `set_slot` | `comfy workflow set-slot`                                   |
| produce-variants | `vary`     | `comfy workflow vary`                                       |

The loop reads/edits a **temp frontend-format workflow JSON on disk** — the model only ever
sees compact slot manifests, never the full workflow JSON (no full-JSON round-trip). It runs
the same **t1–t4** task prompts as arm A, vendored byte-identical from child 1's
`comfy-inapp-agent/agent-server/bench/tasks.mjs` into [`tasks.json`](./tasks.json).

## Telemetry & output

Per Anthropic API call it captures the `usage` block (`input_tokens`, `output_tokens`,
`cache_read_input_tokens`, `cache_creation_input_tokens`) plus tool-call count and per-call
payload bytes, and folds a task's turns into one NDJSON row per `(task, turn)`. Cost is
recomputed from the token counts with the **same pricing table** as
`comfy-inapp-agent/agent-server/usage.mjs` (`PRICING`): Opus 4.8 = $5/M in, $25/M out, cache
read 0.1×, cache write 1.25× — so arm A and arm B are dollar-comparable.

Rows are written to [`out/arm-b.ndjson`](./out/), **shaped identically to arm A's
`arm-a.ndjson`**, so child 1's `report.mjs` consumes them directly:

```bash
# in comfy-inapp-agent/agent-server/
node bench/report.mjs --arm-b /path/to/comfy-cli/comfy_cli/bench/out/arm-b.ndjson
```

## Offline dry run (no API key, no network) — the acceptance path

```bash
python -m comfy_cli.bench.run_arm_b --dry-run
```

Drives a **stubbed, deterministic** model against the committed
[`fixtures/object_info.json`](./fixtures/) and seed workflows, exercising the full tool
dispatch + usage-parsing + NDJSON pipeline, and writes a well-formed `out/arm-b.ndjson`. This
is what CI and the unit tests (`tests/comfy_cli/bench/test_run_arm_b.py`) run — **no live
calls**. t1/t2/t4 run fully offline against the fixture.

**t3 is the documented ceiling.** Arm B's slots/set-slot/vary toolset can only *edit* an
existing graph, and the committed `object_info` covers only the 6 base SD1.5 node classes
(no LoRA/ControlNet/refiner/upscaler/face-detailer). A ~150-node *build* is therefore out of
reach; the runner records this as a **RESULT** — the t3 row carries `"outcome": "ceiling"`
and a `note` explaining the CQL/object_info coverage limit — rather than pretending to
succeed.

## Live run

```bash
pip install -e '.[bench]'          # adds the `anthropic` SDK
export ANTHROPIC_API_KEY=sk-ant-...
python -m comfy_cli.bench.run_arm_b --live
# optional: --only t2 t4   --model claude-opus-4-8   --out out/arm-b.ndjson
#           --object-info <path>   (point at a live object_info dump for wider node coverage)
```

A live run replaces the stub with a real `anthropic.Anthropic` client; everything else — tool
dispatch, telemetry, NDJSON shape — is identical to the dry run. For wider node coverage than
the committed SD1.5 fixture (e.g. to attempt t3 for real), pass `--object-info` a dump saved
from a live server: `comfy --json ... > object_info.json`, or point `slots`/`set-slot` at a
running ComfyUI.

## Swapping in the real prototype (the documented seam)

The proxy is built so Kishore's real CLI-agent prototype drops into one of two narrow seams,
with the telemetry + NDJSON shape unchanged:

1. **Replace the model client.** The client is injected into `Driver` and only needs a
   `messages.create(...)` returning `.content` / `.stop_reason` / `.usage`. `build_live_client()`
   returns the real Anthropic SDK; point it at the prototype's endpoint instead.
2. **Replace the driver.** For a prototype that speaks its own protocol, implement a driver in
   place of `Driver.run_turn` that keeps calling `ToolDispatcher` (the comfy-cli substrate) and
   emits rows via `build_row(...)`. The pricing, usage-parsing, ceiling handling, and NDJSON
   contract all stay put.

When the real prototype lands, drop the `"proxy": true` stamp in `build_row` (and update this
README) so the rows advertise themselves as the genuine article.

## Keeping the task matrix in sync

`tasks.json` is a **vendored copy** of child 1's `tasks.mjs` prompt strings and MUST stay
byte-identical for the two arms to be comparable. It was generated by evaluating `tasks.mjs`
and dumping its `TASKS` prompts, so re-vendor it (don't hand-edit) if the shared matrix
changes upstream.
