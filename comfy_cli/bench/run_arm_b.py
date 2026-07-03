"""Arm-B runner for the BE-2302 A/B micro-edit benchmark — a **SANCTIONED PROXY**.

Arm B of BE-2302 is defined as "4-5 generic micro-edit tools operating on a temp
JSON workflow on disk." This module is a minimal agent loop (Claude Opus
``claude-opus-4-8`` over the raw Anthropic Messages API) that exposes exactly the
``comfy workflow`` micro-edit substrate already in this repo as tools:

    read           → ``slots``   (comfy workflow slots) + ``cql`` (node/type lookup)
    edit           → ``set_slot`` (comfy workflow set-slot)
    produce-variant→ ``vary``     (comfy workflow vary)

It runs the SAME t1–t4 task prompts as arm A (vendored byte-identical from child 1's
``tasks.mjs`` into ``bench/tasks.json``) and emits per-(task, turn) NDJSON rows shaped
IDENTICALLY to arm A's ``arm-a.ndjson`` so child 1's ``report.mjs`` consumes both
directly. Cost is recomputed from the Anthropic ``usage`` block with the SAME pricing
table as ``comfy-inapp-agent/agent-server/usage.mjs`` (Opus 4.8: $5/M in, $25/M out,
cache read 0.1×, cache write 1.25×) so the two arms are dollar-comparable.

────────────────────────────────────────────────────────────────────────────────
PROXY CAVEAT — this is NOT Kishore's real CLI-agent prototype.
────────────────────────────────────────────────────────────────────────────────
BE-2302 sanctions this as a *proxy* for a not-yet-accessible CLI agent prototype.
Every emitted row is stamped ``"proxy": true`` and ``"arm": "B"`` so no downstream
reader mistakes it for the real thing. The swap-in seam is deliberately narrow:

    * The MODEL driver is ``Driver`` (see ``run_task``). To swap in the real
      prototype, implement a driver that speaks the prototype's protocol but keeps
      calling ``ToolDispatcher`` (the comfy-cli substrate) and emitting rows through
      ``build_row`` — the telemetry + NDJSON shape then stay identical for free.
    * The MODEL CLIENT is injected (``client`` arg). ``StubClient`` (offline,
      deterministic) drives ``--dry-run`` and the unit tests; ``build_live_client``
      returns a real ``anthropic.Anthropic`` for ``--live``. Point the client at the
      prototype's endpoint, or replace ``Driver`` wholesale, and nothing else moves.

See ``bench/README.md`` for the live-run recipe and the full swap-in contract.
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Pricing — mirror of comfy-inapp-agent/agent-server/usage.mjs PRICING so the two
# arms price a turn identically. Values are USD per million tokens.
# ---------------------------------------------------------------------------

PRICING: list[tuple[str, float, float]] = [
    # (id-prefix, input $/MTok, output $/MTok)
    ("claude-fable-5", 10.0, 50.0),
    ("claude-opus-4-8", 5.0, 25.0),
    ("claude-opus-4-7", 5.0, 25.0),
    ("claude-opus-4-6", 5.0, 25.0),
    ("claude-sonnet-4-6", 3.0, 15.0),
    ("claude-haiku-4-5", 1.0, 5.0),
]
CACHE_READ_MULT = 0.1
CACHE_WRITE_MULT = 1.25

DEFAULT_MODEL = "claude-opus-4-8"

# Upper bound on variants a single `vary` tool call may produce — guards against an
# unbounded model-supplied list length exhausting memory/disk.
MAX_VARIANTS = 64

# Token fields the runner captures from the Anthropic ``usage`` block, emitted
# verbatim so report.mjs can recompute cost from them.
TOKEN_FIELDS = (
    "input_tokens",
    "output_tokens",
    "cache_read_input_tokens",
    "cache_creation_input_tokens",
)


def _round6(v: float) -> float:
    return round(v * 1e6) / 1e6


def resolve_pricing(model: str | None) -> tuple[str, float, float] | None:
    """Longest-prefix match of a model id against the pricing table; None when unknown."""
    if not isinstance(model, str) or not model:
        return None
    best: tuple[str, float, float] | None = None
    for entry in PRICING:
        prefix = entry[0]
        if model.startswith(prefix) and (best is None or len(prefix) > len(best[0])):
            best = entry
    return best


def estimate_cost_usd(model: str | None, usage: dict[str, Any]) -> float | None:
    """USD cost of one turn's tokens at list pricing. None when the model is unknown.

    Mirrors ``usage.mjs`` ``estimateCostUsd`` exactly (cache read 0.1×, write 1.25×).
    """
    p = resolve_pricing(model)
    if p is None:
        return None
    _, in_rate, out_rate = p
    n = lambda v: v if isinstance(v, int | float) and not isinstance(v, bool) else 0  # noqa: E731
    return _round6(
        (
            n(usage.get("input_tokens")) * in_rate
            + n(usage.get("output_tokens")) * out_rate
            + n(usage.get("cache_read_input_tokens")) * in_rate * CACHE_READ_MULT
            + n(usage.get("cache_creation_input_tokens")) * in_rate * CACHE_WRITE_MULT
        )
        / 1e6
    )


def parse_usage(usage: Any) -> dict[str, int]:
    """Normalize an Anthropic ``usage`` block (SDK object or dict) → the 4 token counts.

    Missing / non-numeric fields default to 0 so a partial usage block never crashes
    the loop. This is the unit-tested seam between the API response and the NDJSON row.
    """

    def get(name: str) -> int:
        if usage is None:
            return 0
        v = usage.get(name) if isinstance(usage, dict) else getattr(usage, name, None)
        if isinstance(v, bool) or not isinstance(v, int | float):
            return 0
        return int(v)

    return {f: get(f) for f in TOKEN_FIELDS}


# ---------------------------------------------------------------------------
# Tool schemas — the comfy-cli micro-edit substrate exposed to the model.
# ---------------------------------------------------------------------------

TOOLS: list[dict[str, Any]] = [
    {
        "name": "slots",
        "description": (
            "READ. List the tweakable slots of the workflow currently on disk "
            "(`comfy workflow slots`). Each slot is {address, type, current_value}; "
            "address is `<node_id>.<input>`. Call this first to discover what you can edit."
        ),
        "input_schema": {"type": "object", "properties": {}, "additionalProperties": False},
    },
    {
        "name": "cql",
        "description": (
            "READ. Validate a node type / inspect the object_info catalog (the CQL layer). "
            "Pass `node_type` to fetch that node class's input/output schema, or omit it to "
            "list every node class available offline. Use this to check a node exists before editing."
        ),
        "input_schema": {
            "type": "object",
            "properties": {"node_type": {"type": "string", "description": "Node class to look up, e.g. 'KSampler'."}},
            "additionalProperties": False,
        },
    },
    {
        "name": "set_slot",
        "description": (
            "EDIT. Apply one or more slot overrides to the workflow in place "
            "(`comfy workflow set-slot`). `overrides` maps slot address → new value, "
            "e.g. {'6.text': 'a cat', '3.steps': 25}."
        ),
        "input_schema": {
            "type": "object",
            "properties": {"overrides": {"type": "object", "description": "address → value map."}},
            "required": ["overrides"],
            "additionalProperties": False,
        },
    },
    {
        "name": "vary",
        "description": (
            "PRODUCE-VARIANTS. Expand the workflow into N variants from a per-slot value list "
            "(`comfy workflow vary`). `slots` maps address → list of values; all lists must be the "
            "same length N. Returns the variant count (variants are written to a temp dir, not to the model)."
        ),
        "input_schema": {
            "type": "object",
            "properties": {"slots": {"type": "object", "description": "address → list-of-values map."}},
            "required": ["slots"],
            "additionalProperties": False,
        },
    },
]

SYSTEM_PROMPT = (
    "You are a proxy for a ComfyUI CLI micro-edit agent. You edit a frontend-format workflow JSON "
    "that lives on disk via four tools (slots, cql, set_slot, vary) — you NEVER receive or emit the "
    "full workflow JSON, only compact slot manifests. Discover slots with `slots`, validate node "
    "types with `cql`, make edits with `set_slot`, and produce variants with `vary`. Your toolset can "
    "only EDIT an existing graph and its known node catalog; it cannot add arbitrary new node types. "
    "If a task needs node classes that `cql` reports are unavailable, say so plainly and stop."
)


# ---------------------------------------------------------------------------
# Tool dispatch — every call hits the real comfy-cli CQL engine against the
# committed object_info fixture + the temp workflow on disk.
# ---------------------------------------------------------------------------


class ToolDispatcher:
    """Maps a model tool call onto the actual comfy-cli micro-edit primitives.

    The graph is built once from the offline ``object_info`` fixture; the workflow
    is (re)read from ``workflow_path`` on every call so a ``set_slot`` write is
    visible to the next ``slots`` read — a genuine on-disk round-trip, no full-JSON
    round-trip to the model.
    """

    def __init__(self, object_info_path: str | Path, workflow_path: str | Path, variant_dir: str | Path):
        from comfy_cli.cql.engine import Graph

        self.graph = Graph.load(input_path=str(object_info_path))
        self.workflow_path = Path(workflow_path)
        self.variant_dir = Path(variant_dir)
        self._vary_calls = 0  # namespaces each vary invocation's output files

    def _load_workflow(self) -> dict[str, Any]:
        return json.loads(self.workflow_path.read_text(encoding="utf-8"))

    def dispatch(self, name: str, tool_input: dict[str, Any]) -> tuple[dict[str, Any], bool]:
        """Run one tool. Returns (result_dict, is_error). Never raises."""
        try:
            handler = {
                "slots": self._slots,
                "cql": self._cql,
                "set_slot": self._set_slot,
                "vary": self._vary,
            }.get(name)
            if handler is None:
                return {"error": f"unknown tool {name!r}"}, True
            return handler(tool_input or {})
        except Exception:  # dispatch must be total — never raise into the loop
            # Log full detail locally, but return only a generic message: the result
            # is sent to the external Anthropic API on a live run and the raw exception
            # string can embed absolute filesystem paths / other local state.
            logger.exception("tool %r raised while dispatching", name)
            return {"error": f"internal error while running tool {name!r}"}, True

    # -- read tools --

    def _slots(self, _inp: dict[str, Any]) -> tuple[dict[str, Any], bool]:
        wf = self._load_workflow()
        schema = self.graph.get_template_schema(self.workflow_path.stem, wf)
        slots = [
            {"address": s.get("address"), "type": s.get("type"), "current_value": s.get("current_value")}
            for s in schema.get("slots") or []
        ]
        return {"count": len(slots), "slots": slots}, False

    def _cql(self, inp: dict[str, Any]) -> tuple[dict[str, Any], bool]:
        node_type = inp.get("node_type")
        if not node_type:
            return {"available_node_types": [m.id for m in self.graph.all_nodes()]}, False
        m = self.graph.node(node_type)
        if m is None:
            import difflib

            names = [n.id for n in self.graph.all_nodes()]
            close = difflib.get_close_matches(node_type, names, n=5, cutoff=0.4)
            return {"node_type": node_type, "found": False, "suggestions": close}, False
        return {"node_type": node_type, "found": True, "schema": self.graph.morphism_to_dict(m)}, False

    # -- edit tool --

    def _set_slot(self, inp: dict[str, Any]) -> tuple[dict[str, Any], bool]:
        overrides = inp.get("overrides")
        if not isinstance(overrides, dict) or not overrides:
            return {"error": "set_slot requires a non-empty `overrides` object (address → value)."}, True
        wf = self._load_workflow()
        try:
            new_wf, warnings = self.graph.apply_slots(wf, overrides)
        except ValueError as e:
            return {"error": str(e), "hint": "run `slots` to see valid addresses + types"}, True
        self.workflow_path.write_text(json.dumps(new_wf, indent=2), encoding="utf-8")
        return {"applied": list(overrides.keys()), "warnings": warnings}, False

    # -- produce-variants tool --

    def _vary(self, inp: dict[str, Any]) -> tuple[dict[str, Any], bool]:
        by_addr = inp.get("slots")
        if not isinstance(by_addr, dict) or not by_addr:
            return {"error": "vary requires a non-empty `slots` object (address → list-of-values)."}, True
        lengths = {a: len(v) for a, v in by_addr.items() if isinstance(v, list)}
        if len(lengths) != len(by_addr):
            return {"error": "every `slots` value must be a JSON array."}, True
        if len(set(lengths.values())) != 1:
            return {"error": f"all `slots` lists must be the same length; got {lengths}."}, True
        n = next(iter(lengths.values()))
        if n < 1:
            return {"error": "vary requires at least one value per slot."}, True
        if n > MAX_VARIANTS:
            # A model-supplied list length is unbounded; deep-copying the workflow n
            # times would exhaust memory/disk/inodes. Cap it so a runaway tool call fails
            # loudly instead of hanging.
            return {"error": f"vary is capped at {MAX_VARIANTS} variants per call; requested {n}."}, True
        variations = [{a: v[i] for a, v in by_addr.items()} for i in range(n)]
        wf = self._load_workflow()
        try:
            workflows, warnings = self.graph.expand_variations(wf, variations)
        except ValueError as e:
            return {"error": str(e)}, True
        # Namespace each vary call's files so a second call in the same task can't
        # silently overwrite the first call's variants (both would restart at _000).
        call_idx = self._vary_calls
        self._vary_calls += 1
        out_dir = self.variant_dir / f"call_{call_idx:03d}"
        out_dir.mkdir(parents=True, exist_ok=True)
        written = []
        for i, w in enumerate(workflows):
            target = out_dir / f"{self.workflow_path.stem}_{i:03d}.json"
            target.write_text(json.dumps(w, indent=2), encoding="utf-8")
            written.append(f"{out_dir.name}/{target.name}")
        return {"count": len(workflows), "written": written, "warnings": warnings}, False


# ---------------------------------------------------------------------------
# Model-response normalization — one interface over the real SDK objects and the
# stub's plain objects so ``run_task`` never branches on client type.
# ---------------------------------------------------------------------------


def _battr(block: Any, name: str, default: Any = None) -> Any:
    if isinstance(block, dict):
        return block.get(name, default)
    return getattr(block, name, default)


@dataclass
class TurnTelemetry:
    """Accumulates one turn's telemetry across its (possibly many) API round-trips."""

    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_input_tokens: int = 0
    cache_creation_input_tokens: int = 0
    num_api_calls: int = 0
    tool_call_count: int = 0
    tool_input_bytes: list[int] = field(default_factory=list)
    tool_output_bytes: list[int] = field(default_factory=list)
    duration_ms: float = 0.0
    # Turn completion status: "ok" when the model ended cleanly, "truncated" when the
    # turn was cut short (non-terminal stop reason or MAX_API_CALLS_PER_TURN exhaustion).
    stop_status: str = "ok"

    def add_usage(self, usage: dict[str, int]) -> None:
        for f in TOKEN_FIELDS:
            setattr(self, f, getattr(self, f) + usage[f])
        self.num_api_calls += 1

    def add_tool(self, input_bytes: int, output_bytes: int) -> None:
        self.tool_call_count += 1
        self.tool_input_bytes.append(input_bytes)
        self.tool_output_bytes.append(output_bytes)


def build_row(
    *,
    task: dict[str, Any],
    turn: int,
    model: str,
    tel: TurnTelemetry,
    outcome: str = "ok",
    note: str | None = None,
) -> dict[str, Any]:
    """Assemble one NDJSON row, shaped identically to arm A's ``arm-a.ndjson``.

    Extra keys (``proxy``, ``outcome``, ``note``) are additive — report.mjs reads only
    the shared schema, so the proxy labelling rides along harmlessly.
    """
    token_usage = {f: getattr(tel, f) for f in TOKEN_FIELDS}
    cost = estimate_cost_usd(model, token_usage)
    cost = 0.0 if cost is None else cost
    ti, to = tel.tool_input_bytes, tel.tool_output_bytes
    payload = [i + o for i, o in zip(ti, to)]
    duration_ms = int(round(tel.duration_ms))
    return {
        "arm": "B",
        "proxy": True,  # SANCTIONED PROXY — not Kishore's real prototype (see module docstring)
        "task": task["id"],
        "title": task["title"],
        "turn": turn,
        "msg_id": f"bench-{task['id']}-{turn}",
        "model": model,
        **token_usage,
        "cost_usd": cost,
        # No separate SDK cost figure on the raw Messages API — mirror our own estimate.
        "sdk_cost_usd": cost,
        "num_turns": tel.num_api_calls,
        "duration_ms": duration_ms,
        "duration_api_ms": duration_ms,
        "wall_ms": duration_ms,
        "tool_call_count": tel.tool_call_count,
        "tool_input_bytes_total": sum(ti),
        "tool_input_bytes_max": max(ti) if ti else 0,
        "tool_output_bytes_total": sum(to),
        "tool_output_bytes_max": max(to) if to else 0,
        "tool_payload_bytes_total": sum(payload),
        "tool_payload_bytes_max": max(payload) if payload else 0,
        # Arm B never compacts a full transcript (it round-trips slot manifests, not JSON).
        "compactions": 0,
        "outcome": outcome,
        "note": note,
    }


# ---------------------------------------------------------------------------
# Driver — the agent loop. This is the swap-in seam for the real prototype.
# ---------------------------------------------------------------------------

MAX_API_CALLS_PER_TURN = 12  # guard against a tool-loop that never ends


class Driver:
    """Runs one task's prompts through the injected model ``client`` + ``dispatcher``.

    To swap in Kishore's real CLI-agent prototype: subclass/replace this driver with
    one that speaks the prototype's protocol, keep ``dispatcher`` (the comfy-cli
    substrate) and ``build_row`` (the telemetry shape), and everything downstream —
    NDJSON, report.mjs — stays identical.
    """

    def __init__(self, client: Any, model: str = DEFAULT_MODEL, max_tokens: int = 2048):
        self.client = client
        self.model = model
        self.max_tokens = max_tokens

    def _create(self, messages: list[dict[str, Any]]) -> Any:
        return self.client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            system=SYSTEM_PROMPT,
            tools=TOOLS,
            messages=messages,
        )

    def run_turn(
        self, dispatcher: ToolDispatcher, prompt: str, messages: list[dict[str, Any]] | None = None
    ) -> TurnTelemetry:
        """Drive one user prompt to completion, dispatching tools; return its telemetry.

        ``messages`` is the running transcript. Pass the SAME list across a task's turns
        (t4's one-session, three-edit flow) so the transcript grows turn-over-turn and the
        cache/regrowth telemetry mirrors arm A's resumed session; omit it for a fresh,
        single-turn task and each turn starts clean.
        """
        if messages is None:
            messages = []
        tel = TurnTelemetry()
        messages.append({"role": "user", "content": prompt})
        completed = False
        for _ in range(MAX_API_CALLS_PER_TURN):
            t0 = time.perf_counter()
            resp = self._create(messages)
            # A stub can supply a synthetic latency so --dry-run timings are
            # deterministic; live runs measure the real wall time.
            stub_latency = _battr(resp, "stub_latency_ms")
            tel.duration_ms += stub_latency if stub_latency is not None else (time.perf_counter() - t0) * 1000.0
            tel.add_usage(parse_usage(_battr(resp, "usage")))

            content = _battr(resp, "content") or []
            messages.append({"role": "assistant", "content": content})
            tool_uses = [b for b in content if _battr(b, "type") == "tool_use"]
            stop_reason = _battr(resp, "stop_reason")

            # Always answer every tool_use the model emitted — even when it also signalled
            # a terminal stop_reason. Leaving a tool_use unanswered makes the transcript
            # invalid, and since a task's turns (t4) share one `messages` list, the next
            # turn's create() would 400 on the dangling tool_use.
            if tool_uses:
                tool_results = []
                for tu in tool_uses:
                    tu_input = _battr(tu, "input") or {}
                    result, is_error = dispatcher.dispatch(_battr(tu, "name"), tu_input)
                    result_json = json.dumps(result, ensure_ascii=False)
                    # Measure true UTF-8 payload bytes (not escaped-ASCII lengths) so the
                    # byte comparison against arm A isn't skewed by unicode-heavy payloads.
                    tel.add_tool(
                        len(json.dumps(tu_input, ensure_ascii=False).encode("utf-8")),
                        len(result_json.encode("utf-8")),
                    )
                    tool_results.append(
                        {
                            "type": "tool_result",
                            "tool_use_id": _battr(tu, "id"),
                            "content": result_json,
                            "is_error": is_error,
                        }
                    )
                messages.append({"role": "user", "content": tool_results})

            if stop_reason != "tool_use" or not tool_uses:
                # The model stopped. "end_turn"/"stop_sequence" are clean completions;
                # anything else (e.g. "max_tokens") means the turn was cut short.
                completed = True
                if stop_reason not in ("end_turn", "stop_sequence", "tool_use"):
                    tel.stop_status = "truncated"
                break
        if not completed:
            # Exhausted MAX_API_CALLS_PER_TURN still mid-tool-loop: the transcript ends on a
            # dangling tool_result (user) turn, so the task must not continue onto it.
            tel.stop_status = "truncated"
        return tel


def _seed_for_task(task: dict[str, Any], fixtures_dir: Path) -> Path:
    """Resolve the on-disk seed graph a task edits.

    t2/t4 name a seed (``txt2img-seed`` / ``edit-session-seed``) → committed fixture.
    t1/t3 build from scratch; arm B (a micro-edit toolset) has no from-scratch node
    creation, so it seeds from the txt2img template and micro-edits it — the honest
    proxy of "build small". t3's true ceiling is recorded as a RESULT (see run_arm_b).
    """
    name = task.get("seedWorkflow")
    mapping = {"txt2img-seed": "txt2img_seed.json", "edit-session-seed": "edit_session_seed.json"}
    if name is None:
        # t1/t3 build from scratch; arm B seeds from the txt2img template and micro-edits it.
        return fixtures_dir / "txt2img_seed.json"
    if name not in mapping:
        # An unrecognized non-null seed id (typo / new upstream id) must fail loudly rather
        # than silently run against the wrong seed graph.
        raise ValueError(
            f"unknown seedWorkflow {name!r} for task {task.get('id')!r}; expected one of {sorted(mapping)}"
        )
    return fixtures_dir / mapping[name]


# t3 is the documented ceiling: arm B's slots/set-slot/vary can only EDIT an existing
# graph, and the committed object_info covers only the 6 base SD1.5 classes (no
# LoRA/ControlNet/refiner/upscaler/face-detailer), so a ~150-node BUILD is out of reach.
_T3_CEILING_NOTE = (
    "arm B micro-edit toolset cannot build a ~150-node graph; object_info fixture "
    "covers only the 6 base SD1.5 node classes (no LoRA/ControlNet/refiner/upscaler). "
    "Recorded as a RESULT per BE-2309."
)


def _task_outcome(task_id: str) -> tuple[str, str | None]:
    """Per-task (outcome, note) stamp — 'ceiling' RESULT for t3, plain 'ok' otherwise."""
    if task_id == "t3":
        return "ceiling", _T3_CEILING_NOTE
    return "ok", None


def _resolve_outcome(task_id: str, tel: TurnTelemetry) -> tuple[str, str | None]:
    """Combine the per-task stamp with the turn's actual completion status.

    A truncated turn (non-terminal stop reason or MAX_API_CALLS_PER_TURN exhaustion)
    must NOT be emitted as a successful row — that would corrupt the arm-A/arm-B
    comparison this runner exists to produce.
    """
    if tel.stop_status == "truncated":
        return (
            "truncated",
            "turn cut short (non-terminal stop reason or MAX_API_CALLS_PER_TURN exhausted); "
            "remaining turns skipped to keep the shared transcript valid",
        )
    return _task_outcome(task_id)


def run_task(
    task: dict[str, Any],
    *,
    driver: Driver,
    fixtures_dir: Path,
    object_info_path: Path,
    work_dir: Path,
    on_row: Callable[[dict[str, Any]], None],
) -> None:
    """Run every turn of a task on a FRESH temp copy of its seed, emitting one row per turn.

    A task's turns share ONE transcript (``messages``) so a multi-prompt task (t4) grows its
    context turn-over-turn — the same one-session semantics arm A measures.
    """
    seed = _seed_for_task(task, fixtures_dir)
    workflow_path = work_dir / f"{task['id']}_workflow.json"
    shutil.copyfile(seed, workflow_path)
    dispatcher = ToolDispatcher(object_info_path, workflow_path, work_dir / f"{task['id']}_variants")

    messages: list[dict[str, Any]] = []
    for i, prompt in enumerate(task["prompts"], start=1):
        tel = driver.run_turn(dispatcher, prompt, messages)
        outcome, note = _resolve_outcome(task["id"], tel)
        on_row(build_row(task=task, turn=i, model=driver.model, tel=tel, outcome=outcome, note=note))
        if tel.stop_status == "truncated":
            break  # shared transcript is now invalid; don't run further turns on it


# ---------------------------------------------------------------------------
# Stub client — offline, deterministic. Drives --dry-run and the unit tests.
# ---------------------------------------------------------------------------


class _StubBlock:
    """A minimal content block matching the attribute surface run_task reads."""

    def __init__(self, type: str, **kw: Any):
        self.type = type
        self.text = kw.get("text")
        self.name = kw.get("name")
        self.input = kw.get("input")
        self.id = kw.get("id")


class _StubResponse:
    def __init__(self, content: list[_StubBlock], stop_reason: str, usage: dict[str, int], latency_ms: float):
        self.content = content
        self.stop_reason = stop_reason
        self.usage = usage
        self.stub_latency_ms = latency_ms


def _text(t: str) -> _StubBlock:
    return _StubBlock("text", text=t)


def _tool(name: str, tid: str, tool_input: dict[str, Any]) -> _StubBlock:
    return _StubBlock("tool_use", name=name, id=tid, input=tool_input)


# A scripted proxy plan per (task, turn): the ordered list of responses the stub
# returns. Each response is (content_blocks, stop_reason, usage). Addresses are the
# real slot addresses of the committed sd15 seed fixtures, so the tools genuinely
# read/edit the temp workflow on disk. Token counts are synthetic-but-plausible and
# show modest cache regrowth across a session (t4).
def _usage(inp: int, out: int, cr: int, cw: int) -> dict[str, int]:
    return {
        "input_tokens": inp,
        "output_tokens": out,
        "cache_read_input_tokens": cr,
        "cache_creation_input_tokens": cw,
    }


def default_stub_plan() -> dict[tuple[str, int], list[tuple[list[_StubBlock], str, dict[str, int]]]]:
    """Deterministic proxy scripts keyed by (task_id, 1-based turn index)."""
    return {
        # t1 — "build small": discover slots, set the requested params, produce 2 variants.
        ("t1", 1): [
            ([_tool("slots", "t1a", {})], "tool_use", _usage(700, 120, 2000, 1800)),
            (
                [
                    _tool(
                        "set_slot",
                        "t1b",
                        {
                            "overrides": {
                                "6.text": "a golden retriever puppy playing in autumn leaves, cinematic lighting",
                                "3.steps": 25,
                                "3.cfg": 7,
                                "3.sampler_name": "euler",
                                "3.scheduler": "normal",
                            }
                        },
                    )
                ],
                "tool_use",
                _usage(950, 240, 4200, 900),
            ),
            (
                [_tool("vary", "t1c", {"slots": {"3.seed": [1, 2]}})],
                "tool_use",
                _usage(1100, 90, 5600, 300),
            ),
            (
                [_text("Built the SDXL txt2img template and produced 2 seed variants.")],
                "end_turn",
                _usage(1200, 160, 6400, 0),
            ),
        ],
        # t2 — micro-edit ONE parameter (the canonical arm-B case).
        ("t2", 1): [
            ([_tool("slots", "t2a", {})], "tool_use", _usage(650, 80, 1800, 1600)),
            (
                [
                    _tool(
                        "set_slot",
                        "t2b",
                        {"overrides": {"6.text": "a photo of a cat sleeping in a sunbeam on a wooden floor"}},
                    )
                ],
                "tool_use",
                _usage(820, 130, 3400, 200),
            ),
            (
                [_text("Changed only the positive prompt; every other node is untouched.")],
                "end_turn",
                _usage(900, 110, 3900, 0),
            ),
        ],
        # t3 — the CEILING: probe the catalog for the classes the task needs, find
        # them absent, and stop. (run_task stamps outcome="ceiling".)
        ("t3", 1): [
            ([_tool("cql", "t3a", {"node_type": "LoraLoader"})], "tool_use", _usage(900, 140, 2200, 2000)),
            ([_tool("cql", "t3b", {"node_type": "ControlNetLoader"})], "tool_use", _usage(1050, 120, 3600, 400)),
            (
                [
                    _text(
                        "The requested LoRA / ControlNet / refiner / upscaler classes are not in the offline "
                        "object_info catalog, and my micro-edit tools cannot create ~150 new nodes. Stopping — "
                        "this is the documented arm-B ceiling."
                    )
                ],
                "end_turn",
                _usage(1200, 260, 4300, 0),
            ),
        ],
        # t4 — three successive edits + read-backs in ONE session (transcript regrowth).
        ("t4", 1): [
            (
                [_tool("set_slot", "t4a", {"overrides": {"3.steps": 40, "3.sampler_name": "euler"}})],
                "tool_use",
                _usage(700, 150, 2400, 2200),
            ),
            ([_tool("slots", "t4b", {})], "tool_use", _usage(880, 90, 4100, 500)),
            (
                [_text("Set KSampler to 40 steps and confirmed the sampler settings on read-back.")],
                "end_turn",
                _usage(1000, 130, 4800, 0),
            ),
        ],
        ("t4", 2): [
            (
                [
                    _tool(
                        "set_slot",
                        "t4c",
                        {
                            "overrides": {
                                "6.text": "a dense bioluminescent alien jungle at night, volumetric fog, highly detailed"
                            }
                        },
                    )
                ],
                "tool_use",
                _usage(760, 160, 6200, 700),
            ),
            ([_tool("slots", "t4d", {})], "tool_use", _usage(1020, 100, 7800, 300)),
            (
                [_text("Updated the positive prompt and confirmed the change landed on read-back.")],
                "end_turn",
                _usage(1150, 120, 8600, 0),
            ),
        ],
        ("t4", 3): [
            (
                [_tool("set_slot", "t4e", {"overrides": {"9.filename_prefix": "jungle_alt"}})],
                "tool_use",
                _usage(820, 170, 9400, 600),
            ),
            ([_tool("slots", "t4f", {})], "tool_use", _usage(1120, 110, 11000, 300)),
            (
                [_text("Set the SaveImage filename prefix to jungle_alt and summarized the final graph.")],
                "end_turn",
                _usage(1260, 200, 11800, 0),
            ),
        ],
    }


class StubClient:
    """Offline, scripted Anthropic-compatible client. ``messages.create`` pops the
    next response from the active (task, turn) script set by ``begin_turn``."""

    def __init__(self, plan: dict[tuple[str, int], list[tuple[list[_StubBlock], str, dict[str, int]]]] | None = None):
        self.plan = plan if plan is not None else default_stub_plan()
        self._queue: list[tuple[list[_StubBlock], str, dict[str, int]]] = []
        self.messages = self._Messages(self)

    def begin_turn(self, task_id: str, turn: int) -> None:
        self._queue = list(self.plan.get((task_id, turn), []))
        if not self._queue:
            # A turn with no script terminates immediately (single empty end_turn).
            self._queue = [([_text("(no scripted action)")], "end_turn", _usage(300, 40, 0, 0))]

    class _Messages:
        def __init__(self, outer: StubClient):
            self._outer = outer

        def create(self, **_kwargs: Any) -> _StubResponse:
            if not self._outer._queue:
                # Ran past the script (would only happen if a turn tool-loops longer
                # than planned) — end cleanly rather than hang.
                return _StubResponse([_text("(script exhausted)")], "end_turn", _usage(200, 20, 0, 0), 5.0)
            content, stop, usage = self._outer._queue.pop(0)
            return _StubResponse(content, stop, usage, latency_ms=250.0)


class _StubDriver(Driver):
    """Driver bound to a StubClient — announces each turn to the stub so it serves
    the right (task, turn) script."""

    def run_turn_for(
        self,
        dispatcher: ToolDispatcher,
        task_id: str,
        turn: int,
        prompt: str,
        messages: list[dict[str, Any]] | None = None,
    ) -> TurnTelemetry:
        self.client.begin_turn(task_id, turn)
        return self.run_turn(dispatcher, prompt, messages)


def run_task_stub(
    task: dict[str, Any],
    *,
    driver: _StubDriver,
    fixtures_dir: Path,
    object_info_path: Path,
    work_dir: Path,
    on_row: Callable[[dict[str, Any]], None],
) -> None:
    """run_task variant that tells the stub which (task, turn) script to serve.

    Mirrors ``run_task``: one shared transcript across a task's turns so t4 grows in-session.
    """
    seed = _seed_for_task(task, fixtures_dir)
    workflow_path = work_dir / f"{task['id']}_workflow.json"
    shutil.copyfile(seed, workflow_path)
    dispatcher = ToolDispatcher(object_info_path, workflow_path, work_dir / f"{task['id']}_variants")
    messages: list[dict[str, Any]] = []
    for i, prompt in enumerate(task["prompts"], start=1):
        tel = driver.run_turn_for(dispatcher, task["id"], i, prompt, messages)
        outcome, note = _resolve_outcome(task["id"], tel)
        on_row(build_row(task=task, turn=i, model=driver.model, tel=tel, outcome=outcome, note=note))
        if tel.stop_status == "truncated":
            break  # shared transcript is now invalid; don't run further turns on it


# ---------------------------------------------------------------------------
# Live client — real Anthropic SDK. Imported lazily so --dry-run + tests stay offline.
# ---------------------------------------------------------------------------


def build_live_client() -> Any:
    """Return a real ``anthropic.Anthropic`` client (reads ANTHROPIC_API_KEY from env).

    Swap-in seam: to point arm B at Kishore's real CLI-agent prototype, return a
    client object exposing ``messages.create(...)`` with the same response surface
    (``.content``, ``.stop_reason``, ``.usage``), or replace ``Driver`` entirely.
    """
    try:
        import anthropic
    except ImportError as e:  # pragma: no cover - only hit on a live run without the extra
        raise SystemExit(
            "The `anthropic` package is required for a live run. Install the bench extra:\n"
            "    pip install -e '.[bench]'\n"
            "and set ANTHROPIC_API_KEY. Use --dry-run for an offline stubbed run."
        ) from e
    return anthropic.Anthropic()


# ---------------------------------------------------------------------------
# Data locations + task loading
# ---------------------------------------------------------------------------

_HERE = Path(__file__).resolve().parent
FIXTURES_DIR = _HERE / "fixtures"
TASKS_PATH = _HERE / "tasks.json"
DEFAULT_OBJECT_INFO = FIXTURES_DIR / "object_info.json"
DEFAULT_OUT = _HERE / "out" / "arm-b.ndjson"


def _validate_task_id(task_id: Any) -> str:
    """Reject task ids unsafe to interpolate into a filesystem path.

    ``task['id']`` flows straight into scratch-file/variant-dir names; a crafted id with
    a path separator or ``..`` would let a malicious ``--tasks`` file escape ``work_dir``.
    """
    if not isinstance(task_id, str) or not task_id:
        raise ValueError(f"task id must be a non-empty string, got {task_id!r}")
    if task_id in (".", "..") or "/" in task_id or "\\" in task_id or "\x00" in task_id:
        raise ValueError(f"task id {task_id!r} may not contain path separators or '..'")
    return task_id


def load_tasks(path: str | Path = TASKS_PATH) -> list[dict[str, Any]]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    tasks = data["tasks"]
    # Normalize: every task has a `prompts` list (matching child 1's taskPrompts()).
    for t in tasks:
        _validate_task_id(t.get("id"))
        if "prompts" not in t or not isinstance(t["prompts"], list):
            t["prompts"] = [t["prompt"]] if t.get("prompt") else []
    return tasks


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def run(
    *,
    dry_run: bool,
    out_path: Path,
    object_info_path: Path,
    tasks_path: Path,
    model: str,
    work_dir: Path,
    only: list[str] | None = None,
) -> list[dict[str, Any]]:
    """Run the whole matrix, write NDJSON to ``out_path``, and return the rows."""
    if resolve_pricing(model) is None:
        # Fail fast: an unknown model would price every turn at $0 (estimate_cost_usd → None
        # → 0.0), making the run look free and silently skewing the dollar comparison.
        known = ", ".join(entry[0] for entry in PRICING)
        raise ValueError(f"model {model!r} has no pricing entry; add one to PRICING (known prefixes: {known}).")
    tasks = load_tasks(tasks_path)
    if only:
        tasks = [t for t in tasks if t["id"] in set(only)]
    work_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w", encoding="utf-8") as fh:

        def on_row(row: dict[str, Any]) -> None:
            rows.append(row)
            fh.write(json.dumps(row) + "\n")
            fh.flush()

        if dry_run:
            driver = _StubDriver(StubClient(), model=model)
            for task in tasks:
                run_task_stub(
                    task,
                    driver=driver,
                    fixtures_dir=FIXTURES_DIR,
                    object_info_path=object_info_path,
                    work_dir=work_dir,
                    on_row=on_row,
                )
        else:
            driver = Driver(build_live_client(), model=model)
            for task in tasks:
                run_task(
                    task,
                    driver=driver,
                    fixtures_dir=FIXTURES_DIR,
                    object_info_path=object_info_path,
                    work_dir=work_dir,
                    on_row=on_row,
                )
    return rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m comfy_cli.bench.run_arm_b",
        description="Arm-B (CLI micro-edit PROXY) runner for the BE-2302 A/B benchmark.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Offline: drive a stubbed model against the committed object_info fixture (no API key, no network).",
    )
    parser.add_argument(
        "--live", action="store_true", help="Real run against the Anthropic API (needs ANTHROPIC_API_KEY)."
    )
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT, help=f"NDJSON output path (default: {DEFAULT_OUT}).")
    parser.add_argument(
        "--object-info", type=Path, default=DEFAULT_OBJECT_INFO, help="object_info JSON (offline catalog)."
    )
    parser.add_argument("--tasks", type=Path, default=TASKS_PATH, help="tasks.json (shared t1–t4 prompts).")
    parser.add_argument("--model", default=DEFAULT_MODEL, help=f"Model id (default: {DEFAULT_MODEL}).")
    parser.add_argument(
        "--work-dir", type=Path, default=None, help="Scratch dir for temp workflows (default: a tempdir)."
    )
    parser.add_argument("--only", nargs="*", default=None, help="Restrict to specific task ids, e.g. --only t2 t4.")
    args = parser.parse_args(argv)

    if not args.dry_run and not args.live:
        parser.error("choose one of --dry-run (offline, stubbed) or --live (real Anthropic API).")
    if args.dry_run and args.live:
        parser.error("--dry-run and --live are mutually exclusive.")

    import tempfile

    work_dir = args.work_dir
    tmp_ctx = None
    if work_dir is None:
        tmp_ctx = tempfile.TemporaryDirectory(prefix="arm-b-bench-")
        work_dir = Path(tmp_ctx.name)
    try:
        rows = run(
            dry_run=args.dry_run,
            out_path=args.out,
            object_info_path=args.object_info,
            tasks_path=args.tasks,
            model=args.model,
            work_dir=Path(work_dir),
            only=args.only,
        )
    finally:
        if tmp_ctx is not None:
            tmp_ctx.cleanup()

    mode = "dry-run (stubbed, PROXY)" if args.dry_run else "live (PROXY)"
    total_cost = sum(r["cost_usd"] for r in rows)
    sys.stderr.write(
        f"✓ arm B {mode}: wrote {len(rows)} row(s) → {args.out}  (est ${total_cost:.4f}). "
        f"Feed to child 1's report.mjs: `node bench/report.mjs --arm-b {args.out}`.\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
