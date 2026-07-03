"""Offline unit tests for the arm-B (CLI micro-edit proxy) bench runner.

All tests use the committed object_info + seed fixtures and a stubbed model client —
NO live Anthropic calls, NO network — so CI stays offline. They cover the two pieces
BE-2309 calls out (tool dispatch + usage parsing) plus the end-to-end dry run and the
NDJSON row shape that must match arm A's ``arm-a.ndjson`` for report.mjs to consume it.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from comfy_cli.bench import run_arm_b as arm_b

FIXTURES = Path(arm_b.__file__).resolve().parent / "fixtures"
OBJECT_INFO = FIXTURES / "object_info.json"
SEED = FIXTURES / "txt2img_seed.json"

# The report.mjs / arm-a.ndjson shared row schema (extra keys like `proxy` are allowed).
REQUIRED_FIELDS = {
    "arm",
    "task",
    "title",
    "turn",
    "msg_id",
    "model",
    "input_tokens",
    "output_tokens",
    "cache_read_input_tokens",
    "cache_creation_input_tokens",
    "cost_usd",
    "sdk_cost_usd",
    "num_turns",
    "duration_ms",
    "duration_api_ms",
    "wall_ms",
    "tool_call_count",
    "tool_input_bytes_total",
    "tool_input_bytes_max",
    "tool_output_bytes_total",
    "tool_output_bytes_max",
    "tool_payload_bytes_total",
    "tool_payload_bytes_max",
    "compactions",
}


# --------------------------------------------------------------------------- #
# usage parsing
# --------------------------------------------------------------------------- #


def test_parse_usage_from_dict():
    usage = {
        "input_tokens": 100,
        "output_tokens": 50,
        "cache_read_input_tokens": 2000,
        "cache_creation_input_tokens": 300,
    }
    assert arm_b.parse_usage(usage) == usage


def test_parse_usage_from_object():
    class Usage:
        input_tokens = 12
        output_tokens = 7
        cache_read_input_tokens = 900
        cache_creation_input_tokens = 40

    assert arm_b.parse_usage(Usage()) == {
        "input_tokens": 12,
        "output_tokens": 7,
        "cache_read_input_tokens": 900,
        "cache_creation_input_tokens": 40,
    }


def test_parse_usage_missing_and_none_default_to_zero():
    # Partial block (no cache fields) + None → all missing counts are 0, never a crash.
    assert arm_b.parse_usage({"input_tokens": 5}) == {
        "input_tokens": 5,
        "output_tokens": 0,
        "cache_read_input_tokens": 0,
        "cache_creation_input_tokens": 0,
    }
    assert arm_b.parse_usage(None) == dict.fromkeys(arm_b.TOKEN_FIELDS, 0)


def test_parse_usage_ignores_bool_and_nonnumeric():
    assert arm_b.parse_usage({"input_tokens": True, "output_tokens": "x"}) == dict.fromkeys(arm_b.TOKEN_FIELDS, 0)


# --------------------------------------------------------------------------- #
# pricing — must match comfy-inapp-agent/agent-server/usage.mjs exactly
# --------------------------------------------------------------------------- #


def test_estimate_cost_matches_usage_mjs_pricing():
    # Opus 4.8: $5/M in, $25/M out, cache read 0.1x, cache write 1.25x.
    usage = {
        "input_tokens": 3950,
        "output_tokens": 610,
        "cache_read_input_tokens": 18200,
        "cache_creation_input_tokens": 3000,
    }
    expected = (3950 * 5 + 610 * 25 + 18200 * 5 * 0.1 + 3000 * 5 * 1.25) / 1e6
    assert arm_b.estimate_cost_usd("claude-opus-4-8", usage) == pytest.approx(expected)


def test_estimate_cost_unknown_model_is_none():
    assert arm_b.estimate_cost_usd("gpt-4o", {"input_tokens": 1000}) is None
    assert arm_b.estimate_cost_usd(None, {}) is None


def test_resolve_pricing_longest_prefix():
    # A dated snapshot resolves to its family entry by longest prefix.
    entry = arm_b.resolve_pricing("claude-opus-4-8-20260101")
    assert entry is not None and entry[0] == "claude-opus-4-8"


# --------------------------------------------------------------------------- #
# tool dispatch — real CQL engine against the committed fixtures
# --------------------------------------------------------------------------- #


@pytest.fixture
def dispatcher(tmp_path):
    wf = tmp_path / "wf.json"
    wf.write_text(SEED.read_text(encoding="utf-8"), encoding="utf-8")
    return arm_b.ToolDispatcher(OBJECT_INFO, wf, tmp_path / "variants")


def test_dispatch_slots_lists_addresses(dispatcher):
    result, is_error = dispatcher.dispatch("slots", {})
    assert not is_error
    addrs = {s["address"] for s in result["slots"]}
    assert "6.text" in addrs and "3.steps" in addrs
    assert result["count"] == len(result["slots"])


def test_dispatch_cql_found_and_not_found(dispatcher):
    ok, err = dispatcher.dispatch("cql", {"node_type": "KSampler"})
    assert not err and ok["found"] is True and ok["schema"]["id"] == "KSampler"

    miss, err = dispatcher.dispatch("cql", {"node_type": "LoraLoader"})
    assert not err and miss["found"] is False and "suggestions" in miss


def test_dispatch_cql_no_arg_lists_catalog(dispatcher):
    result, err = dispatcher.dispatch("cql", {})
    assert not err and "CheckpointLoaderSimple" in result["available_node_types"]


def test_dispatch_set_slot_mutates_file_on_disk(dispatcher):
    result, is_error = dispatcher.dispatch("set_slot", {"overrides": {"6.text": "a brand new prompt"}})
    assert not is_error and result["applied"] == ["6.text"]
    # The write is a genuine on-disk round-trip: a fresh `slots` read sees the change.
    slots, _ = dispatcher.dispatch("slots", {})
    positive = next(s for s in slots["slots"] if s["address"] == "6.text")
    assert positive["current_value"] == "a brand new prompt"


def test_dispatch_set_slot_bad_address_is_soft_error(dispatcher):
    result, is_error = dispatcher.dispatch("set_slot", {"overrides": {"999.nope": 1}})
    assert is_error and "error" in result  # surfaced to the model, never raised


def test_dispatch_set_slot_requires_overrides(dispatcher):
    result, is_error = dispatcher.dispatch("set_slot", {})
    assert is_error and "error" in result


def test_dispatch_vary_produces_variants(dispatcher):
    result, is_error = dispatcher.dispatch("vary", {"slots": {"3.seed": [1, 2, 3]}})
    assert not is_error and result["count"] == 3 and len(result["written"]) == 3


def test_dispatch_vary_mismatched_lengths_is_error(dispatcher):
    result, is_error = dispatcher.dispatch("vary", {"slots": {"3.seed": [1, 2], "3.steps": [10]}})
    assert is_error and "error" in result


def test_dispatch_unknown_tool_is_error(dispatcher):
    result, is_error = dispatcher.dispatch("nonesuch", {})
    assert is_error and "unknown tool" in result["error"]


def test_dispatch_vary_over_cap_is_error(dispatcher):
    # An unbounded model-supplied list length is capped so a runaway call fails loudly.
    result, is_error = dispatcher.dispatch("vary", {"slots": {"3.seed": list(range(arm_b.MAX_VARIANTS + 1))}})
    assert is_error and "capped" in result["error"]


def test_dispatch_vary_calls_do_not_overwrite(dispatcher):
    # Two vary calls in one task must not clobber each other's variant files.
    r1, e1 = dispatcher.dispatch("vary", {"slots": {"3.seed": [1, 2]}})
    r2, e2 = dispatcher.dispatch("vary", {"slots": {"3.seed": [3, 4]}})
    assert not e1 and not e2
    assert set(r1["written"]).isdisjoint(r2["written"])
    # Both calls' files coexist on disk (namespaced per call).
    written = set(r1["written"]) | set(r2["written"])
    assert len(list((dispatcher.variant_dir).rglob("*.json"))) == len(written) == 4


def test_dispatch_catch_all_error_is_generic(dispatcher, monkeypatch):
    # An unexpected exception must not leak its raw message (possibly a local path) to the model.
    def boom(*_a, **_k):
        raise RuntimeError("/secret/local/path leaked")

    monkeypatch.setattr(dispatcher, "_slots", boom)
    result, is_error = dispatcher.dispatch("slots", {})
    assert is_error and "secret" not in result["error"] and "internal error" in result["error"]


def test_edit_session_seed_ksampler_slots_align():
    # Regression: the KSampler widgets_values must include control_after_generate so slots
    # map positionally (steps=20, cfg=7.0), not off-by-one.
    import tempfile

    with tempfile.TemporaryDirectory() as td:
        wf = Path(td) / "wf.json"
        wf.write_text((FIXTURES / "edit_session_seed.json").read_text(encoding="utf-8"), encoding="utf-8")
        disp = arm_b.ToolDispatcher(OBJECT_INFO, wf, Path(td) / "v")
        slots, _ = disp.dispatch("slots", {})
        by_addr = {s["address"]: s["current_value"] for s in slots["slots"]}
        assert by_addr["3.steps"] == 20
        assert by_addr["3.cfg"] == 7.0
        assert by_addr["3.sampler_name"] == "euler"


# --------------------------------------------------------------------------- #
# driver loop with the stub client
# --------------------------------------------------------------------------- #


def test_stub_driver_runs_a_turn_and_accumulates_telemetry(tmp_path):
    wf = tmp_path / "wf.json"
    wf.write_text(SEED.read_text(encoding="utf-8"), encoding="utf-8")
    disp = arm_b.ToolDispatcher(OBJECT_INFO, wf, tmp_path / "v")
    driver = arm_b._StubDriver(arm_b.StubClient(), model=arm_b.DEFAULT_MODEL)

    tel = driver.run_turn_for(disp, "t2", 1, "change the prompt")
    # t2 script: slots + set_slot + final text = 3 API calls, 2 tool calls.
    assert tel.num_api_calls == 3
    assert tel.tool_call_count == 2
    assert tel.input_tokens > 0 and tel.tool_input_bytes and tel.tool_output_bytes
    # The set_slot in the script really edited the temp workflow.
    assert json.loads(wf.read_text())["nodes"]  # still valid frontend JSON


def test_build_row_has_full_schema_and_proxy_label():
    tel = arm_b.TurnTelemetry()
    tel.add_usage(
        {"input_tokens": 100, "output_tokens": 20, "cache_read_input_tokens": 0, "cache_creation_input_tokens": 0}
    )
    tel.add_tool(50, 200)
    task = {"id": "t2", "title": "Micro-edit one parameter in a seed workflow"}
    row = arm_b.build_row(task=task, turn=1, model=arm_b.DEFAULT_MODEL, tel=tel)
    assert REQUIRED_FIELDS <= set(row.keys())
    assert row["arm"] == "B" and row["proxy"] is True
    assert row["tool_payload_bytes_total"] == 250 and row["tool_payload_bytes_max"] == 250
    assert row["cost_usd"] == row["sdk_cost_usd"]


# --------------------------------------------------------------------------- #
# end-to-end dry run — the acceptance criterion
# --------------------------------------------------------------------------- #


def test_dry_run_writes_well_formed_ndjson(tmp_path):
    out = tmp_path / "arm-b.ndjson"
    rows = arm_b.run(
        dry_run=True,
        out_path=out,
        object_info_path=OBJECT_INFO,
        tasks_path=arm_b.TASKS_PATH,
        model=arm_b.DEFAULT_MODEL,
        work_dir=tmp_path / "work",
    )
    # One row per (task, turn): t1, t2, t3, t4x3 = 6.
    assert len(rows) == 6
    file_rows = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert file_rows == rows
    for r in file_rows:
        assert REQUIRED_FIELDS <= set(r.keys())
        assert r["arm"] == "B" and r["proxy"] is True
        assert isinstance(r["cost_usd"], int | float) and r["cost_usd"] >= 0

    by_task = {(r["task"], r["turn"]): r for r in file_rows}
    # t3 is the documented ceiling, recorded as a RESULT.
    assert by_task[("t3", 1)]["outcome"] == "ceiling"
    assert by_task[("t3", 1)]["note"]
    # t4 is a 3-turn session; cache-read tokens grow turn-over-turn (transcript regrowth).
    assert (
        by_task[("t4", 1)]["cache_read_input_tokens"]
        < by_task[("t4", 2)]["cache_read_input_tokens"]
        < by_task[("t4", 3)]["cache_read_input_tokens"]
    )


def test_dry_run_only_filter(tmp_path):
    out = tmp_path / "arm-b.ndjson"
    rows = arm_b.run(
        dry_run=True,
        out_path=out,
        object_info_path=OBJECT_INFO,
        tasks_path=arm_b.TASKS_PATH,
        model=arm_b.DEFAULT_MODEL,
        work_dir=tmp_path / "work",
        only=["t2"],
    )
    assert [r["task"] for r in rows] == ["t2"]


def test_tasks_json_prompts_are_present_and_nonempty():
    tasks = arm_b.load_tasks()
    ids = [t["id"] for t in tasks]
    assert ids == ["t1", "t2", "t3", "t4"]
    assert len(next(t for t in tasks if t["id"] == "t4")["prompts"]) == 3
    for t in tasks:
        for p in t["prompts"]:
            assert isinstance(p, str) and p.strip()


# --------------------------------------------------------------------------- #
# input validation + fail-fast guards
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("bad", ["../evil", "a/b", "a\\b", "..", ".", "", None, 5])
def test_validate_task_id_rejects_unsafe(bad):
    with pytest.raises(ValueError):
        arm_b._validate_task_id(bad)


def test_validate_task_id_accepts_normal():
    assert arm_b._validate_task_id("t4") == "t4"


def test_seed_for_task_unknown_seed_fails_fast():
    with pytest.raises(ValueError):
        arm_b._seed_for_task({"id": "tx", "seedWorkflow": "bogus-seed"}, FIXTURES)
    # None seed (t1/t3 build-from-scratch) still resolves to the txt2img template.
    assert arm_b._seed_for_task({"id": "t1", "seedWorkflow": None}, FIXTURES).name == "txt2img_seed.json"


def test_run_unknown_model_fails_fast(tmp_path):
    with pytest.raises(ValueError):
        arm_b.run(
            dry_run=True,
            out_path=tmp_path / "out.ndjson",
            object_info_path=OBJECT_INFO,
            tasks_path=arm_b.TASKS_PATH,
            model="gpt-4o",
            work_dir=tmp_path / "work",
        )


def test_truncated_turn_marks_outcome_and_stops(tmp_path):
    # A turn cut short (non-terminal stop reason) is stamped "truncated" and the task's
    # remaining turns are skipped so the shared transcript never goes invalid.
    plan = {("t4", 1): [([arm_b._text("cut off mid-thought")], "max_tokens", arm_b._usage(100, 20, 0, 0))]}
    driver = arm_b._StubDriver(arm_b.StubClient(plan), model=arm_b.DEFAULT_MODEL)
    task = {"id": "t4", "title": "edit session", "seedWorkflow": "edit-session-seed", "prompts": ["a", "b", "c"]}
    rows: list[dict] = []
    arm_b.run_task_stub(
        task,
        driver=driver,
        fixtures_dir=FIXTURES,
        object_info_path=OBJECT_INFO,
        work_dir=tmp_path,
        on_row=rows.append,
    )
    assert len(rows) == 1  # turns 2 and 3 skipped
    assert rows[0]["outcome"] == "truncated" and rows[0]["note"]
