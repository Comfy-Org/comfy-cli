"""``set_node_field`` — a per-field write to a node's durable scalar state.

The vocabulary can already express "this node changed" only as an ``add_node``
upsert, which replaces the WHOLE node: it rewrites the node's widget values and
clears its widget stamps, so a title or flag change concurrent with a widget
write on that node discards the write. ``set_node_field`` claims one LWW
register per ``(node, field)`` instead.

These tests pin the CLI half of that: the ``comfy workflow set-node-field``
command mints a replayable op, ``apply_op`` replays it, the field allowlist is
closed, and it rides inside a batch.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from comfy_cli import workflow_ops
from comfy_cli.caller import Caller
from comfy_cli.command import workflow as workflow_cmd
from comfy_cli.cql.engine import Graph
from comfy_cli.output.renderer import OutputMode, Renderer, reset_renderer_for_testing, set_renderer


@pytest.fixture(autouse=True)
def reset_singleton():
    reset_renderer_for_testing()
    yield
    reset_renderer_for_testing()


def _force_json_renderer():
    r = Renderer.resolve(
        is_stdout_tty=False,
        env={},
        caller=Caller(kind="user", agentic=False, source_env=None),
        json_flag=True,
    )
    r.mode = OutputMode.JSON
    set_renderer(r)
    return r


def _object_info() -> dict[str, Any]:
    return {
        "TinyLoader": {
            "input": {"required": {"ckpt_name": [["a.safetensors"]]}},
            "input_order": {"required": ["ckpt_name"]},
            "output": ["MODEL"],
            "output_name": ["MODEL"],
            "category": "loaders",
            "display_name": "Tiny Loader",
            "python_module": "nodes",
        },
    }


def _graph() -> Graph:
    return Graph.from_object_info(_object_info())


def _populated() -> dict[str, Any]:
    return {
        "id": "wf-1",
        "revision": 0,
        "nodes": [
            {
                "id": 1,
                "type": "TinyLoader",
                "pos": [0, 0],
                "title": "Tiny Loader",
                "mode": 0,
                "flags": {},
                "inputs": [],
                "outputs": [],
                "widgets_values": ["a.safetensors"],
            },
        ],
        "links": [],
        "last_node_id": 1,
        "last_link_id": 0,
    }


def _run(args: list[str], capsys) -> dict[str, Any]:
    _force_json_renderer()
    runner = CliRunner()
    result = runner.invoke(workflow_cmd.app, args, standalone_mode=False)
    captured = capsys.readouterr().out
    if not captured.strip():
        captured = result.stdout or ""
    for line in reversed(captured.strip().splitlines()):
        try:
            return json.loads(line)
        except json.JSONDecodeError:
            continue
    raise AssertionError(f"no JSON envelope (rc={result.exit_code}, exc={result.exception}, out={captured[:600]})")


class TestSetNodeFieldCommand:
    @pytest.mark.parametrize(
        ("field", "raw", "expected"),
        [
            ("title", "Renamed", "Renamed"),
            ("mode", "4", 4),
            ("flags.collapsed", "true", True),
            ("flags.pinned", "true", True),
        ],
    )
    def test_writes_the_field_and_emits_a_replayable_op(self, tmp_path: Path, capsys, field, raw, expected):
        wf = tmp_path / "wf_set_node_field.json"
        wf.write_text(json.dumps(_populated()), encoding="utf-8")

        env = _run(["set-node-field", str(wf), "1", field, raw], capsys)

        assert env["ok"] is True, env
        op = env["data"]["op"]
        assert op["op"] == "set_node_field"
        # Pin the TYPE, not just the stringified value: sibling commands
        # (`_split_addr` for set-widget/connect, delete/delete-nodes) coerce a
        # numeric node id to `int` before minting, and `set-node-field` must
        # match — otherwise the same node id splits across two LWW registers
        # for a non-normalizing consumer (`str(op["node_id"]) == "1"` would
        # pass even if `op["node_id"]` were left as the raw string `"1"`).
        assert op["node_id"] == 1
        assert isinstance(op["node_id"], int)
        assert op["field"] == field
        assert op["value"] == expected
        assert op["stamp"] == [op["base_version"], op["actor"]]

        node = json.loads(wf.read_text(encoding="utf-8"))["nodes"][0]
        head, _, leaf = field.partition(".")
        assert (node[head][leaf] if leaf else node[head]) == expected

    def test_rejects_a_field_outside_the_allowlist(self, tmp_path: Path, capsys):
        """`widgets_values` is `set_widget`'s register and `type` is node
        identity; neither may be moved by a field write."""
        wf = tmp_path / "wf_bad_field.json"
        before = _populated()
        wf.write_text(json.dumps(before), encoding="utf-8")

        env = _run(["set-node-field", str(wf), "1", "widgets_values", "[]"], capsys)

        assert env["ok"] is False
        assert json.loads(wf.read_text(encoding="utf-8")) == before

    def test_rejects_a_missing_node(self, tmp_path: Path, capsys):
        wf = tmp_path / "wf_missing_node.json"
        before = _populated()
        wf.write_text(json.dumps(before), encoding="utf-8")

        env = _run(["set-node-field", str(wf), "999", "title", "ghost"], capsys)

        assert env["ok"] is False
        assert json.loads(wf.read_text(encoding="utf-8")) == before


class TestSetNodeFieldReplay:
    def test_apply_op_replays_the_op_idempotently(self):
        workflow = _populated()
        _, op = workflow_ops.set_node_field(_populated(), "1", "title", "Renamed", actor="cli", base_version=1)

        workflow_ops.apply_op(workflow, op, _graph())
        assert workflow["nodes"][0]["title"] == "Renamed"

        workflow_ops.apply_op(workflow, op, _graph())
        assert workflow["_applied_ops"].count(op["op_id"]) == 1

    def test_two_fields_of_one_node_do_not_contend(self):
        """One register per (node, field): a title write and a flag write on
        the same node claim different LWW targets, so neither drops the other
        whatever order they replay in."""
        title_op = workflow_ops.set_node_field(_populated(), "1", "title", "T", actor="a", base_version=1)[1]
        flag_op = workflow_ops.set_node_field(_populated(), "1", "flags.collapsed", True, actor="b", base_version=1)[1]

        assert workflow_ops._write_target(title_op) != workflow_ops._write_target(flag_op)
        assert not workflow_ops.detect_conflict(title_op, flag_op)

        for order in ([title_op, flag_op], [flag_op, title_op]):
            workflow = _populated()
            for op in order:
                workflow_ops.apply_op(workflow, op, _graph())
            node = workflow["nodes"][0]
            assert node["title"] == "T"
            assert node["flags"]["collapsed"] is True

    def test_two_writers_of_one_field_resolve_by_stamp(self):
        early = workflow_ops.set_node_field(_populated(), "1", "title", "early", actor="a", base_version=1)[1]
        late = workflow_ops.set_node_field(_populated(), "1", "title", "late", actor="b", base_version=2)[1]

        for order in ([early, late], [late, early]):
            workflow = _populated()
            for op in order:
                workflow_ops.apply_op(workflow, op, _graph())
            assert workflow["nodes"][0]["title"] == "late"


class TestSetNodeFieldIsBatchable:
    def test_rides_inside_a_batch(self):
        # "node" is the canonical spec key (matches `set_widget`'s "node",
        # `delete_node`'s "node" — every sibling branch in `apply_specs`).
        workflow, ops, _aliases = workflow_ops.apply_specs(
            _populated(),
            _graph(),
            [{"op": "set_node_field", "node": "1", "field": "title", "value": "batched"}],
        )

        assert [op["op"] for op in ops] == ["set_node_field"]
        assert workflow["nodes"][0]["title"] == "batched"

    def test_missing_node_key_is_the_standard_missing_field_error(self):
        """`spec.get("node")` used to silently pass `None` through instead of
        raising — and since node matching uses `str()`, `None` could
        spuriously match a node whose id happens to be the string `"None"`.
        A missing `node` must fail loudly with the same message every other
        op kind gets for a missing required field, and must never reach the
        node lookup at all."""
        workflow = _populated()
        workflow["nodes"].append(
            {
                "id": "None",
                "type": "TinyLoader",
                "pos": [100, 0],
                "title": "Decoy",
                "mode": 0,
                "flags": {},
                "inputs": [],
                "outputs": [],
                "widgets_values": ["a.safetensors"],
            }
        )
        before = copy.deepcopy(workflow)

        with pytest.raises(ValueError, match=r"spec #0 \(set_node_field\) is missing required field 'node'"):
            workflow_ops.apply_specs(
                workflow,
                _graph(),
                [{"op": "set_node_field", "field": "title", "value": "hijacked"}],
            )

        # Nothing applied: the decoy "None"-id node was never touched.
        assert workflow == before

    def test_undocumented_node_id_key_alone_is_rejected_not_silently_accepted(self):
        """`node_id` was never the documented spec key; a spec carrying only
        it must fail the same way a spec missing `node` entirely does, not be
        quietly accepted as an alias for `node`."""
        with pytest.raises(ValueError, match=r"spec #0 \(set_node_field\) is missing required field 'node'"):
            workflow_ops.apply_specs(
                _populated(),
                _graph(),
                [{"op": "set_node_field", "node_id": "1", "field": "title", "value": "batched"}],
            )


class TestSetNodeFieldValueValidationAtMint:
    """``set_node_field`` allowlists the field NAME but, before this fix, never
    validated the VALUE. A bogus ``mode`` (a string, an out-of-range int, a
    bool, a container) would sail straight into the document and later break
    ``workflow_to_api``'s exact ``mode in (_MODE_MUTED, _MODE_BYPASS)`` check
    or make ``ls-nodes``/``print`` raise ``TypeError: unhashable type``."""

    @pytest.mark.parametrize(
        ("field", "bad_value"),
        [
            ("title", 123),
            ("title", ["not", "a", "string"]),
            ("mode", "bypass"),
            ("mode", 99),
            ("mode", True),  # bool is an int subclass but not a legal mode
            ("mode", 2.0),
            ("mode", [4]),
            ("flags.collapsed", "true"),
            ("flags.collapsed", 1),
            ("flags.pinned", "yes"),
            ("flags.pinned", 0),
        ],
    )
    def test_rejects_a_malformed_value_at_mint_time(self, field, bad_value):
        with pytest.raises(ValueError, match="malformed_op"):
            workflow_ops.set_node_field(_populated(), "1", field, bad_value, actor="cli", base_version=0)
        # Nothing mutated: mint-time validation runs before `apply_op`.
        workflow = _populated()
        with pytest.raises(ValueError):
            workflow_ops.set_node_field(workflow, "1", field, bad_value, actor="cli", base_version=0)
        assert workflow == _populated()

    @pytest.mark.parametrize(
        ("field", "good_value"),
        [
            ("title", "A fine title"),
            ("title", None),
            ("mode", 0),
            ("mode", 4),
            ("mode", None),
            ("flags.collapsed", True),
            ("flags.collapsed", False),
            ("flags.collapsed", None),
            ("flags.pinned", True),
            ("flags.pinned", None),
        ],
    )
    def test_accepts_every_legal_value_including_null(self, field, good_value):
        workflow_ops.set_node_field(_populated(), "1", field, good_value, actor="cli", base_version=0)

    def test_cli_command_rejects_a_malformed_mode(self, tmp_path: Path, capsys):
        wf = tmp_path / "wf_bad_mode.json"
        before = _populated()
        wf.write_text(json.dumps(before), encoding="utf-8")

        env = _run(["set-node-field", str(wf), "1", "mode", '"bypass"'], capsys)

        assert env["ok"] is False, env
        assert json.loads(wf.read_text(encoding="utf-8")) == before


class TestSetNodeFieldValueValidationAtReplay:
    """A peer-authored op that bypasses the CLI's mint-time check (or an old
    replica replaying a stale op) must be rejected by ``apply_op`` too — the
    replay path gets no less scrutiny than mint."""

    @pytest.mark.parametrize(
        ("field", "bad_value"),
        [
            ("mode", "bypass"),
            ("mode", 99),
            ("mode", True),
            ("title", 123),
            ("flags.collapsed", "yes"),
        ],
    )
    def test_apply_op_rejects_a_malformed_value(self, field, bad_value):
        workflow = _populated()
        before = copy.deepcopy(workflow)
        op = {
            "op": "set_node_field",
            "op_id": "deadbeef",
            "actor": "peer",
            "base_version": 0,
            "stamp": [0, "peer"],
            "node_id": 1,
            "field": field,
            "value": bad_value,
        }

        with pytest.raises(ValueError, match="malformed_op"):
            workflow_ops.apply_op(workflow, op, _graph())

        # No partial mutation: validation runs before the node is touched.
        assert workflow["nodes"] == before["nodes"]


class TestSetNodeFieldFlagsContainerRobustness:
    """A node's ``flags`` is not schema-forbidden from being ``null`` or any
    other non-dict shape. ``node.setdefault(head, {})`` assumed dict-shaped,
    so a malformed ``flags`` raised a raw ``TypeError``/``AttributeError`` that
    escaped the handler's ``ValueError``/``KeyError`` envelope wrapping and
    corrupted replay."""

    @pytest.mark.parametrize("malformed_flags", [None, "oops", 42, ["not", "a", "dict"]])
    def test_write_coerces_a_non_dict_flags_container_instead_of_crashing(self, malformed_flags):
        workflow = _populated()
        workflow["nodes"][0]["flags"] = malformed_flags

        workflow_ops.set_node_field(workflow, "1", "flags.collapsed", True, actor="cli", base_version=0)

        assert workflow["nodes"][0]["flags"] == {"collapsed": True}

    @pytest.mark.parametrize("malformed_flags", [None, "oops", 42, ["not", "a", "dict"]])
    def test_null_delete_on_a_non_dict_flags_is_a_true_no_op(self, malformed_flags):
        """A ``value: null`` delete on a node whose ``flags`` is absent or
        malformed must not materialize an empty ``flags`` object — that would
        leave replicas that did/didn't see the op divergent."""
        workflow = _populated()
        workflow["nodes"][0]["flags"] = malformed_flags

        workflow_ops.set_node_field(workflow, "1", "flags.collapsed", None, actor="cli", base_version=0)

        assert workflow["nodes"][0]["flags"] == malformed_flags

    def test_null_delete_on_a_node_with_no_flags_key_never_materializes_one(self):
        workflow = _populated()
        del workflow["nodes"][0]["flags"]

        workflow_ops.set_node_field(workflow, "1", "flags.pinned", None, actor="cli", base_version=0)

        assert "flags" not in workflow["nodes"][0]

    def test_replay_never_raises_typeerror_or_attributeerror(self):
        """Whatever shape validation allows through must at worst raise
        ``ValueError`` — never an uncaught ``TypeError``/``AttributeError``
        that skips the handler's envelope wrapping."""
        workflow = _populated()
        workflow["nodes"][0]["flags"] = None
        op = {
            "op": "set_node_field",
            "op_id": "abc123",
            "actor": "peer",
            "base_version": 0,
            "stamp": [0, "peer"],
            "node_id": 1,
            "field": "flags.collapsed",
            "value": None,
        }
        # Must not raise at all (a null-delete on a null flags is a no-op).
        workflow_ops.apply_op(workflow, op, _graph())
        assert workflow["nodes"][0]["flags"] is None


class TestSetNodeFieldConflictEqualValueCarveOut:
    """``set_node_field`` has the same LWW-register semantics as
    ``set_widget``, so two actors independently writing the identical value
    (e.g. both collapsing a node) must not be escalated to ask-to-merge."""

    def test_two_writers_of_the_identical_value_do_not_conflict(self):
        a = workflow_ops.set_node_field(_populated(), "1", "flags.collapsed", True, actor="a", base_version=1)[1]
        b = workflow_ops.set_node_field(_populated(), "1", "flags.collapsed", True, actor="b", base_version=1)[1]

        assert workflow_ops._write_target(a) == workflow_ops._write_target(b)
        assert workflow_ops.detect_conflict(a, b) is False

    def test_two_writers_of_different_values_still_conflict(self):
        a = workflow_ops.set_node_field(_populated(), "1", "title", "Alice's title", actor="a", base_version=1)[1]
        b = workflow_ops.set_node_field(_populated(), "1", "title", "Bob's title", actor="b", base_version=1)[1]

        assert workflow_ops.detect_conflict(a, b) is True


class TestSetNodeFieldDiscoveryRegistration:
    """``comfy discover`` must advertise an ``output_schema`` for
    ``set-node-field`` like every sibling structured-edit command."""

    def test_registered_in_command_schemas(self):
        from comfy_cli.discovery import COMMAND_SCHEMAS

        assert COMMAND_SCHEMAS.get("comfy workflow set-node-field") == "workflow"
