"""`comfy workflow apply --ack summary|full` — envelope acknowledgment detail.

The default `full` ack echoes every applied op back in `data.ops` — for a big
batch that is the whole batch again, and an agent that just SENT the ops
learns nothing new from the echo. `--ack summary` returns a compact receipt
instead. The summary payload shape is PINNED (its field names feed the cloud
field-contract manifest):

    {count, ops_by_kind: {<kind>: n}, nodes_added: [ids], nodes_deleted: [ids],
     aliases: {alias: id}, base_version, version, changed}

with NO `ops` echo. `--ack` only changes the SHAPE of the payload — never the
outcome: file writes, version bookkeeping, error codes, and exit semantics are
identical in both modes, and the default (no flag) payload stays byte-identical
to today's.

`foreach` deliberately has no `--ack`: its payload never echoed ops (it reports
`{recipe, count, out_dir, written}`), so there is nothing to summarize.
"""

from __future__ import annotations

import json
from typing import Any

import pytest
from test_workflow_edit import (  # type: ignore[import-not-found]
    _base_workflow,
    _force_json_renderer,
    _graph,
    _run,
    _write,
    reset_singleton,  # noqa: F401  (autouse fixture)
)
from typer.testing import CliRunner

from comfy_cli import workflow_ops
from comfy_cli.command import workflow as workflow_cmd
from comfy_cli.command import workflow_edit

# The pinned summary payload contract — exactly these keys, nothing else.
SUMMARY_KEYS = {
    "count",
    "ops_by_kind",
    "nodes_added",
    "nodes_deleted",
    "aliases",
    "base_version",
    "version",
    "changed",
}


@pytest.fixture
def patched_graph(monkeypatch):
    monkeypatch.setattr(workflow_edit, "_get_graph", lambda *a, **kw: _graph())


def _empty(tmp_path):
    return _write(tmp_path, {"nodes": [], "links": [], "last_node_id": 0, "last_link_id": 0})


def _ops_file(tmp_path, specs: list[dict], name: str = "ops.json"):
    p = tmp_path / name
    p.write_text(json.dumps(specs), encoding="utf-8")
    return p


def _three_op_specs() -> list[dict]:
    """A 3-op batch: two aliased add_nodes + a connect through the aliases."""
    return [
        {"op": "add_node", "class_type": "CheckpointLoaderSimple", "as": "ckpt"},
        {"op": "add_node", "class_type": "CLIPTextEncode", "as": "pos"},
        {"op": "connect", "from": "ckpt.CLIP", "to": "pos.clip"},
    ]


class TestAckSummary:
    def test_apply_ops_ack_summary_returns_counts_and_aliases(self, patched_graph, tmp_path, capsys):
        path = _empty(tmp_path)
        ops_path = _ops_file(tmp_path, _three_op_specs())
        env = _run(["apply", str(path), "--ops", str(ops_path), "--ack", "summary"], capsys)
        assert env["ok"] is True, env
        data = env["data"]
        # Exactly the pinned shape — a stray extra field would silently enter
        # the cloud field-contract manifest, a missing one would break it.
        assert set(data) == SUMMARY_KEYS, data
        assert "ops" not in data
        assert data["count"] == 3
        assert data["ops_by_kind"] == {"add_node": 2, "connect": 1}
        assert set(data["aliases"]) == {"ckpt", "pos"}
        assert sorted(data["nodes_added"]) == sorted(data["aliases"].values())
        assert data["nodes_deleted"] == []
        assert data["base_version"] == 0
        assert data["version"] == 3  # base_version + len(ops), same math as full mode
        assert data["changed"] is True
        assert env["changed"] is True  # envelope-level flag identical to full mode
        # The file really was written — summary changes the receipt, not the write.
        assert len(json.loads(path.read_text())["nodes"]) == 2

    def test_ack_summary_reports_deleted_nodes(self, patched_graph, tmp_path, capsys):
        path = _write(tmp_path, _base_workflow())
        ops_path = _ops_file(tmp_path, [{"op": "delete_node", "node": 7}])
        env = _run(["apply", str(path), "--ops", str(ops_path), "--ack", "summary"], capsys)
        assert env["ok"] is True, env
        data = env["data"]
        assert set(data) == SUMMARY_KEYS
        assert data["ops_by_kind"] == {"delete_node": 1}
        assert data["nodes_added"] == []
        assert data["nodes_deleted"] == [7]

    def test_ack_rejects_unknown_value(self, patched_graph, tmp_path, capsys):
        path = _empty(tmp_path)
        ops_path = _ops_file(tmp_path, _three_op_specs())
        env = _run(["apply", str(path), "--ops", str(ops_path), "--ack", "bogus"], capsys)
        assert env["ok"] is False
        assert env["error"]["code"] == "workflow_edit_invalid"


class TestAckDefaultUnchanged:
    def test_apply_ack_default_byte_identical(self, patched_graph, tmp_path, capsys, monkeypatch):
        """`--ack full` must be byte-identical to today's default payload.

        Soundness of the assertion: node ids (`mint_id`, `random.getrandbits`)
        and op ids (`uuid.uuid4`) are the ONLY nondeterminism in the apply
        path (layout is documented "no randomness, no clock"; the CRDT stamp
        is [base_version, actor]). We pin both to counters, run the same batch
        on the same file path twice — resetting the file content and the
        counters in between — so the two invocations are observationally
        identical except for the explicit `--ack full` flag. Byte-equal
        envelope lines then prove `--ack full` == default. The shape
        assertions below additionally pin that default to TODAY'S payload
        (full `ops` echo + aliases), so a change to either mode fails here.
        """
        state = {"node": 0, "op": 0}

        def fake_mint() -> int:
            state["node"] += 1
            return (1 << 40) + state["node"]

        class _FakeUUID:
            def __init__(self, n: int):
                self.hex = f"{n:032x}"

        class _FakeUuidModule:
            @staticmethod
            def uuid4() -> Any:
                state["op"] += 1
                return _FakeUUID(state["op"])

        monkeypatch.setattr(workflow_ops, "mint_id", fake_mint)
        monkeypatch.setattr(workflow_ops, "uuid", _FakeUuidModule)

        initial = json.dumps({"nodes": [], "links": [], "last_node_id": 0, "last_link_id": 0}, indent=2)
        path = tmp_path / "wf.json"
        ops_path = _ops_file(tmp_path, _three_op_specs())

        def run_once(extra: list[str]) -> str:
            path.write_text(initial, encoding="utf-8")
            state["node"] = state["op"] = 0
            _force_json_renderer()
            runner = CliRunner()
            result = runner.invoke(
                workflow_cmd.app, ["apply", str(path), "--ops", str(ops_path), *extra], standalone_mode=False
            )
            out = capsys.readouterr().out or result.stdout or ""
            lines = [ln for ln in out.strip().splitlines() if ln.strip()]
            for line in reversed(lines):
                try:
                    json.loads(line)
                    return line
                except json.JSONDecodeError:
                    continue
            raise AssertionError(f"no JSON envelope (rc={result.exit_code}, exc={result.exception})")

        default_line = run_once([])
        full_line = run_once(["--ack", "full"])
        assert default_line == full_line

        data = json.loads(default_line)["data"]
        # Today's payload, pinned: full ops echo with per-op identity.
        assert set(data) == {"workflow", "count", "ops", "aliases", "base_version", "version", "wrote"}
        assert data["count"] == 3 and len(data["ops"]) == 3
        assert all("op_id" in op and "op" in op for op in data["ops"])
        assert data["version"] == 3


class TestAckSummaryPartialFailure:
    def test_ack_summary_partial_failure_reports_index_and_code(self, patched_graph, tmp_path, capsys):
        """Op 2 of 3 (0-based index 1) fails → same code/atomicity as full
        mode, plus a structured receipt: failed.{index,op,code} + applied_count.

        `applied_count` counts specs applied before the abort; the batch is
        atomic, so all of them were then discarded (nothing was written).
        """
        path = _empty(tmp_path)
        before = path.read_text()
        specs = [
            {"op": "add_node", "class_type": "KSampler", "as": "ks"},
            {"op": "add_node", "class_type": "NoSuchNode"},  # fails
            {"op": "set_widget", "node": "ks", "widget": "steps", "value": 30},
        ]
        ops_path = _ops_file(tmp_path, specs)
        env = _run(["apply", str(path), "--ops", str(ops_path), "--ack", "summary"], capsys)
        assert env["ok"] is False
        # Outcome identical to full mode: same code, nothing written.
        assert env["error"]["code"] == "workflow_edit_invalid"
        assert path.read_text() == before
        details = env["error"]["details"]
        assert details["failed"] == {"index": 1, "op": "add_node", "code": "workflow_edit_invalid"}
        assert details["applied_count"] == 1

    def test_full_mode_failure_envelope_unchanged(self, patched_graph, tmp_path, capsys):
        """Default mode keeps today's failure envelope: no details block."""
        path = _empty(tmp_path)
        specs = [
            {"op": "add_node", "class_type": "KSampler", "as": "ks"},
            {"op": "add_node", "class_type": "NoSuchNode"},
        ]
        ops_path = _ops_file(tmp_path, specs)
        env = _run(["apply", str(path), "--ops", str(ops_path)], capsys)
        assert env["ok"] is False
        assert env["error"]["code"] == "workflow_edit_invalid"
        assert env["error"]["details"] is None


class TestSingleEditVerbsUnaffected:
    def test_single_edit_verbs_unaffected(self, patched_graph, tmp_path, capsys):
        path = _write(tmp_path, _base_workflow())
        env = _run(["add-node", str(path), "VAEDecode"], capsys)
        assert env["ok"] is True, env
        # Today's single-edit payload shape, untouched by the ack work.
        assert set(env["data"]) == {"workflow", "op", "base_version", "version", "wrote"}
        assert "ops_by_kind" not in env["data"]

    def test_single_edit_verbs_do_not_take_ack(self, patched_graph, tmp_path, capsys):
        path = _write(tmp_path, _base_workflow())
        _force_json_renderer()
        runner = CliRunner()
        result = runner.invoke(
            workflow_cmd.app, ["add-node", str(path), "VAEDecode", "--ack", "summary"], standalone_mode=False
        )
        assert result.exit_code != 0 or result.exception is not None


class TestAckSummaryPretty:
    def test_pretty_summary_renders_counts_and_aliases(self, patched_graph, tmp_path, capsys):
        from comfy_cli.caller import Caller
        from comfy_cli.output.renderer import OutputMode, Renderer, set_renderer

        r = Renderer.resolve(
            is_stdout_tty=True,
            env={},
            caller=Caller(kind="user", agentic=False, source_env=None),
        )
        r.mode = OutputMode.PRETTY
        set_renderer(r)
        path = _empty(tmp_path)
        ops_path = _ops_file(tmp_path, _three_op_specs())
        runner = CliRunner()
        result = runner.invoke(
            workflow_cmd.app, ["apply", str(path), "--ops", str(ops_path), "--ack", "summary"], standalone_mode=False
        )
        out = capsys.readouterr().out + (result.stdout or "")
        assert "applied 3 edit(s)" in out
        assert "add_node" in out and "2" in out  # per-kind count line
        assert "ckpt" in out and "pos" in out  # aliases rendered
