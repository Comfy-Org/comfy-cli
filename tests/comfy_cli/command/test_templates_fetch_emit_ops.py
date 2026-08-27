"""``comfy templates fetch --emit-ops`` (V1-038).

``templates fetch -o workflow.json`` is a BULK WRITER: it replaces the working
file wholesale. Downstream (the cloud agent's document) that replacement had to
be expressed as a **re-mint** — a brand-new document with no attributed,
incremental history, and §8.6's "one common initial snapshot" rule makes an
independent re-seed the one thing a replica must never do.

``--emit-ops`` closes that: the fetch also emits ``data.ops`` — the stamped op
batch that turns the file it is replacing INTO the template, in the frozen
vocabulary. Two contracts, both tested here:

* **the op contract** (what the cloud forwards): replaying the batch with
  ``apply_op`` reproduces the template's graph exactly — same node types, same
  wiring, same widget values;
* **the spec contract** (what ``nodes path --emit-ops`` already promises): the
  same array is accepted by ``apply_specs`` verbatim, so the batch is a legal
  ``comfy workflow apply --ops`` input.

Templates the frozen vocabulary cannot express (subgraph definitions, groups)
emit NO ops and say why, so the consumer falls back to its re-mint path rather
than silently landing a partial graph.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from comfy_cli import workflow_ops
from comfy_cli.caller import Caller
from comfy_cli.command import templates as templates_cmd
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
            "input": {"required": {"ckpt_name": [["a.safetensors", "b.safetensors"]]}},
            "input_order": {"required": ["ckpt_name"]},
            "output": ["MODEL"],
            "output_name": ["MODEL"],
            "category": "loaders",
            "display_name": "Tiny Loader",
            "python_module": "nodes",
        },
        "TinySink": {
            "input": {"required": {"model": ["MODEL"]}},
            "input_order": {"required": ["model"]},
            "output": [],
            "output_name": [],
            "category": "test",
            "display_name": "Tiny Sink",
            "python_module": "nodes",
        },
    }


def _graph() -> Graph:
    return Graph.from_object_info(_object_info())


# A two-node template in frontend/save format: loader -> sink, one link.
def _template() -> dict[str, Any]:
    return {
        "id": "tpl-1",
        "revision": 0,
        "last_node_id": 2,
        "last_link_id": 1,
        "nodes": [
            {
                "id": 1,
                "type": "TinyLoader",
                "pos": [10, 20],
                "inputs": [],
                "outputs": [{"name": "MODEL", "type": "MODEL", "links": [1]}],
                "widgets_values": ["b.safetensors"],
            },
            {
                "id": 2,
                "type": "TinySink",
                "pos": [300, 20],
                "inputs": [{"name": "model", "type": "MODEL", "link": 1}],
                "outputs": [],
                "widgets_values": [],
            },
        ],
        "links": [[1, 1, 0, 2, 0, "MODEL"]],
        "groups": [],
    }


def _existing() -> dict[str, Any]:
    """A workflow already on the canvas — what the fetch replaces."""
    return {
        "id": "wf-old",
        "nodes": [
            {
                "id": 77,
                "type": "TinyLoader",
                "pos": [0, 0],
                "inputs": [],
                "outputs": [{"name": "MODEL", "type": "MODEL", "links": []}],
                "widgets_values": ["a.safetensors"],
            }
        ],
        "links": [],
        "last_node_id": 77,
        "last_link_id": 0,
    }


_GALLERY_ROW = {
    "name": "tiny_template",
    "title": "Tiny Template",
    "output_type": "image",
    "category": "Basics",
    "tags": [],
    "models": [],
    "providers": [],
}


@pytest.fixture
def patched_fetch(monkeypatch: pytest.MonkeyPatch):
    """Resolve the gallery + the workflow body locally: no network."""
    monkeypatch.setattr(templates_cmd, "_load_gallery", lambda *a, **kw: [{"templates": []}])
    monkeypatch.setattr(templates_cmd, "_flatten_templates", lambda cats: [dict(_GALLERY_ROW)])
    monkeypatch.setattr(
        templates_cmd,
        "_fetch_template_workflow",
        lambda name, **kw: json.dumps(_template()).encode("utf-8"),
    )


def _run(args: list[str], capsys) -> dict[str, Any]:
    _force_json_renderer()
    runner = CliRunner()
    result = runner.invoke(templates_cmd.app, args, standalone_mode=False)
    captured = capsys.readouterr().out
    if not captured.strip():
        captured = result.stdout or ""
    for line in reversed(captured.strip().splitlines()):
        try:
            return json.loads(line)
        except json.JSONDecodeError:
            continue
    raise AssertionError(f"no JSON envelope (rc={result.exit_code}, exc={result.exception}, out={captured[:600]})")


class TestTemplateFetchEmitsOpBatch:
    def test_template_fetch_emits_op_batch(self, tmp_path: Path, patched_fetch, capsys):
        """The batch is stamped, ordered delete→add→connect, and replaying it
        onto the file being replaced reproduces the template exactly."""
        out = tmp_path / "workflow_bulkops.json"
        base = _existing()
        out.write_text(json.dumps(base), encoding="utf-8")

        env = _run(
            ["fetch", "tiny_template", "-o", str(out), "--emit-ops", "--actor", "agent:th_1:7", "--base-version", "4"],
            capsys,
        )

        assert env["ok"] is True, env
        ops = env["data"]["ops"]

        # Every entry is a real, stamped op — the cloud drops anything without
        # an op_id, so an unstamped entry is silently lost, not rejected.
        for op in ops:
            assert len(op["op_id"]) == 32 and op["op_id"].islower()
            assert op["actor"] == "agent:th_1:7"
            assert op["base_version"] == 4
            assert op["stamp"] == [4, "agent:th_1:7"]
        assert len({op["op_id"] for op in ops}) == len(ops)

        # Replace = delete what was there, then build the template.
        assert [op["op"] for op in ops] == ["delete_node", "add_node", "add_node", "connect"]
        assert ops[0]["node_id"] == 77

        # THE OP CONTRACT: replay onto the pre-fetch graph == the template.
        replayed: dict[str, Any] = json.loads(json.dumps(base))
        for op in ops:
            replayed = workflow_ops.apply_op(replayed, op, _graph())
        workflow_ops.strip_internal(replayed)

        assert [n["type"] for n in replayed["nodes"]] == ["TinyLoader", "TinySink"]
        # Widget values ride inside the add_node payload (§8.5), so a fetched
        # template keeps its demo values instead of catalog defaults.
        loader = next(n for n in replayed["nodes"] if n["type"] == "TinyLoader")
        assert loader["widgets_values"] == ["b.safetensors"]
        assert loader["pos"] == [10, 20]
        assert len(replayed["links"]) == 1
        link = replayed["links"][0]
        sink = next(n for n in replayed["nodes"] if n["type"] == "TinySink")
        assert link[1] == loader["id"] and link[3] == sink["id"]
        assert sink["inputs"][0]["link"] == link[0]

        # Identity is minted, never inherited: the template's small counter ids
        # (1, 2) would resurrect ids a concurrent replica may still hold.
        assert all(n["id"] >= 1 << 40 for n in replayed["nodes"])
        assert link[0] >= 1 << 40

        # The file the fetch wrote is still the template itself (unchanged
        # behavior — --emit-ops adds a payload, it does not change the write).
        assert [n["id"] for n in json.loads(out.read_text(encoding="utf-8"))["nodes"]] == [1, 2]

    def test_emitted_batch_round_trips_through_apply_specs(self, tmp_path: Path, patched_fetch, capsys):
        """THE SPEC CONTRACT: the same array is a legal `workflow apply --ops`
        batch — apply_specs accepts it verbatim and rebuilds the structure."""
        out = tmp_path / "workflow_bulkspecs.json"
        base = _existing()
        out.write_text(json.dumps(base), encoding="utf-8")

        env = _run(["fetch", "tiny_template", "-o", str(out), "--emit-ops"], capsys)
        specs = env["data"]["ops"]

        wf, ops, aliases = workflow_ops.apply_specs(json.loads(json.dumps(base)), _graph(), specs)

        assert [n["type"] for n in wf["nodes"]] == ["TinyLoader", "TinySink"]
        assert 77 not in [n["id"] for n in wf["nodes"]]
        assert len(wf["links"]) == 1
        assert wf["links"][0][1] == aliases[specs[1]["as"]]
        assert wf["links"][0][3] == aliases[specs[2]["as"]]

    def test_without_the_flag_the_envelope_is_unchanged(self, tmp_path: Path, patched_fetch, capsys):
        out = tmp_path / "workflow_noops.json"
        out.write_text(json.dumps(_existing()), encoding="utf-8")
        env = _run(["fetch", "tiny_template", "-o", str(out)], capsys)
        assert env["ok"] is True
        assert "ops" not in env["data"]
        assert "ops_skipped" not in env["data"]

    def test_emit_ops_on_a_fresh_canvas_has_no_deletes(self, tmp_path: Path, patched_fetch, capsys):
        out = tmp_path / "does_not_exist_yet.json"
        env = _run(["fetch", "tiny_template", "-o", str(out), "--emit-ops"], capsys)
        assert [op["op"] for op in env["data"]["ops"]] == ["add_node", "add_node", "connect"]

    def test_inexpressible_template_emits_no_ops_and_says_why(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, patched_fetch, capsys
    ):
        """A template the frozen vocabulary cannot express (a subgraph
        definition) must emit NOTHING — a partial batch would land a graph that
        is not the template. The consumer keeps its re-mint fallback for these."""
        tpl = _template()
        tpl["definitions"] = {"subgraphs": [{"id": "sg-1", "nodes": []}]}
        monkeypatch.setattr(
            templates_cmd, "_fetch_template_workflow", lambda name, **kw: json.dumps(tpl).encode("utf-8")
        )
        out = tmp_path / "workflow_subgraph.json"
        env = _run(["fetch", "tiny_template", "-o", str(out), "--emit-ops"], capsys)

        assert env["ok"] is True
        assert "ops" not in env["data"]
        assert "subgraph" in env["data"]["ops_skipped"]

    def test_emit_ops_without_out_still_emits(self, patched_fetch, capsys):
        """Without -o there is no file being replaced, so the batch is a pure
        build — still emitted, so a caller that materializes the envelope's
        workflow itself can use the ops."""
        env = _run(["fetch", "tiny_template", "--emit-ops"], capsys)
        assert [op["op"] for op in env["data"]["ops"]] == ["add_node", "add_node", "connect"]
