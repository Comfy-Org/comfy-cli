"""Tests for ``comfy nodes path --emit-ops`` (V1-019 / BE-7152).

``nodes path`` reports a plan (a routed node sequence); agents then hand-build
the apply batch from it. ``--emit-ops`` closes that gap: each returned path
gains ``ops`` — a ready-to-apply spec batch in the frozen edit vocabulary
(``add_node`` with ``as:`` aliases + ``connect`` referencing those aliases as
bare names).

THE CONTRACT IS THE ROUND-TRIP: the emitted specs must pass
``workflow_ops.apply_specs`` unchanged — the tests feed them to ``apply_specs``
on an empty workflow and assert the nodes and links exist.
"""

from __future__ import annotations

import json
from typing import Any

import pytest
from typer.testing import CliRunner

from comfy_cli import workflow_ops
from comfy_cli.caller import Caller
from comfy_cli.command import nodes as nodes_cmd
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
    """A minimal MODEL → LATENT → IMAGE chain so `path MODEL IMAGE --exact`
    finds exactly one two-step path."""
    return {
        "TinySampler": {
            "input": {"required": {"model": ["MODEL"]}},
            "input_order": {"required": ["model"]},
            "output": ["LATENT"],
            "output_name": ["LATENT"],
            "category": "sampling",
            "display_name": "Tiny Sampler",
            "python_module": "nodes",
        },
        "TinyDecode": {
            "input": {"required": {"samples": ["LATENT"]}},
            "input_order": {"required": ["samples"]},
            "output": ["IMAGE"],
            "output_name": ["IMAGE"],
            "category": "latent",
            "display_name": "Tiny Decode",
            "python_module": "nodes",
        },
    }


def _graph() -> Graph:
    return Graph.from_object_info(_object_info())


@pytest.fixture
def patched_loader(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(nodes_cmd, "_get_graph", lambda *a, **kw: _graph())


def _run(args: list[str], capsys) -> dict[str, Any]:
    _force_json_renderer()
    runner = CliRunner()
    result = runner.invoke(nodes_cmd.app, args, standalone_mode=False)
    captured = capsys.readouterr().out
    if not captured.strip():
        captured = result.stdout or ""
    for line in reversed(captured.strip().splitlines()):
        try:
            return json.loads(line)
        except json.JSONDecodeError:
            continue
    raise AssertionError(f"no JSON envelope (rc={result.exit_code}, exc={result.exception}, out={captured[:600]})")


class TestEmitOps:
    def test_round_trip_through_apply_specs(self, patched_loader, capsys):
        """The emitted specs ARE the contract: applied unchanged onto an empty
        workflow they materialize the path's nodes and intra-path links."""
        env = _run(["path", "MODEL", "IMAGE", "--emit-ops"], capsys)
        assert env["ok"] is True
        paths = env["data"]["paths"]
        assert paths, env["data"]
        specs = paths[0]["ops"]

        # Shape: frozen vocabulary only, bare alias references.
        kinds = [s["op"] for s in specs]
        assert kinds == ["add_node", "add_node", "connect"]
        adds = [s for s in specs if s["op"] == "add_node"]
        assert [a["class_type"] for a in adds] == ["TinySampler", "TinyDecode"]
        assert all(a.get("as") for a in adds)
        connect = next(s for s in specs if s["op"] == "connect")
        assert connect["from"] == f"{adds[0]['as']}.LATENT"
        assert connect["to"] == f"{adds[1]['as']}.samples"

        # Round-trip: apply_specs on an empty workflow, unchanged.
        wf: dict = {"nodes": [], "links": [], "last_node_id": 0, "last_link_id": 0}
        wf, ops, aliases = workflow_ops.apply_specs(wf, _graph(), specs)
        assert {n["type"] for n in wf["nodes"]} == {"TinySampler", "TinyDecode"}
        assert len(wf["links"]) == 1
        decode = next(n for n in wf["nodes"] if n["type"] == "TinyDecode")
        samples = next(i for i in decode["inputs"] if i["name"] == "samples")
        assert samples["link"] == wf["links"][0][0]
        # The link's source really is the sampler minted by the batch.
        sampler_id = aliases[adds[0]["as"]]
        assert wf["links"][0][1] == sampler_id

    def test_loose_mode_also_emits_ops(self, patched_loader, capsys):
        env = _run(["path", "MODEL", "IMAGE", "--loose", "--emit-ops"], capsys)
        assert env["ok"] is True
        for p in env["data"]["paths"]:
            assert isinstance(p["ops"], list) and p["ops"]

    def test_without_flag_byte_identical(self, patched_loader, capsys):
        base = _run(["path", "MODEL", "IMAGE"], capsys)
        again = _run(["path", "MODEL", "IMAGE"], capsys)
        assert again["data"] == base["data"]
        for p in base["data"]["paths"]:
            assert "ops" not in p
        assert "emit_ops" not in base["data"]

    def test_alias_uniqueness_is_deterministic(self):
        """Two steps with the same class dedup deterministically (slug, slug_2)
        and later connects reference the deduped alias, not the clobbered one."""
        steps = [
            {"node": "TinySampler", "input_type": "MODEL", "output_type": "LATENT"},
            {"node": "TinyDecode", "input_type": "LATENT", "output_type": "IMAGE"},
            {"node": "TinyDecode", "input_type": "LATENT", "output_type": "IMAGE"},
        ]
        specs = nodes_cmd._emit_path_ops(_graph(), steps)
        aliases = [s["as"] for s in specs if s["op"] == "add_node"]
        assert aliases == ["tinysampler", "tinydecode", "tinydecode_2"]
        assert len(set(aliases)) == len(aliases)
        connects = [s for s in specs if s["op"] == "connect"]
        # Each decode is fed by the nearest prior LATENT producer (the sampler).
        assert {c["to"] for c in connects} == {"tinydecode.samples", "tinydecode_2.samples"}
        # Deterministic: same input → same output.
        assert specs == nodes_cmd._emit_path_ops(_graph(), steps)

    def test_seed_type_has_no_phantom_producer(self, patched_loader, capsys):
        """The path's FROM type is an unbound input by design (nothing in the
        path produces it) — no connect spec may reference a nonexistent source."""
        env = _run(["path", "MODEL", "IMAGE", "--emit-ops"], capsys)
        specs = env["data"]["paths"][0]["ops"]
        aliases = {s["as"] for s in specs if s["op"] == "add_node"}
        for c in (s for s in specs if s["op"] == "connect"):
            assert c["from"].partition(".")[0] in aliases
            assert c["to"].partition(".")[0] in aliases
