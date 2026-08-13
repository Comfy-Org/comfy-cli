"""Tests for ``comfy nodes search --expand-top N`` (V1-017 / BE-7150).

The measured agent loop is search → show × N (the show args are ~92% a copy of
the search hit). ``--expand-top N`` folds the show payload for the top-N hits
into the search envelope so the follow-up ``show`` calls disappear.

Contract under test:
  * ``--expand-top N`` attaches ``data.expanded`` — one entry per expanded hit,
    carrying ``class_type`` plus the exact ``nodes show`` field vocabulary
    (``inputs`` with options/defaults/choices, ``outputs``, …).
  * ``--expand-top 0`` / flag omitted → payload byte-identical to today.
  * a per-hit catalog miss degrades to a per-hit error entry (code
    ``expand_miss``) and never fails the search.
"""

from __future__ import annotations

import json
from typing import Any

import pytest
from typer.testing import CliRunner

from comfy_cli.caller import Caller
from comfy_cli.command import nodes as nodes_cmd
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
        "KSampler": {
            "input": {
                "required": {
                    "model": ["MODEL"],
                    "steps": ["INT", {"default": 20, "min": 1, "max": 10000}],
                    "sampler_name": [["euler", "heun", "dpmpp_2m"]],
                },
            },
            "input_order": {"required": ["model", "steps", "sampler_name"]},
            "output": ["LATENT"],
            "output_name": ["LATENT"],
            "category": "sampling",
            "display_name": "KSampler",
            "description": "Denoise the latent via the provided model.",
            "output_node": False,
            "python_module": "nodes",
        },
        "KSamplerAdvanced": {
            "input": {"required": {"model": ["MODEL"]}},
            "input_order": {"required": ["model"]},
            "output": ["LATENT"],
            "output_name": ["LATENT"],
            "category": "sampling",
            "display_name": "KSampler (Advanced)",
            "description": "Denoise the latent with extra knobs.",
            "output_node": False,
            "python_module": "nodes",
        },
        "VAEDecode": {
            "input": {"required": {"samples": ["LATENT"], "vae": ["VAE"]}},
            "input_order": {"required": ["samples", "vae"]},
            "output": ["IMAGE"],
            "output_name": ["IMAGE"],
            "category": "latent",
            "display_name": "VAE Decode",
            "description": "Turn a latent back into pixels.",
            "output_node": False,
            "python_module": "nodes",
        },
    }


def _graph():
    from comfy_cli.cql.engine import Graph

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


class TestExpandTop:
    def test_expand_returns_top_hit_schema(self, patched_loader, capsys):
        """--expand-top 1 folds the full `nodes show` payload for the top hit
        into the search envelope: inputs with defaults + enum choices, outputs."""
        env = _run(["search", "KSampler", "--expand-top", "1"], capsys)
        assert env["ok"] is True
        data = env["data"]
        # Ranking unchanged: exact name is the top hit.
        assert data["rows"][0]["name"] == "KSampler"
        expanded = data["expanded"]
        assert len(expanded) == 1
        entry = expanded[0]
        assert entry["class_type"] == "KSampler"
        # `nodes show` field vocabulary, verbatim (morphism_to_dict).
        inputs = {i["name"]: i for i in entry["inputs"]}
        assert inputs["steps"]["options"]["default"] == 20
        assert inputs["steps"]["options"]["min"] == 1
        assert inputs["sampler_name"]["choices"] == ["euler", "heun", "dpmpp_2m"]
        assert inputs["model"]["is_link"] is True
        assert entry["outputs"] == [{"name": "LATENT", "type": "LATENT"}]
        assert entry["output_types"] == ["LATENT"]

    def test_expand_covers_top_n_in_rank_order(self, patched_loader, capsys):
        env = _run(["search", "sampler", "--expand-top", "2"], capsys)
        assert env["ok"] is True
        expanded = env["data"]["expanded"]
        assert [e["class_type"] for e in expanded] == [r["name"] for r in env["data"]["rows"][:2]]
        assert len(expanded) == 2

    def test_omitted_and_zero_are_byte_identical_to_baseline(self, patched_loader, capsys):
        base = _run(["search", "KSampler"], capsys)
        zero = _run(["search", "KSampler", "--expand-top", "0"], capsys)
        assert zero["data"] == base["data"]
        assert "expanded" not in base["data"]
        assert "expanded" not in zero["data"]

    def test_zero_matches_baseline_yields_empty_expanded(self, patched_loader, capsys):
        """A query with no hits (and no close-name fallback) still succeeds and
        carries an empty `expanded` — never an error."""
        env = _run(["search", "xyzzy_zzq_nothing", "--expand-top", "3"], capsys)
        assert env["ok"] is True
        assert env["data"]["total"] == 0
        assert env["data"]["rows"] == []
        assert env["data"]["expanded"] == []

    def test_per_hit_catalog_miss_degrades_to_error_entry(self, monkeypatch, capsys):
        """A hit that can't be re-resolved through the show path yields a per-hit
        `expand_miss` error entry; the search itself still succeeds and the other
        hits still expand."""
        graph = _graph()
        real_node = graph.node

        def flaky_node(name: str):
            if name == "KSampler":
                return None  # simulate a catalog miss for this one hit
            return real_node(name)

        monkeypatch.setattr(graph, "node", flaky_node)
        monkeypatch.setattr(nodes_cmd, "_get_graph", lambda *a, **kw: graph)

        env = _run(["search", "sampler", "--expand-top", "2"], capsys)
        assert env["ok"] is True
        expanded = env["data"]["expanded"]
        assert len(expanded) == 2
        by_class = {e["class_type"]: e for e in expanded}
        miss = by_class["KSampler"]
        assert miss["error"]["code"] == "expand_miss"
        assert "inputs" not in miss
        hit = by_class["KSamplerAdvanced"]
        assert "error" not in hit
        assert {i["name"] for i in hit["inputs"]} == {"model"}

    def test_expand_applies_to_close_match_fallback_rows(self, patched_loader, capsys):
        """Typo queries fall back to close-name matches; those rows are real
        catalog nodes and expand the same way."""
        env = _run(["search", "KSampeler", "--expand-top", "1"], capsys)
        assert env["ok"] is True
        assert env["data"]["close_match"] is True
        expanded = env["data"]["expanded"]
        assert len(expanded) == 1
        assert expanded[0]["class_type"] == env["data"]["rows"][0]["name"]
        assert "inputs" in expanded[0]

    def test_expand_miss_is_registered(self):
        from comfy_cli import error_codes

        assert error_codes.is_registered("expand_miss")
