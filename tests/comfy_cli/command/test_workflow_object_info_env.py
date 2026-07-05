"""COMFY_OBJECT_INFO_FILE lets a host set the offline node catalog once (env),
instead of threading ``--input <path>`` through every CQL command."""

from __future__ import annotations

import comfy_cli.command.workflow as wf


def _patch_graph_load(monkeypatch):
    """Capture the input_path _get_graph resolves, without a real Graph/network."""
    captured: dict[str, str | None] = {}

    import comfy_cli.cql.engine as engine

    class FakeGraph:
        @staticmethod
        def load(input_path, host, port):
            captured["input_path"] = input_path
            return "graph"

        @staticmethod
        def from_object_info(_raw):  # pragma: no cover - live path not exercised here
            raise AssertionError("live fetch should not run when a catalog file is resolved")

    monkeypatch.setattr(engine, "Graph", FakeGraph)
    return captured


def test_get_graph_defaults_to_object_info_file_env(monkeypatch, tmp_path):
    dump = tmp_path / "object_info.json"
    dump.write_text("{}")
    captured = _patch_graph_load(monkeypatch)
    monkeypatch.setenv("COMFY_OBJECT_INFO_FILE", str(dump))

    result = wf._get_graph(None, None, None)

    assert result == "graph"
    assert captured["input_path"] == str(dump), "no --input => COMFY_OBJECT_INFO_FILE is used"


def test_explicit_input_wins_over_env(monkeypatch, tmp_path):
    dump = tmp_path / "object_info.json"
    dump.write_text("{}")
    explicit = tmp_path / "explicit.json"
    explicit.write_text("{}")
    captured = _patch_graph_load(monkeypatch)
    monkeypatch.setenv("COMFY_OBJECT_INFO_FILE", str(dump))

    wf._get_graph(str(explicit), None, None)

    assert captured["input_path"] == str(explicit), "explicit --input overrides the env default"


def test_no_input_no_env_falls_through_to_live_fetch(monkeypatch, tmp_path):
    # With neither --input nor the env var, input_path stays None and the live
    # (Graph.from_object_info) path runs — asserted by FakeGraph.load NOT being called.
    captured = _patch_graph_load(monkeypatch)
    monkeypatch.delenv("COMFY_OBJECT_INFO_FILE", raising=False)

    # from_object_info raises in the fake, so a live attempt surfaces as that error,
    # never as a Graph.load("...") call.
    try:
        wf._get_graph(None, None, None)
    except Exception:
        pass
    assert "input_path" not in captured, "empty env => no offline load, falls through to live fetch"
