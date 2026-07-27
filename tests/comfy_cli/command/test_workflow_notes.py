"""Tests for `comfy workflow notes` — reading Note/MarkdownNote text out of a workflow.

Purely offline: no Graph / object_info is involved, so unlike the slots tests
there is no `patched_graph` fixture here.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from comfy_cli.caller import Caller
from comfy_cli.command import workflow as workflow_cmd
from comfy_cli.output.renderer import (
    OutputMode,
    Renderer,
    reset_renderer_for_testing,
    set_renderer,
)


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


def _force_pretty_renderer():
    r = Renderer.resolve(
        is_stdout_tty=True,
        env={},
        caller=Caller(kind="user", agentic=False, source_env=None),
        json_flag=False,
    )
    r.mode = OutputMode.PRETTY
    set_renderer(r)
    return r


def _write_workflow(tmp_path: Path, data: dict, name: str = "test.json") -> Path:
    p = tmp_path / name
    p.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return p


def _notes_workflow():
    """Frontend-format workflow: a MarkdownNote (titled), a Note (untitled), a real node."""
    return {
        "last_node_id": 12,
        "last_link_id": 3,
        "nodes": [
            {
                "id": 5,
                "type": "MarkdownNote",
                "title": "Trigger words",
                "pos": [100, 200],
                "size": [400, 180],
                "flags": {},
                "order": 0,
                "mode": 0,
                "widgets_values": ["Use **ohwx man** to fire the LoRA."],
            },
            {
                "id": 6,
                "type": "CheckpointLoaderSimple",
                "pos": [10, 10],
                "size": [315, 98],
                "widgets_values": ["sd_xl_base_1.0.safetensors"],
            },
            {
                "id": 7,
                "type": "Note",
                "title": None,
                "pos": [600, 200],
                "size": [300, 120],
                "widgets_values": ["Set steps to 30 for the final render."],
            },
        ],
        "links": [],
    }


def _run(args: list[str], capsys) -> dict[str, Any]:
    _force_json_renderer()
    runner = CliRunner()
    result = runner.invoke(workflow_cmd.app, args, standalone_mode=False)
    captured = capsys.readouterr().out
    if not captured.strip():
        captured = result.stdout or ""
    lines = [ln for ln in captured.strip().splitlines() if ln.strip()]
    for line in reversed(lines):
        try:
            return json.loads(line)
        except json.JSONDecodeError:
            continue
    raise AssertionError(f"no JSON envelope (rc={result.exit_code}, exc={result.exception}, out={captured[:500]})")


def _invoke(args: list[str]):
    """Invoke in standalone mode so ``typer.Exit`` shows up as a real exit code.

    ``_run`` deliberately uses ``standalone_mode=False`` (matching the slots
    tests) to inspect the envelope; that path swallows the exit status, so
    exit-code assertions go through here instead. Pretty-mode output lands in
    ``result.output`` because Rich's Console binds to CliRunner's redirected
    stdout, not to capsys.
    """
    return CliRunner().invoke(workflow_cmd.app, args)


# ---------------------------------------------------------------------------
# happy path
# ---------------------------------------------------------------------------


class TestNotes:
    def test_lists_only_note_nodes_with_full_fields(self, tmp_path, capsys):
        path = _write_workflow(tmp_path, _notes_workflow())
        env = _run(["notes", str(path)], capsys)

        assert env["ok"] is True
        data = env["data"]
        assert data["workflow"] == str(path)
        assert data["count"] == 2
        assert len(data["notes"]) == 2

        md, plain = data["notes"]
        assert md == {
            "id": 5,
            "type": "MarkdownNote",
            "title": "Trigger words",
            "text": "Use **ohwx man** to fire the LoRA.",
            "pos": [100, 200],
            "size": [400, 180],
            "subgraph": None,
        }
        assert plain == {
            "id": 7,
            "type": "Note",
            "title": None,
            "text": "Set steps to 30 for the final render.",
            "pos": [600, 200],
            "size": [300, 120],
            "subgraph": None,
        }

    def test_ignores_other_ui_only_node_types(self, tmp_path, capsys):
        """PrimitiveNode/Reroute/GetNode/SetNode carry no documentation — not notes."""
        wf = {
            "nodes": [
                {"id": 1, "type": "PrimitiveNode", "widgets_values": ["not a note"]},
                {"id": 2, "type": "Reroute", "widgets_values": ["nope"]},
                {"id": 3, "type": "GetNode", "widgets_values": ["nope"]},
                {"id": 4, "type": "SetNode", "widgets_values": ["nope"]},
            ],
            "links": [],
        }
        env = _run(["notes", str(_write_workflow(tmp_path, wf))], capsys)
        assert env["ok"] is True
        assert env["data"]["count"] == 0

    def test_does_not_read_group_titles(self, tmp_path, capsys):
        wf = {"nodes": [], "links": [], "groups": [{"title": "Sampling", "bounding": [0, 0, 10, 10]}]}
        env = _run(["notes", str(_write_workflow(tmp_path, wf))], capsys)
        assert env["ok"] is True
        assert env["data"] == {"workflow": str(tmp_path / "test.json"), "count": 0, "notes": []}

    def test_never_writes_the_file(self, tmp_path, capsys):
        path = _write_workflow(tmp_path, _notes_workflow())
        before = path.read_text(encoding="utf-8")
        env = _run(["notes", str(path)], capsys)
        assert env["ok"] is True
        assert path.read_text(encoding="utf-8") == before


# ---------------------------------------------------------------------------
# subgraphs
# ---------------------------------------------------------------------------


class TestNotesInSubgraphs:
    def test_note_inside_subgraph_definition_is_surfaced(self, tmp_path, capsys):
        wf = {
            "nodes": [{"id": 1, "type": "KSampler", "widgets_values": [42]}],
            "links": [],
            "definitions": {
                "subgraphs": [
                    {
                        "id": "b6a1e0c2-0000-4000-8000-000000000001",
                        "name": "Upscale pass",
                        "nodes": [
                            {
                                "id": 3,
                                "type": "Note",
                                "title": "Inner note",
                                "pos": [0, 0],
                                "size": [200, 100],
                                "widgets_values": ["runs at 2x"],
                            }
                        ],
                    }
                ]
            },
        }
        env = _run(["notes", str(_write_workflow(tmp_path, wf))], capsys)
        assert env["ok"] is True
        assert env["data"]["count"] == 1
        note = env["data"]["notes"][0]
        assert note["text"] == "runs at 2x"
        assert note["subgraph"] == {
            "id": "b6a1e0c2-0000-4000-8000-000000000001",
            "name": "Upscale pass",
        }

    def test_top_level_and_subgraph_notes_both_listed(self, tmp_path, capsys):
        wf = _notes_workflow()
        wf["definitions"] = {
            "subgraphs": [
                {
                    "id": "sg-1",
                    "name": "Detailer",
                    "nodes": [{"id": 20, "type": "MarkdownNote", "widgets_values": ["subgraph doc"]}],
                }
            ]
        }
        env = _run(["notes", str(_write_workflow(tmp_path, wf))], capsys)
        assert env["data"]["count"] == 3
        assert [n["subgraph"] is None for n in env["data"]["notes"]] == [True, True, False]
        assert env["data"]["notes"][-1]["subgraph"] == {"id": "sg-1", "name": "Detailer"}


# ---------------------------------------------------------------------------
# malformed / missing note payloads
# ---------------------------------------------------------------------------


class TestNotesTolerateMalformedPayloads:
    @pytest.mark.parametrize(
        "node",
        [
            pytest.param({"id": 1, "type": "Note"}, id="missing-widgets_values"),
            pytest.param({"id": 1, "type": "Note", "widgets_values": []}, id="empty-widgets_values"),
            pytest.param({"id": 1, "type": "Note", "widgets_values": None}, id="null-widgets_values"),
            pytest.param({"id": 1, "type": "Note", "widgets_values": {"0": "x"}}, id="dict-widgets_values"),
            pytest.param({"id": 1, "type": "Note", "widgets_values": [None]}, id="null-first-widget"),
            pytest.param({"id": 1, "type": "Note", "widgets_values": [123]}, id="non-string-first-widget"),
        ],
    )
    def test_note_without_usable_text_yields_empty_string(self, tmp_path, capsys, node):
        env = _run(["notes", str(_write_workflow(tmp_path, {"nodes": [node], "links": []}))], capsys)
        assert env["ok"] is True
        assert env["data"]["count"] == 1
        assert env["data"]["notes"][0]["text"] == ""

    def test_non_dict_entries_in_nodes_are_skipped(self, tmp_path, capsys):
        wf = {"nodes": ["junk", None, 7, {"id": 1, "type": "Note", "widgets_values": ["ok"]}], "links": []}
        env = _run(["notes", str(_write_workflow(tmp_path, wf))], capsys)
        assert env["ok"] is True
        assert env["data"]["count"] == 1
        assert env["data"]["notes"][0]["text"] == "ok"

    def test_malformed_definitions_block_is_ignored(self, tmp_path, capsys):
        wf = {"nodes": [], "links": [], "definitions": {"subgraphs": [None, "junk", {"nodes": None}]}}
        env = _run(["notes", str(_write_workflow(tmp_path, wf))], capsys)
        assert env["ok"] is True
        assert env["data"]["count"] == 0


# ---------------------------------------------------------------------------
# empty + error envelopes
# ---------------------------------------------------------------------------


class TestNotesEnvelopes:
    def test_workflow_without_notes(self, tmp_path, capsys):
        wf = {"nodes": [{"id": 1, "type": "KSampler", "widgets_values": [42]}], "links": []}
        path = _write_workflow(tmp_path, wf)
        _force_json_renderer()
        assert _invoke(["notes", str(path)]).exit_code == 0
        env = _run(["notes", str(path)], capsys)
        assert env["ok"] is True
        assert env["data"]["count"] == 0
        assert env["data"]["notes"] == []

    def test_rejects_api_format(self, tmp_path, capsys):
        api_wf = {"3": {"class_type": "KSampler", "inputs": {}}}
        path = _write_workflow(tmp_path, api_wf)
        _force_json_renderer()
        assert _invoke(["notes", str(path)]).exit_code == 1
        env = _run(["notes", str(path)], capsys)
        assert env["ok"] is False
        assert env["error"]["code"] == "workflow_not_frontend_format"

    def test_rejects_missing_file(self, tmp_path, capsys):
        missing = str(tmp_path / "nope.json")
        _force_json_renderer()
        assert _invoke(["notes", missing]).exit_code == 1
        env = _run(["notes", missing], capsys)
        assert env["ok"] is False
        assert env["error"]["code"] == "workflow_not_found"

    def test_rejects_invalid_json(self, tmp_path, capsys):
        path = tmp_path / "broken.json"
        path.write_text("{not json", encoding="utf-8")
        env = _run(["notes", str(path)], capsys)
        assert env["ok"] is False
        assert env["error"]["code"] == "workflow_invalid_json"


# ---------------------------------------------------------------------------
# pretty mode
# ---------------------------------------------------------------------------


class TestNotesPrettyMode:
    def test_pretty_renders_title_id_and_text(self, tmp_path, capsys):
        path = _write_workflow(tmp_path, _notes_workflow())
        _force_pretty_renderer()
        result = _invoke(["notes", str(path)])
        out = result.output
        assert result.exit_code == 0
        assert "Trigger words" in out
        assert "#5" in out
        assert "ohwx man" in out
        # Untitled note falls back to its type as the heading.
        assert "Note" in out
        assert "#7" in out
        assert "Set steps to 30" in out
        # Pretty mode must not also dump the JSON envelope.
        assert '"envelope"' not in out

    def test_pretty_reports_empty_workflow(self, tmp_path, capsys):
        path = _write_workflow(tmp_path, {"nodes": [], "links": []})
        _force_pretty_renderer()
        result = _invoke(["notes", str(path)])
        out = result.output
        assert result.exit_code == 0
        assert "No notes in this workflow." in out

    def test_pretty_neutralizes_markup_and_control_chars(self, tmp_path, capsys):
        """Note text is untrusted file content — it must not style or drive the terminal."""
        wf = {
            "nodes": [
                {
                    "id": 1,
                    "type": "Note",
                    "title": "[red]spoofed[/red]",
                    "widgets_values": ["danger \x1b[31mred\x1b[0m and [bold]markup[/bold]"],
                }
            ],
            "links": [],
        }
        path = _write_workflow(tmp_path, wf)
        _force_pretty_renderer()
        result = _invoke(["notes", str(path)])
        out = result.output
        assert result.exit_code == 0
        assert "\x1b[31m" not in out
        assert "[bold]markup[/bold]" in out
        assert "[red]spoofed[/red]" in out

    def test_pretty_survives_subgraph_note(self, tmp_path, capsys):
        wf = {
            "nodes": [],
            "links": [],
            "definitions": {
                "subgraphs": [
                    {"id": "sg-9", "name": None, "nodes": [{"id": 2, "type": "Note", "widgets_values": ["inner"]}]}
                ]
            },
        }
        path = _write_workflow(tmp_path, wf)
        _force_pretty_renderer()
        result = _invoke(["notes", str(path)])
        out = result.output
        assert result.exit_code == 0
        # name is null, so the id is the fallback label.
        assert "sg-9" in out
        assert "inner" in out


# ---------------------------------------------------------------------------
# _extract_notes unit coverage (no CLI)
# ---------------------------------------------------------------------------


def test_notes_payload_validates_against_the_registered_schema(tmp_path, capsys):
    """`comfy workflow notes` is registered in COMMAND_SCHEMAS -> workflow.json."""
    import jsonschema

    from comfy_cli.discovery import COMMAND_SCHEMAS

    assert COMMAND_SCHEMAS["comfy workflow notes"] == "workflow"
    schema_path = Path(workflow_cmd.__file__).parent.parent / "schemas" / "workflow.json"
    schema = json.loads(schema_path.read_text(encoding="utf-8"))

    wf = _notes_workflow()
    wf["definitions"] = {
        "subgraphs": [{"id": "sg-1", "name": "Detailer", "nodes": [{"id": 20, "type": "Note", "widgets_values": []}]}]
    }
    env = _run(["notes", str(_write_workflow(tmp_path, wf))], capsys)
    jsonschema.Draft202012Validator(schema).validate(env["data"])


def test_extract_notes_handles_missing_nodes_key():
    assert workflow_cmd._extract_notes({}) == []
    assert workflow_cmd._extract_notes({"nodes": None}) == []
    assert workflow_cmd._extract_notes({"nodes": {}}) == []
    assert workflow_cmd._extract_notes({"definitions": "junk"}) == []
    assert workflow_cmd._extract_notes({"definitions": {"subgraphs": None}}) == []
