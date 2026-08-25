"""Tests for `comfy workflow print` — rendering a workflow as Python-like source.

The pure printer (`comfy_cli.workflow_print`) has its own exhaustive test suite
in `tests/comfy_cli/test_workflow_print.py`. This file only covers the CLI
wiring: envelope shape, `--select`, pretty-mode verbatim source, the
frontend-format guard, and the `workflow_print_unsupported` error path.
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

FIXTURES = Path(__file__).parent.parent / "fixtures"
SD15 = FIXTURES / "sd15_ui_workflow.json"
SD15_OI = FIXTURES / "sd15_object_info.json"


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


def _run(args: list[str], capsys, expect_ok: bool = True) -> dict[str, Any]:
    _force_json_renderer()
    runner = CliRunner()
    result = runner.invoke(workflow_cmd.app, args, standalone_mode=False)
    captured = capsys.readouterr().out
    if not captured.strip():
        captured = result.stdout or ""
    lines = [ln for ln in captured.strip().splitlines() if ln.strip()]
    for line in reversed(lines):
        try:
            env = json.loads(line)
        except json.JSONDecodeError:
            continue
        if expect_ok:
            assert env.get("ok") is True, env
        return env
    raise AssertionError(f"no JSON envelope (rc={result.exit_code}, exc={result.exception}, out={captured[:500]})")


def test_print_json_envelope(capsys):
    env = _run(["print", str(SD15), "--input", str(SD15_OI)], capsys)
    assert env["ok"] is True and env["command"] == "workflow print"
    d = env["data"]
    assert d["format"] == "py" and d["node_count"] == 7
    assert (
        d["source"].splitlines()[0]
        == 'checkpoint_loader_simple = CheckpointLoaderSimple(ckpt_name="v1-5-pruned-emaonly-fp16.safetensors")  # 4'
    )
    assert d["bindings"]["ksampler"] == "3"
    assert d["workflow"].endswith("sd15_ui_workflow.json")


def test_print_select_projects(capsys):
    env = _run(["print", str(SD15), "--input", str(SD15_OI), "--select", "bindings"], capsys)
    assert env["data"] == {
        "checkpoint_loader_simple": "4",
        "empty_latent_image": "5",
        "clip_text_encode": "6",
        "clip_text_encode_2": "7",
        "ksampler": "3",
        "vae_decode": "8",
        "save_image": "9",
    }


def test_print_pretty_prints_source_verbatim(capsys):
    _force_pretty_renderer()
    result = CliRunner().invoke(workflow_cmd.app, ["print", str(SD15), "--input", str(SD15_OI)])
    assert result.exit_code == 0, result.output
    assert "Node[" not in result.output  # sanity: no rich-markup mangling of brackets
    assert "ksampler = KSampler(" in result.output and "  # 3" in result.output


def test_print_select_in_pretty_mode_prints_only_the_selection():
    _force_pretty_renderer()
    result = CliRunner().invoke(workflow_cmd.app, ["print", str(SD15), "--input", str(SD15_OI), "--select", "bindings"])
    assert result.exit_code == 0, result.output
    assert "ksampler = KSampler(" not in result.output
    assert '"ksampler": "3"' in result.output


def test_print_select_source_pretty_mode_strips_terminal_controls(tmp_path):
    # A note's title/text goes straight into `source` as a comment without
    # passing through `json.dumps` escaping, so an escape sequence in it
    # would otherwise reach the terminal raw via `--select source` in pretty
    # mode (the non-select pretty path already strips `res.source` itself;
    # `--select` bypassed that stripping before this fix).
    wf = json.loads(SD15.read_text())
    mutated = False
    for n in wf["nodes"]:
        if n.get("type") == "MarkdownNote":
            n["widgets_values"][0] = "\x1b[31mEVIL\x1b[0m rest of note"
            mutated = True
            break
    assert mutated, "fixture has no MarkdownNote to mutate"
    p = _write_workflow(tmp_path, wf)

    _force_pretty_renderer()
    result = CliRunner().invoke(workflow_cmd.app, ["print", str(p), "--input", str(SD15_OI), "--select", "source"])
    assert result.exit_code == 0, result.output
    assert "\x1b" not in result.output
    assert "EVIL" in result.output


def test_print_rejects_api_format(tmp_path, capsys):
    p = _write_workflow(tmp_path, {"3": {"class_type": "KSampler", "inputs": {}}})
    env = _run(["print", str(p), "--input", str(SD15_OI)], capsys, expect_ok=False)
    assert env["error"]["code"] == "workflow_not_frontend_format"


def test_print_unsupported_lists_reasons(tmp_path, capsys):
    wf = {
        "nodes": [
            {"id": 1, "type": "workflow>Grp", "inputs": [], "outputs": [], "widgets_values": []},
            {
                "id": 2,
                "type": "VAEDecode",
                "inputs": [{"name": "samples", "type": "LATENT", "link": 7}],
                "outputs": [],
                "widgets_values": [],
            },
        ],
        "links": [[7, 99, 0, 2, 0, "LATENT"]],
        "version": 0.4,
    }
    env = _run(["print", str(_write_workflow(tmp_path, wf)), "--input", str(SD15_OI)], capsys, expect_ok=False)
    assert env["error"]["code"] == "workflow_print_unsupported"
    assert env["error"]["details"]["reasons"] == [
        "node 1 is a legacy group node (workflow>Grp)",
        "link 7 references missing node 99",
    ]


def test_print_rejects_unknown_format(capsys):
    env = _run(["print", str(SD15), "--input", str(SD15_OI), "--format", "json"], capsys, expect_ok=False)
    assert env["error"]["code"] == "workflow_print_unsupported"
    assert env["error"]["details"]["reasons"] == ["format 'json'"]


def test_print_payload_validates_against_schema(capsys):
    import jsonschema

    from comfy_cli.discovery import COMMAND_SCHEMAS

    assert COMMAND_SCHEMAS["comfy workflow print"] == "workflow"
    schema = json.loads((Path(workflow_cmd.__file__).parent.parent / "schemas" / "workflow.json").read_text())
    env = _run(["print", str(SD15), "--input", str(SD15_OI)], capsys)
    jsonschema.Draft202012Validator(schema).validate(env["data"])


def test_print_help_mentions_only_real_commands():
    from comfy_cli.help_json import HELP_EXAMPLES

    assert "comfy workflow print" in HELP_EXAMPLES
