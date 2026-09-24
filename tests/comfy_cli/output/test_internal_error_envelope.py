"""An unexpected exception inside a command still ends the --json stream with an envelope.

`comfy --json workflow set-widget ...` could escape with a Python traceback
and no envelope. Edit commands catch only ValueError, and nothing above them
turned any other exception into `ok:false`. So a --json caller got a stack
dump, an empty stdout and exit 1, which it cannot tell from a transport
failure.

These drive the real CLI, because the fix lives at the click entrypoint
(`_RootGroup.invoke`), the same place usage errors become envelopes.
"""

from __future__ import annotations

import json

import pytest
from typer.testing import CliRunner

from comfy_cli import workflow_ops
from comfy_cli.cmdline import app


def _boom(*_a, **_kw):
    raise KeyError("widgets_values")


@pytest.fixture
def workflow_file(tmp_path):
    p = tmp_path / "wf.json"
    p.write_text(json.dumps({"nodes": [{"id": 1, "type": "VAELoader", "widgets_values": ["a"]}], "links": []}))
    return p


@pytest.fixture
def object_info(tmp_path):
    oi = tmp_path / "oi.json"
    oi.write_text(
        json.dumps(
            {
                "VAELoader": {
                    "input": {"required": {"vae_name": [["a", "b"], {}]}},
                    "input_order": {"required": ["vae_name"]},
                    "output": ["VAE"],
                    "output_name": ["VAE"],
                    "category": "loaders",
                    "display_name": "Load VAE",
                    "description": "",
                    "output_node": False,
                    "python_module": "nodes",
                }
            }
        )
    )
    return oi


def _set_widget(mode_flag: str, workflow_file, object_info):
    return CliRunner().invoke(
        app,
        [mode_flag, "workflow", "set-widget", str(workflow_file), "1.vae_name", "b", "--input", str(object_info)],
        env={"NO_COLOR": "1", "COLUMNS": "400"},
    )


def test_a_crash_in_a_command_ends_the_json_stream_with_an_envelope(monkeypatch, workflow_file, object_info):
    monkeypatch.setattr(workflow_ops, "set_widget", _boom)
    result = _set_widget("--json", workflow_file, object_info)
    assert result.exit_code == 1, result.output
    lines = [ln for ln in result.stdout.splitlines() if ln.strip()]
    assert lines, "stdout must not be empty"
    envelope = json.loads(lines[-1])
    assert envelope["ok"] is False
    err = envelope["error"]
    assert err["code"] == "internal_error", err
    assert err["details"]["exception"] == "KeyError", err
    assert "widgets_values" in err["message"], err
    assert envelope["command"] == "workflow set-widget", envelope


def test_pretty_mode_prints_no_envelope(monkeypatch, workflow_file, object_info):
    monkeypatch.setattr(workflow_ops, "set_widget", _boom)
    result = _set_widget("--no-json", workflow_file, object_info)
    assert result.exit_code == 1
    assert isinstance(result.exception, KeyError), "pretty mode keeps the crash as it was"
    assert "internal_error" not in result.stdout


def test_a_handled_refusal_is_not_relabelled(workflow_file, object_info):
    """typer.Exit after a renderer.error is control flow, not a crash."""
    result = CliRunner().invoke(
        app,
        ["--json", "workflow", "set-widget", str(workflow_file), "1.nope", "b", "--input", str(object_info)],
        env={"NO_COLOR": "1", "COLUMNS": "400"},
    )
    envelopes = [json.loads(ln) for ln in result.stdout.splitlines() if ln.strip().startswith("{")]
    assert [e["error"]["code"] for e in envelopes] == ["workflow_edit_invalid"], envelopes
