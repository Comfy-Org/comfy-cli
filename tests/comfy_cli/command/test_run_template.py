"""Tests for ``comfy run-template`` — fetch → fill params → spend-gate → run.

All offline: the gallery index comes from a tmp fixture file, the template
workflow fetch and the server probe / object_info / run handoff are
monkeypatched. Covers name resolution, --param filling (address and name
keys), the spend gate (gallery signals + object_info partner nodes,
--allow-spend, interactive decline), and the handoff into the run path.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import typer
from typer.testing import CliRunner

from comfy_cli.caller import Caller
from comfy_cli.command import templates as templates_cmd
from comfy_cli.output.renderer import (
    OutputMode,
    Renderer,
    reset_renderer_for_testing,
    set_renderer,
)

# Mirrors the real gallery schema: OSS templates carry no logos; paid
# templates carry the API tag and/or provider logos (both observed in the
# live index — some paid rows have only one of the two signals).
FIXTURE = [
    {
        "moduleName": "default",
        "category": "GENERATION TYPE",
        "title": "Image",
        "type": "image",
        "templates": [
            {
                "name": "api_flux2",
                "title": "Flux 2 (API)",
                "description": "Text-to-image via the BFL API.",
                "mediaType": "image",
                "mediaSubtype": "webp",
                "tags": ["API", "Text to Image"],
                "models": [],
                "logos": [{"provider": ["Black Forest Labs"]}],
                "openSource": False,
                "usage": 100,
            },
            {
                "name": "image_local_sd",
                "title": "Local SD",
                "description": "Local text-to-image.",
                "mediaType": "image",
                "mediaSubtype": "webp",
                "tags": ["Local", "Text to Image"],
                "models": ["SD 1.5"],
                "logos": [],
                "openSource": True,
                "usage": 50,
            },
        ],
    },
]

# Frontend-format template body with one editable text widget and a sampler.
TEMPLATE_WORKFLOW = {
    "nodes": [
        {"id": 3, "type": "KSampler", "widgets_values": [42, "fixed", 20, 8.0, "euler", "normal", 1.0]},
        {"id": 6, "type": "CLIPTextEncode", "widgets_values": ["a cat in space"]},
    ],
    "links": [],
}

OBJECT_INFO = {
    "KSampler": {
        "input": {
            "required": {
                "seed": ["INT", {"default": 0, "control_after_generate": True}],
                "steps": ["INT", {"default": 20}],
                "cfg": ["FLOAT", {"default": 8.0}],
                "sampler_name": [["euler", "euler_ancestral"]],
                "scheduler": [["normal", "karras"]],
                "denoise": ["FLOAT", {"default": 1.0}],
            },
        },
        "input_order": {"required": ["seed", "steps", "cfg", "sampler_name", "scheduler", "denoise"]},
        "output": ["LATENT"],
        "output_name": ["LATENT"],
        "category": "sampling",
        "display_name": "KSampler",
        "python_module": "nodes",
    },
    "CLIPTextEncode": {
        "input": {"required": {"text": ["STRING", {"multiline": True}], "clip": "CLIP"}},
        "input_order": {"required": ["clip", "text"]},
        "output": ["CONDITIONING"],
        "output_name": ["CONDITIONING"],
        "category": "conditioning",
        "display_name": "CLIP Text Encode",
        "python_module": "nodes",
    },
    "FluxProNode": {
        "input": {"required": {"prompt": ["STRING", {"default": ""}]}},
        "input_order": {"required": ["prompt"]},
        "output": ["IMAGE"],
        "output_name": ["IMAGE"],
        "category": "partner/image/BFL",
        "api_node": True,
        "display_name": "Flux Pro",
        "python_module": "nodes",
    },
}


@pytest.fixture
def gallery_file(tmp_path: Path) -> str:
    path = tmp_path / "index.json"
    path.write_text(json.dumps(FIXTURE))
    return str(path)


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


def _envelope(stdout: str) -> dict:
    for line in reversed(stdout.strip().splitlines()):
        try:
            return json.loads(line)
        except json.JSONDecodeError:
            continue
    raise AssertionError(f"no JSON envelope in stdout:\n{stdout}")


@pytest.fixture
def app() -> typer.Typer:
    a = typer.Typer()
    a.command("run-template")(templates_cmd.run_template_cmd)

    # A second (hidden) command so typer doesn't collapse the app into
    # single-command mode, which would swallow the "run-template" prefix.
    @a.command("noop", hidden=True)
    def _noop():  # pragma: no cover
        pass

    return a


class _RunSpy:
    """Records the run-path handoff; captures the workflow file content
    before the command's finally-block unlinks it."""

    def __init__(self):
        self.calls: list[dict[str, Any]] = []

    def __call__(self, workflow_path, host, port, **kwargs):
        self.calls.append(
            {
                "workflow": json.loads(Path(workflow_path).read_text(encoding="utf-8")),
                "host": host,
                "port": port,
                **kwargs,
            }
        )


@pytest.fixture
def run_spy(monkeypatch) -> _RunSpy:
    spy = _RunSpy()
    monkeypatch.setattr("comfy_cli.command.run.execute", spy)
    return spy


@pytest.fixture
def server_up(monkeypatch):
    monkeypatch.setattr("comfy_cli.env_checker.check_comfy_server_running", lambda *a, **k: True)
    monkeypatch.setattr("comfy_cli.command.run._fetch_object_info", lambda host, port: OBJECT_INFO)
    # Keep the user's real config (background server address) out of the tests.
    monkeypatch.setattr("comfy_cli.config_manager.ConfigManager.background", None, raising=False)


@pytest.fixture
def template_body(monkeypatch):
    def set_body(data) -> None:
        body = json.dumps(data).encode()
        monkeypatch.setattr(templates_cmd, "_fetch_template_workflow", lambda name, **kw: body)

    set_body(TEMPLATE_WORKFLOW)
    return set_body


HOSTPORT = ["--host", "127.0.0.1", "--port", "8188"]


# ---------------------------------------------------------------------------
# Resolution + validation failures (no run handoff)
# ---------------------------------------------------------------------------


def test_unknown_template_surfaces_not_found_with_close_matches(app, gallery_file, run_spy):
    _force_json_renderer()
    result = CliRunner().invoke(app, ["run-template", "--gallery", gallery_file, "flux"])
    assert result.exit_code == 1
    env = _envelope(result.output)
    assert env["error"]["code"] == "template_not_found"
    assert env["error"]["details"]["close_matches"] == ["api_flux2"]
    assert run_spy.calls == []


def test_malformed_param_fails_before_any_io(app, gallery_file, run_spy):
    _force_json_renderer()
    result = CliRunner().invoke(app, ["run-template", "--gallery", gallery_file, "image_local_sd", "--param", "seed"])
    assert result.exit_code == 1
    env = _envelope(result.output)
    assert env["error"]["code"] == "workflow_slot_invalid"
    assert run_spy.calls == []


def test_server_not_running_short_circuits(app, gallery_file, template_body, run_spy, monkeypatch):
    _force_json_renderer()
    monkeypatch.setattr("comfy_cli.env_checker.check_comfy_server_running", lambda *a, **k: False)
    monkeypatch.setattr("comfy_cli.config_manager.ConfigManager.background", None, raising=False)
    result = CliRunner().invoke(app, ["run-template", "--gallery", gallery_file, "image_local_sd", *HOSTPORT])
    assert result.exit_code == 1
    env = _envelope(result.output)
    assert env["error"]["code"] == "server_not_running"
    assert run_spy.calls == []


def test_unknown_param_key_lists_available_slots(app, gallery_file, template_body, server_up, run_spy):
    _force_json_renderer()
    result = CliRunner().invoke(
        app,
        ["run-template", "--gallery", gallery_file, "image_local_sd", "--param", "nope=1", *HOSTPORT],
    )
    assert result.exit_code == 1
    env = _envelope(result.output)
    assert env["error"]["code"] == "workflow_slot_invalid"
    assert "6.text" in env["error"]["details"]["available"]
    assert run_spy.calls == []


def test_param_on_api_format_template_is_rejected(app, gallery_file, template_body, server_up, run_spy):
    _force_json_renderer()
    template_body({"9": {"class_type": "KSampler", "inputs": {}}})  # API format, no `nodes` list
    result = CliRunner().invoke(
        app,
        ["run-template", "--gallery", gallery_file, "image_local_sd", "--param", "seed=1", *HOSTPORT],
    )
    assert result.exit_code == 1
    env = _envelope(result.output)
    assert env["error"]["code"] == "workflow_slot_invalid"
    assert run_spy.calls == []


# ---------------------------------------------------------------------------
# Spend gate
# ---------------------------------------------------------------------------


def test_paid_template_without_consent_is_blocked(app, gallery_file, template_body, server_up, run_spy):
    _force_json_renderer()
    result = CliRunner().invoke(app, ["run-template", "--gallery", gallery_file, "api_flux2", *HOSTPORT])
    assert result.exit_code == 1
    env = _envelope(result.output)
    assert env["error"]["code"] == "spend_consent_required"
    assert "tag:API" in env["error"]["details"]["gallery_signals"]
    assert run_spy.calls == []


def test_partner_nodes_gate_even_without_gallery_signals(app, gallery_file, template_body, server_up, run_spy):
    # An OSS-listed template whose workflow embeds a partner node (api_node
    # flag in object_info) must still hit the gate — node detection is the
    # authoritative signal, the gallery index is the fallback.
    _force_json_renderer()
    template_body({"nodes": [{"id": 1, "type": "FluxProNode", "widgets_values": ["x"]}], "links": []})
    result = CliRunner().invoke(app, ["run-template", "--gallery", gallery_file, "image_local_sd", *HOSTPORT])
    assert result.exit_code == 1
    env = _envelope(result.output)
    assert env["error"]["code"] == "spend_consent_required"
    assert env["error"]["details"]["partner_nodes"] == ["FluxProNode"]
    assert run_spy.calls == []


def test_partner_node_inside_subgraph_definition_is_detected():
    wf = {
        "nodes": [{"id": 1, "type": "uuid-subgraph"}],
        "links": [],
        "definitions": {"subgraphs": [{"id": "uuid-subgraph", "nodes": [{"id": 10, "type": "FluxProNode"}]}]},
    }
    assert templates_cmd._detect_paid_nodes(wf, OBJECT_INFO) == ["FluxProNode"]


def test_allow_spend_unblocks_paid_template(app, gallery_file, template_body, server_up, run_spy):
    _force_json_renderer()
    result = CliRunner().invoke(
        app, ["run-template", "--gallery", gallery_file, "api_flux2", "--allow-spend", *HOSTPORT]
    )
    assert result.exit_code == 0, result.output
    assert len(run_spy.calls) == 1


def test_interactive_decline_blocks_without_submitting(
    app, gallery_file, template_body, server_up, run_spy, monkeypatch
):
    # Pretty renderer + a tty stdin → the gate confirms interactively; a "no"
    # answer must not submit anything.
    monkeypatch.setattr("sys.stdin.isatty", lambda: True, raising=False)
    monkeypatch.setattr("typer.confirm", lambda *a, **k: False)
    result = CliRunner().invoke(app, ["run-template", "--gallery", gallery_file, "api_flux2", *HOSTPORT])
    assert result.exit_code == 1
    assert run_spy.calls == []


def test_oss_template_runs_without_spend_flag(app, gallery_file, template_body, server_up, run_spy):
    _force_json_renderer()
    result = CliRunner().invoke(app, ["run-template", "--gallery", gallery_file, "image_local_sd", *HOSTPORT])
    assert result.exit_code == 0, result.output
    assert len(run_spy.calls) == 1


# ---------------------------------------------------------------------------
# Param filling + run handoff
# ---------------------------------------------------------------------------


def test_params_fill_by_address_and_name_then_run_waits(app, gallery_file, template_body, server_up, run_spy):
    _force_json_renderer()
    result = CliRunner().invoke(
        app,
        [
            "run-template",
            "--gallery",
            gallery_file,
            "image_local_sd",
            "--param",
            "6.text=a dog on the moon",  # full address
            "--param",
            "seed=7",  # unique slot name → resolves to 3.seed, JSON-parsed int
            *HOSTPORT,
        ],
    )
    assert result.exit_code == 0, result.output
    assert len(run_spy.calls) == 1
    call = run_spy.calls[0]
    assert call["wait"] is True  # run-template runs to completion by default
    assert call["host"] == "127.0.0.1"
    assert call["port"] == 8188
    nodes = {n["id"]: n for n in call["workflow"]["nodes"]}
    assert nodes[6]["widgets_values"][0] == "a dog on the moon"
    assert nodes[3]["widgets_values"][0] == 7
    assert isinstance(nodes[3]["widgets_values"][0], int)


def test_async_flag_hands_off_without_waiting(app, gallery_file, template_body, server_up, run_spy):
    _force_json_renderer()
    result = CliRunner().invoke(
        app, ["run-template", "--gallery", gallery_file, "image_local_sd", "--async", *HOSTPORT]
    )
    assert result.exit_code == 0, result.output
    assert run_spy.calls[0]["wait"] is False


def test_temp_workflow_file_is_cleaned_up(app, gallery_file, template_body, server_up, run_spy, tmp_path, monkeypatch):
    _force_json_renderer()
    monkeypatch.setenv("TMPDIR", str(tmp_path))
    import tempfile

    monkeypatch.setattr(tempfile, "tempdir", None)  # re-resolve from TMPDIR
    result = CliRunner().invoke(app, ["run-template", "--gallery", gallery_file, "image_local_sd", *HOSTPORT])
    assert result.exit_code == 0, result.output
    assert list(tmp_path.glob("comfy_template_*")) == []
