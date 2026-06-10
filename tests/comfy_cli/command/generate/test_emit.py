"""Tests for ``comfy generate <model> --emit-workflow`` and the underlying
``emit`` module: model→node-class mapping, param translation, and the emitted
API-format workflow shape.
"""

import json

import pytest
from typer.testing import CliRunner

from comfy_cli.cmdline import app as cli_app
from comfy_cli.command.generate import emit


@pytest.fixture(autouse=True)
def disable_tracking_prompt(monkeypatch):
    monkeypatch.setattr("comfy_cli.tracking.prompt_tracking_consent", lambda *a, **kw: None)
    monkeypatch.setattr("comfy_cli.tracking.track_event", lambda *a, **kw: None)


@pytest.fixture
def runner():
    return CliRunner()


# ─── unit: build_workflow ─────────────────────────────────────────────────


def test_build_flux_text_to_image_class_type_and_params():
    wf = emit.build_workflow("flux-pro", {"prompt": "a fox", "width": 512})
    # partner node is "1"
    assert wf["1"]["class_type"] == "Flux2ProImageNode"
    assert wf["1"]["inputs"]["prompt"] == "a fox"
    # user override applied, fixed default preserved for unset params
    assert wf["1"]["inputs"]["width"] == 512
    assert wf["1"]["inputs"]["height"] == 768
    # save node references the partner output
    save = [n for n in wf.values() if n["class_type"] == "SaveImage"]
    assert len(save) == 1
    assert save[0]["inputs"]["images"] == ["1", 0]


def test_build_nano_banana_wires_load_image():
    wf = emit.build_workflow("nano-banana", {"prompt": "add sunglasses", "image": "cat.png"})
    assert wf["1"]["class_type"] == "GeminiImageNode"
    # an image param becomes a LoadImage node wired into `images`
    loaders = [(k, v) for k, v in wf.items() if v["class_type"] == "LoadImage"]
    assert len(loaders) == 1
    loader_id, loader = loaders[0]
    assert loader["inputs"]["image"] == "cat.png"
    assert wf["1"]["inputs"]["images"] == [loader_id, 0]


def test_build_seedance_emits_save_video():
    wf = emit.build_workflow("seedance", {"prompt": "drift", "image": "frame.png", "duration": 8})
    assert wf["1"]["class_type"] == "ByteDanceImageToVideoNode"
    assert wf["1"]["inputs"]["duration"] == 8
    save = [n for n in wf.values() if n["class_type"] == "SaveVideo"]
    assert len(save) == 1
    assert save[0]["inputs"]["video"] == ["1", 0]


def test_build_kling_i2v_class_and_start_frame():
    wf = emit.build_workflow("kling-i2v", {"prompt": "zoom in", "image": "start.png"})
    assert wf["1"]["class_type"] == "KlingImage2VideoNode"
    loader_id = next(k for k, v in wf.items() if v["class_type"] == "LoadImage")
    assert wf["1"]["inputs"]["start_frame"] == [loader_id, 0]


def test_unknown_model_lists_supported():
    with pytest.raises(emit.EmitError) as ei:
        emit.build_workflow("dalle", {"prompt": "x"})
    msg = str(ei.value)
    assert "flux-pro" in msg and "nano-banana" in msg


def test_emitted_workflow_is_api_format_node_ids_are_strings():
    wf = emit.build_workflow("flux-pro", {"prompt": "p"})
    for k, node in wf.items():
        assert isinstance(k, str)
        assert "class_type" in node
        assert "inputs" in node


# ─── CLI integration ──────────────────────────────────────────────────────


def test_cli_emit_writes_file_no_api_key(runner, tmp_path, monkeypatch):
    # No COMFY_API_KEY set: emit must not require one.
    monkeypatch.delenv("COMFY_API_KEY", raising=False)
    out = tmp_path / "wf.json"
    r = runner.invoke(
        cli_app,
        ["generate", "flux-pro", "--prompt", "a cat", "--emit-workflow", str(out)],
    )
    assert r.exit_code == 0, r.stdout
    assert out.is_file()
    wf = json.loads(out.read_text())
    assert wf["1"]["class_type"] == "Flux2ProImageNode"
    assert wf["1"]["inputs"]["prompt"] == "a cat"


def test_cli_emit_json_mode_prints_workflow(runner, tmp_path, monkeypatch):
    # --json (generate-local flag) is now superseded by the global renderer
    # envelope. Use COMFY_OUTPUT=json to put the renderer in JSON mode and
    # assert the output is a proper envelope, not a bare workflow dict.
    monkeypatch.delenv("COMFY_API_KEY", raising=False)
    monkeypatch.setenv("COMFY_OUTPUT", "json")
    out = tmp_path / "wf.json"
    r = runner.invoke(
        cli_app,
        ["generate", "nano-banana", "--prompt", "hi", "--emit-workflow", str(out)],
    )
    assert r.exit_code == 0, r.stdout
    lines = [ln for ln in r.stdout.splitlines() if ln.strip().startswith("{")]
    env = json.loads(lines[-1])
    assert env.get("ok") is True
    assert env["data"]["out"].endswith("wf.json")


def test_emit_workflow_uses_envelope(runner, monkeypatch, tmp_path):
    # Force the renderer into JSON-envelope mode via the COMFY_OUTPUT env var
    # (which Renderer.resolve() reads from os.environ in the @app.callback).
    # This is the global mechanism — distinct from the generate-local --json
    # flag that _separate_meta_flags() parses (which no longer drives emit output).
    monkeypatch.delenv("COMFY_API_KEY", raising=False)
    monkeypatch.setenv("COMFY_OUTPUT", "json")
    out = tmp_path / "wf.json"
    r = runner.invoke(
        cli_app,
        ["generate", "flux-pro", "--prompt", "x", "--width", "1024", "--height", "768", "--emit-workflow", str(out)],
    )
    assert r.exit_code == 0, r.stdout
    lines = [ln for ln in r.stdout.splitlines() if ln.strip().startswith("{")]
    env = json.loads(lines[-1])
    assert env.get("ok") is True
    assert "command" in env and "data" in env  # envelope shape, not a bare workflow dict
    assert env["data"]["out"].endswith("wf.json")


def test_cli_emit_unsupported_model_errors(runner, tmp_path, monkeypatch):
    monkeypatch.setenv("COMFY_API_KEY", "comfyui-test")
    out = tmp_path / "wf.json"
    r = runner.invoke(
        cli_app,
        ["generate", "dalle", "--prompt", "x", "--emit-workflow", str(out)],
    )
    assert r.exit_code == 1
    assert "does not support" in r.stdout
    assert not out.exists()


def test_cli_emit_output_prefix(runner, tmp_path, monkeypatch):
    monkeypatch.delenv("COMFY_API_KEY", raising=False)
    out = tmp_path / "wf.json"
    r = runner.invoke(
        cli_app,
        [
            "generate",
            "flux-pro",
            "--prompt",
            "p",
            "--emit-workflow",
            str(out),
            "--output-prefix",
            "myfox",
        ],
    )
    assert r.exit_code == 0, r.stdout
    wf = json.loads(out.read_text())
    save = next(n for n in wf.values() if n["class_type"] == "SaveImage")
    assert save["inputs"]["filename_prefix"] == "myfox"
