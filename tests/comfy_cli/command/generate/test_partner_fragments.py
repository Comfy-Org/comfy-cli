"""Validate the shipped partner-node starter fragments.

Each fragment must parse under the same loader ``comfy workflow fragment
validate`` uses (``comfy_cli.fragments.load_fragment``), declare the expected
typed interface, and wrap the real partner node class.
"""

from pathlib import Path

import pytest
from typer.testing import CliRunner

from comfy_cli.cmdline import app as cli_app
from comfy_cli.fragments import load_fragment

FRAGMENT_DIR = Path(__file__).resolve().parents[4] / "comfy_cli" / "fragments_lib"

# name -> (partner node class_type, expected inputs, expected param subset)
EXPECTED = {
    "nano_banana_edit": ("GeminiImageNode", {"image"}, {"prompt", "model", "seed", "aspect_ratio"}),
    "seedance_i2v": (
        "ByteDanceImageToVideoNode",
        {"image"},
        {"prompt", "model", "resolution", "aspect_ratio", "duration", "seed"},
    ),
    "flux_t2i": ("Flux2ProImageNode", set(), {"prompt", "width", "height", "seed", "prompt_upsampling"}),
    "kling_i2v": (
        "KlingImage2VideoNode",
        {"start_frame"},
        {"prompt", "negative_prompt", "model_name", "cfg_scale", "mode", "aspect_ratio", "duration"},
    ),
}


@pytest.fixture(autouse=True)
def disable_tracking(monkeypatch):
    monkeypatch.setattr("comfy_cli.tracking.prompt_tracking_consent", lambda *a, **kw: None)
    monkeypatch.setattr("comfy_cli.tracking.track_event", lambda *a, **kw: None)


@pytest.mark.parametrize("name", sorted(EXPECTED))
def test_fragment_loads_and_declares_interface(name):
    frag = load_fragment(FRAGMENT_DIR / f"{name}.json")
    node_class, exp_inputs, exp_params = EXPECTED[name]
    # Interface matches.
    assert set(frag.inputs) == exp_inputs
    assert exp_params <= set(frag.params)
    # The interior node is the real partner class.
    classes = {n["class_type"] for n in frag.nodes.values()}
    assert node_class in classes


@pytest.mark.parametrize("name", sorted(EXPECTED))
def test_fragment_binds_point_at_real_nodes(name):
    frag = load_fragment(FRAGMENT_DIR / f"{name}.json")
    for port in list(frag.inputs.values()) + list(frag.params.values()):
        node_id = port.binds.split(".", 1)[0]
        assert node_id in frag.nodes
    for port in frag.outputs.values():
        assert port.from_node in frag.nodes


@pytest.mark.parametrize("name", sorted(EXPECTED))
def test_cli_fragment_validate_passes(name):
    runner = CliRunner()
    r = runner.invoke(
        cli_app,
        ["--json", "workflow", "fragment", "validate", str(FRAGMENT_DIR / f"{name}.json")],
    )
    assert r.exit_code == 0, r.stdout
    assert '"valid": true' in r.stdout
