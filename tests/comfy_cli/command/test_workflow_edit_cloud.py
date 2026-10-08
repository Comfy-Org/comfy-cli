"""Cloud red→green for the structured-edit primitives — the real-vendor gate.

Unlike the offline unit tests (which stub `object_info`), these build a graph
against the **live Comfy Cloud node catalog** and prove it converts + submits.
This is the test that catches drift between our primitives and the real schemas.

Gating (so the default suite stays offline + free):
  * All tests skip unless `COMFY_CLOUD_E2E=1` AND a cloud session exists
    (`comfy cloud login`).
  * The submit test additionally needs `COMFY_CLOUD_E2E_RUN=1` — it spends credits.

Run after login:
    COMFY_CLOUD_E2E=1 uv run --extra dev pytest tests/comfy_cli/command/test_workflow_edit_cloud.py -v
    # include a real job submission (spends credits):
    COMFY_CLOUD_E2E=1 COMFY_CLOUD_E2E_RUN=1 uv run --extra dev pytest tests/comfy_cli/command/test_workflow_edit_cloud.py -v
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest


def _cloud_ready() -> bool:
    """True iff a cloud target with usable credentials is configured."""
    try:
        from comfy_cli.target import resolve_target

        t = resolve_target(where="cloud")
    except Exception:
        return False
    return bool(getattr(t, "api_key", None) or getattr(t, "auth_token", None))


pytestmark = pytest.mark.skipif(
    not (os.environ.get("COMFY_CLOUD_E2E") and _cloud_ready()),
    reason="cloud e2e: set COMFY_CLOUD_E2E=1 and run `comfy cloud login` first",
)


def _cloud_object_info() -> dict:
    from comfy_cli.cql.loader import resilient_load_object_info

    return resilient_load_object_info(mode="cloud", host="127.0.0.1", port=8188)


def _first_enum(graph, class_type: str, widget: str):
    """A real, catalog-valid value for a COMBO widget (e.g. a checkpoint name)."""
    m = graph.node(class_type)
    if m is None:
        pytest.skip(f"cloud catalog has no {class_type}")
    port = next((p for p in m.inputs if p.name == widget), None)
    if port is None or not port.enum_values:
        pytest.skip(f"cloud catalog exposes no choices for {class_type}.{widget}")
    return port.enum_values[0]


def _build_txt2img(graph):
    """Build a minimal txt2img graph with the edit primitives, using real
    catalog values. Returns (workflow, id_map)."""
    from comfy_cli import workflow_ops as w

    wf = {"nodes": [], "links": [], "last_node_id": 0, "last_link_id": 0}
    ids: dict[str, int] = {}
    for key, ct in [
        ("ckpt", "CheckpointLoaderSimple"),
        ("pos", "CLIPTextEncode"),
        ("neg", "CLIPTextEncode"),
        ("latent", "EmptyLatentImage"),
        ("ks", "KSampler"),
        ("vae", "VAEDecode"),
        ("save", "SaveImage"),
    ]:
        wf, op = w.add_node(wf, graph, ct)
        ids[key] = op["node_id"]

    def C(a, aslot, b, bslot):
        nonlocal wf
        wf, _ = w.connect(wf, graph, ids[a], aslot, ids[b], bslot)

    C("ckpt", "MODEL", "ks", "model")
    C("ckpt", "CLIP", "pos", "clip")
    C("ckpt", "CLIP", "neg", "clip")
    C("pos", "CONDITIONING", "ks", "positive")
    C("neg", "CONDITIONING", "ks", "negative")
    C("latent", "LATENT", "ks", "latent_image")
    C("ks", "LATENT", "vae", "samples")
    C("ckpt", "VAE", "vae", "vae")
    C("vae", "IMAGE", "save", "images")

    wf, _ = w.set_widget(wf, graph, ids["ckpt"], "ckpt_name", _first_enum(graph, "CheckpointLoaderSimple", "ckpt_name"))
    wf, _ = w.set_widget(wf, graph, ids["pos"], "text", "a serene mountain lake at dawn")
    wf, _ = w.set_widget(wf, graph, ids["neg"], "text", "blurry, low quality")
    wf, _ = w.set_widget(wf, graph, ids["ks"], "steps", 8)
    return wf, ids


def test_build_txt2img_against_live_cloud_catalog():
    """RED→GREEN: the primitives must produce a graph that converts cleanly
    against the REAL cloud schemas (not a stub)."""
    from comfy_cli.cql.engine import Graph
    from comfy_cli.workflow_to_api import convert_ui_to_api

    oi = _cloud_object_info()
    graph = Graph.from_object_info(oi)
    wf, ids = _build_txt2img(graph)

    api = convert_ui_to_api(wf, oi)
    # every node survived conversion
    for key, nid in ids.items():
        assert str(nid) in api, f"{key} (node {nid}) dropped by converter"
    # KSampler's model input resolved to the checkpoint node (link not dropped)
    ks = api[str(ids["ks"])]
    assert ks["inputs"]["model"][0] == str(ids["ckpt"])
    # no null required enum left behind (the add-node default-fill guarantee)
    assert api[str(ids["ckpt"])]["inputs"]["ckpt_name"] is not None


@pytest.mark.skipif(
    not os.environ.get("COMFY_CLOUD_E2E_RUN"), reason="submit spends credits: set COMFY_CLOUD_E2E_RUN=1"
)
def test_submit_built_graph_to_cloud(tmp_path):
    """RED→GREEN (credit-gated): a primitive-built graph submits to cloud and
    returns a prompt_id. Runs the real `comfy run --where cloud`."""
    from comfy_cli.cql.engine import Graph

    graph = Graph.from_object_info(_cloud_object_info())
    wf, _ = _build_txt2img(graph)
    wf_path = tmp_path / "built.json"
    wf_path.write_text(json.dumps(wf), encoding="utf-8")

    proc = subprocess.run(
        [sys.executable, "-m", "comfy_cli", "--json", "run", "--workflow", str(wf_path), "--where", "cloud"],
        capture_output=True,
        text=True,
        timeout=180,
        cwd=str(Path(__file__).resolve().parents[3]),
    )
    env = None
    for line in reversed([ln for ln in proc.stdout.strip().splitlines() if ln.strip()]):
        try:
            env = json.loads(line)
            break
        except json.JSONDecodeError:
            continue
    assert env is not None, f"no envelope (rc={proc.returncode}, stderr={proc.stderr[:500]})"
    assert env["ok"] is True, env.get("error")
    assert env["data"].get("prompt_id"), f"expected a prompt_id, got {env['data']}"
