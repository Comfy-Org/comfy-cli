"""Tests for ``comfy generate <model> --emit-workflow`` and the underlying
``emit`` module: model→node-class mapping, param translation, and the emitted
frontend-format workflow + op-stream shape.

``build_workflow`` used to hand-construct a raw API-format
dict (``workflow["1"] = {"class_type": ..., "inputs": ...}``). It now routes
through ``workflow_ops.py``'s CRDT-ready primitives (``add_node``/``connect``/
``set_widget``), so the output is frontend-format (``nodes``/``links``) plus a
replayable op list, and downstream ``comfy workflow`` slot-editing commands
(gated on ``_is_frontend_format``) accept it instead of rejecting it as
``workflow_not_frontend_format``. Assertions below read the emitted graph by
node ``type``/``widgets_values``/``links`` instead of API-format literals.
"""

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from comfy_cli.cmdline import app as cli_app
from comfy_cli.command.generate import emit, spec
from comfy_cli.cql.engine import Graph

# Recorded object_info for the partner nodes MODEL_NODE_MAP targets, snapshotted
# from the cloud catalog. Used to enforce NodeSpec's completeness contract: the
# emitted node must carry EVERY widget input, optional section included (a
# schema-`optional` input may still be positionally required by execute()).
#
# Same file `emit.py` bundles as `comfy_cli/command/generate/data/emit_object_info.json`
# (single source of truth — see that module's `_load_catalog` docstring for why
# this cannot be a live `object_info` fetch).
PARTNER_OBJECT_INFO = json.loads(
    (Path(__file__).parent / "fixtures" / "partner_nodes_object_info.json").read_text(encoding="utf-8")
)


@pytest.fixture(autouse=True)
def disable_tracking_prompt(monkeypatch):
    monkeypatch.setattr("comfy_cli.tracking.prompt_tracking_consent", lambda *a, **kw: None)
    monkeypatch.setattr("comfy_cli.tracking.track_event", lambda *a, **kw: None)


@pytest.fixture
def runner():
    return CliRunner()


def _by_type(wf: dict, class_type: str) -> list[dict]:
    return [n for n in wf["nodes"] if n["type"] == class_type]


def _one(wf: dict, class_type: str) -> dict:
    nodes = _by_type(wf, class_type)
    assert len(nodes) == 1, f"expected exactly one {class_type!r} node, found {len(nodes)}"
    return nodes[0]


def _widget(node: dict, graph: Graph, name: str):
    """The value ``node``'s widget ``name`` carries, resolved by the schema's
    positional widget order (frontend-format stores ``widgets_values``
    positionally, never by name)."""
    order = graph.widget_order_default(node["type"])
    idx = order.index(name)
    return node["widgets_values"][idx]


def _linked_from(wf: dict, node: dict, input_name: str) -> dict:
    """The upstream node feeding ``node``'s link input ``input_name``."""
    port = next(i for i in node["inputs"] if i["name"] == input_name)
    link = next(link_ for link_ in wf["links"] if link_[0] == port["link"])
    _link_id, from_node, _from_slot, _to_node, _to_slot, _type = link
    return next(n for n in wf["nodes"] if n["id"] == from_node)


def _catalog() -> Graph:
    return emit._catalog()


# ─── unit: build_workflow ─────────────────────────────────────────────────


def test_build_flux_text_to_image_class_type_and_params():
    wf, ops = emit.build_workflow("flux-2", {"prompt": "a fox", "width": 512})
    graph = _catalog()
    partner = _one(wf, "Flux2ProImageNode")
    assert _widget(partner, graph, "prompt") == "a fox"
    # user override applied, fixed default preserved for unset params
    assert _widget(partner, graph, "width") == 512
    assert _widget(partner, graph, "height") == 768
    # save node references the partner output
    save = _one(wf, "SaveImage")
    assert _linked_from(wf, save, "images") is partner
    assert ops and all("op_id" in op for op in ops)


def test_build_nano_banana_wires_load_image():
    wf, _ops = emit.build_workflow("nano-banana", {"prompt": "add sunglasses", "image": "cat.png"})
    graph = _catalog()
    partner = _one(wf, "GeminiImageNode")
    # an image param becomes a LoadImage node wired into `images`
    loader = _one(wf, "LoadImage")
    assert _widget(loader, graph, "image") == "cat.png"
    assert _linked_from(wf, partner, "images") is loader


def test_build_seedance_emits_save_video():
    wf, _ops = emit.build_workflow("seedance", {"prompt": "drift", "image": "frame.png", "duration": 8})
    graph = _catalog()
    partner = _one(wf, "ByteDanceImageToVideoNode")
    assert _widget(partner, graph, "duration") == 8
    save = _one(wf, "SaveVideo")
    assert _linked_from(wf, save, "video") is partner


def test_build_kling_i2v_class_and_start_frame():
    wf, _ops = emit.build_workflow("kling-i2v", {"prompt": "zoom in", "image": "start.png"})
    partner = _one(wf, "KlingImage2VideoNode")
    loader = _one(wf, "LoadImage")
    assert _linked_from(wf, partner, "start_frame") is loader


def test_build_seedance_fills_execute_required_defaults():
    """Regression: ByteDanceImageToVideoNode declares seed/camera_fixed/watermark
    `optional=True` in its schema but its execute() takes them WITHOUT Python
    defaults — omitting them validates cleanly and then fails the run with
    "missing 3 required positional arguments" (observed live on cloud). The
    emitter must always write them."""
    wf, _ops = emit.build_workflow(
        "seedance",
        {"prompt": "drift", "image": "frame.png", "model": "seedance-1-0-lite-i2v-250428"},
    )
    graph = _catalog()
    partner = _one(wf, "ByteDanceImageToVideoNode")
    assert _widget(partner, graph, "seed") == 0
    assert _widget(partner, graph, "camera_fixed") is False
    assert _widget(partner, graph, "watermark") is False
    assert _widget(partner, graph, "generate_audio") is False


def test_build_seedance_proxy_flag_spellings_reach_node_inputs():
    """The generate proxy flags are --ratio/--camerafixed; the node inputs are
    aspect_ratio/camera_fixed. User-passed values must not be dropped."""
    wf, _ops = emit.build_workflow(
        "seedance",
        {"prompt": "drift", "image": "frame.png", "ratio": "9:16", "camerafixed": True, "watermark": True},
    )
    graph = _catalog()
    partner = _one(wf, "ByteDanceImageToVideoNode")
    assert _widget(partner, graph, "aspect_ratio") == "9:16"
    assert _widget(partner, graph, "camera_fixed") is True
    assert _widget(partner, graph, "watermark") is True


@pytest.mark.parametrize("model", sorted(emit.MODEL_NODE_MAP))
def test_emitted_node_covers_every_widget_input(model):
    """Completeness contract (see NodeSpec docstring): for every model the
    emitter supports, the emitted partner node must contain ALL of the node's
    widget (non-link) inputs — required AND optional — plus every required link
    input. Schema-`optional` does not imply optional-at-execute for V3 nodes,
    so any absent widget input is a potential run-time crash."""
    ns = emit.MODEL_NODE_MAP[model]
    graph = Graph.from_object_info(PARTNER_OBJECT_INFO)
    meta = graph.node(ns.node_class)
    assert meta is not None, f"{ns.node_class} missing from the fixture snapshot — refresh it"

    values = {"prompt": "p"}
    if ns.image_params:
        values[next(iter(ns.image_params))] = "img.png"
    wf, _ops = emit.build_workflow(model, values)
    partner = _one(wf, ns.node_class)
    order = graph.widget_order_default(ns.node_class)
    widget_names = {name for name in order if name in graph.widget_defaults(ns.node_class) or True}
    input_names = {i["name"] for i in partner["inputs"]}

    for port in meta.inputs:
        if port.is_link:
            if port.required:
                assert port.name in input_names, f"{model}: required link input {port.name!r} not wired"
            continue
        assert port.name in widget_names, (
            f"{model}: widget input {port.name!r} missing from the emitted node — "
            f"schema-optional inputs may still be positionally required at execute() "
            f"time; add a default to MODEL_NODE_MAP[{model!r}].fixed"
        )


def test_unknown_model_lists_supported():
    with pytest.raises(emit.EmitError) as ei:
        emit.build_workflow("dalle", {"prompt": "x"})
    msg = str(ei.value)
    assert "flux-2" in msg and "flux-ultra" in msg and "nano-banana" in msg


def test_every_mapped_node_endpoint_matches_its_alias():
    # A NodeSpec's endpoint records what the ComfyUI node actually calls;
    # the alias must proxy to the same endpoint, or emit silently swaps models.
    assert emit.MODEL_NODE_MAP, "an empty map would make this invariant vacuous"
    known = spec.aliases()
    for alias, ns in emit.MODEL_NODE_MAP.items():
        # `resolve_alias` echoes an unrecognized name back unchanged, so a
        # typo'd key would otherwise satisfy the equality below against itself.
        assert alias in known, alias
        assert spec.resolve_alias(alias) == ns.endpoint, alias


def test_flux_pro_is_rejected_no_node_for_flux_pro_1_1():
    # `flux-pro` means BFL Flux Pro 1.1, which has no ComfyUI node — emit must
    # fail loudly rather than quietly emit a graph for a different Flux model.
    with pytest.raises(emit.EmitError) as ei:
        emit.build_workflow("flux-pro", {"prompt": "x"})
    assert "flux-ultra" in str(ei.value)


def test_build_flux_ultra_folds_width_height_into_aspect_ratio():
    wf, _ops = emit.build_workflow("flux-ultra", {"prompt": "a fox", "width": 1024, "height": 768})
    graph = _catalog()
    partner = _one(wf, "FluxProUltraImageNode")
    # the node takes an aspect ratio, not w/h — the two flags fold into it
    assert _widget(partner, graph, "aspect_ratio") == "1024:768"
    assert _widget(partner, graph, "prompt") == "a fox"
    # the Ultra node's own default, unlike Flux2Pro's True
    assert _widget(partner, graph, "prompt_upsampling") is False
    save = _one(wf, "SaveImage")
    assert _linked_from(wf, save, "images") is partner


def test_build_flux_ultra_without_width_height_keeps_default_aspect_ratio():
    wf, _ops = emit.build_workflow("flux-ultra", {"prompt": "a fox"})
    graph = _catalog()
    partner = _one(wf, "FluxProUltraImageNode")
    assert _widget(partner, graph, "aspect_ratio") == "16:9"


def test_build_flux_ultra_only_width_errors_instead_of_dropping_it():
    # flux-ultra has no fixed width/height fallback (unlike flux-2), so a lone
    # --width would otherwise be silently dropped in favor of the "16:9" default.
    with pytest.raises(emit.EmitError) as ei:
        emit.build_workflow("flux-ultra", {"prompt": "a fox", "width": 1024})
    assert "--width" in str(ei.value) and "--height" in str(ei.value)


def test_build_flux_ultra_only_height_errors_instead_of_dropping_it():
    with pytest.raises(emit.EmitError) as ei:
        emit.build_workflow("flux-ultra", {"prompt": "a fox", "height": 768})
    assert "--width" in str(ei.value) and "--height" in str(ei.value)


def test_emitted_workflow_is_frontend_format():
    wf, _ops = emit.build_workflow("flux-2", {"prompt": "p"})
    assert isinstance(wf["nodes"], list)
    assert isinstance(wf["links"], list)
    for node in wf["nodes"]:
        assert isinstance(node["id"], int)
        assert "type" in node
        assert "widgets_values" in node


def test_build_workflow_ops_replay_to_the_same_workflow():
    """P1 fidelity (workflow_ops.py docstring): apply(base, op) == primitive(base).
    The op stream `build_workflow` returns must independently replay, through
    `apply_op`, to the exact same graph — this is what makes it a real op batch
    an `apply_ops` consumer can use, not just receipts alongside a fixed dict."""
    from comfy_cli.workflow_ops import apply_op

    wf, ops = emit.build_workflow("nano-banana", {"prompt": "p", "image": "cat.png"})
    graph = _catalog()
    replayed: dict = {"nodes": [], "links": [], "last_node_id": 0, "last_link_id": 0}
    for op in ops:
        replayed = apply_op(replayed, op, graph)
    # Compare the graph-shape (ids are minted fresh per call, so compare types/
    # widgets/links structurally rather than asserting dict equality against a
    # second, independently-minted `wf`).
    assert sorted(n["type"] for n in replayed["nodes"]) == sorted(n["type"] for n in wf["nodes"])
    assert len(replayed["links"]) == len(wf["links"])


def test_is_supported_answers_for_alias_and_canonical_id():
    """The flag `generate list` exposes must agree with what `build_workflow`
    accepts — alias or the canonical endpoint id the alias resolves to."""
    assert emit.is_supported("flux-2") is True
    assert emit.is_supported(spec.resolve_alias("flux-2")) is True
    assert emit.is_supported("flux-pro") is False
    assert emit.is_supported("bfl/flux-pro-1.1/generate") is False
    assert emit.is_supported("no-such-model") is False


def test_unsupported_model_raises_a_typed_error_carrying_the_supported_list():
    with pytest.raises(emit.UnsupportedModelError) as ei:
        emit.build_workflow("flux-pro", {"prompt": "x"})
    assert ei.value.model == "flux-pro"
    assert ei.value.supported == emit.supported_models()
    assert isinstance(ei.value, emit.EmitError)  # callers catching the base still work


# ─── CLI integration ──────────────────────────────────────────────────────


def test_cli_emit_unsupported_model_has_its_own_error_code(runner, tmp_path, monkeypatch):
    """The prod payload: `generate_workflow flux-pro` → `emit_workflow_failed`
    "--emit-workflow does not support model 'flux-pro'. Supported: …". The
    umbrella code also covers bad params and unwritable paths, so the agent
    could not tell "pick another model" from "fix your arguments". Now it is
    its own code, with the supported aliases as data, not prose."""
    monkeypatch.delenv("COMFY_API_KEY", raising=False)
    monkeypatch.setenv("COMFY_OUTPUT", "json")
    out = tmp_path / "wf.json"
    r = runner.invoke(cli_app, ["generate", "flux-pro", "--prompt", "x", "--emit-workflow", str(out)])
    assert r.exit_code == 1
    lines = [ln for ln in r.stdout.splitlines() if ln.strip().startswith("{")]
    env = json.loads(lines[-1])
    assert env["ok"] is False
    err = env["error"]
    assert err["code"] == "emit_workflow_unsupported_model"
    assert err["details"]["model"] == "flux-pro"
    assert err["details"]["supported"] == emit.supported_models()
    assert "generate list" in err["hint"]
    assert not out.exists()


def test_cli_emit_writes_file_no_api_key(runner, tmp_path, monkeypatch):
    # No COMFY_API_KEY set: emit must not require one.
    monkeypatch.delenv("COMFY_API_KEY", raising=False)
    out = tmp_path / "wf.json"
    r = runner.invoke(
        cli_app,
        ["generate", "flux-2", "--prompt", "a cat", "--emit-workflow", str(out)],
    )
    assert r.exit_code == 0, r.stdout
    assert out.is_file()
    wf = json.loads(out.read_text())
    graph = _catalog()
    partner = _one(wf, "Flux2ProImageNode")
    assert _widget(partner, graph, "prompt") == "a cat"


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
        ["generate", "flux-2", "--prompt", "x", "--width", "1024", "--height", "768", "--emit-workflow", str(out)],
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
            "flux-2",
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
    save = next(n for n in wf["nodes"] if n["type"] == "SaveImage")
    graph = _catalog()
    assert _widget(save, graph, "filename_prefix") == "myfox"


def test_build_workflow_single_element_list_unwraps():
    wf, _ops = emit.build_workflow("nano-banana", {"prompt": "p", "image": ["ref.jpg"]})
    assert len(_by_type(wf, "LoadImage")) == 1
    assert not _by_type(wf, "ImageBatch")


def test_build_workflow_two_images_chains_imagebatch():
    wf, _ops = emit.build_workflow("nano-banana", {"prompt": "p", "image": ["a.jpg", "b.jpg"]})
    loaders = _by_type(wf, "LoadImage")
    batches = _by_type(wf, "ImageBatch")
    assert len(loaders) == 2 and len(batches) == 1
    batch = batches[0]
    loader_ids = {n["id"] for n in loaders}
    fed_ids = {_linked_from(wf, batch, "image1")["id"], _linked_from(wf, batch, "image2")["id"]}
    assert fed_ids == loader_ids
    partner = _one(wf, "GeminiImageNode")
    assert _linked_from(wf, partner, "images") is batch


def test_build_workflow_three_images_chains_two_batches():
    wf, _ops = emit.build_workflow("nano-banana", {"prompt": "p", "image": ["a.jpg", "b.jpg", "c.jpg"]})
    batches = _by_type(wf, "ImageBatch")
    assert len(batches) == 2
    # terminal batch feeds the partner node
    partner = _one(wf, "GeminiImageNode")
    upstream = _linked_from(wf, partner, "images")
    assert upstream["type"] == "ImageBatch"
