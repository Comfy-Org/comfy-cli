"""Tests for `comfy validate` — frontend-format (UI-export) auto-conversion.

`comfy validate --workflow <ui-export.json>` used to validate vacuously: a
UI-export file's wrapper keys (`nodes`, `links`, `groups`, `config`, …) each
emitted a `non_node_key` warning, zero nodes were checked, and the verdict was
`valid:true`. The command now detects UI format (`is_ui_workflow`) and lowers it
to API format with `convert_ui_to_api` — exactly as `comfy run` does — before
validating, so the verdict reflects the real graph and the payload carries
`converted_from_ui: true` plus the converted node count.

Offline mode (`--input <object_info.json>`) is used throughout so no server is
needed: the same file supplies both the graph and the converter's object_info.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest
from typer.testing import CliRunner

from comfy_cli.cmdline import app

FIXTURES = Path(__file__).parent.parent / "fixtures"
OBJECT_INFO = FIXTURES / "sd15_object_info.json"
UI_WORKFLOW = FIXTURES / "sd15_ui_workflow.json"


@pytest.fixture
def runner():
    return CliRunner()


def _write(tmp_path: Path, name: str, obj) -> Path:
    p = tmp_path / name
    p.write_text(json.dumps(obj), encoding="utf-8")
    return p


def _envelope(result) -> dict:
    """Parse the final JSON envelope line emitted in `--json` mode."""
    return json.loads(result.stdout.strip().splitlines()[-1])


def _validate(runner: CliRunner, workflow: Path):
    """Invoke `comfy --json validate` offline against the sd15 object_info."""
    return runner.invoke(
        app,
        ["--json", "validate", "--workflow", str(workflow), "--input", str(OBJECT_INFO)],
        env={"COMFY_WHERE": "local"},
    )


def test_ui_export_is_converted_and_validated(runner):
    """A UI-export fixture validates against the CONVERTED graph: a truthful
    verdict, `converted_from_ui: true`, the converted node count, and zero
    `non_node_key` wrapper-key noise."""
    result = _validate(runner, UI_WORKFLOW)

    assert result.exit_code == 0, result.stdout
    data = _envelope(result)["data"]
    assert data["valid"] is True
    assert data["converted_from_ui"] is True
    # The sd15 UI workflow lowers to 7 API nodes.
    assert data["converted_node_count"] == 7
    # The wrapper keys (nodes/links/groups/config/…) are gone after conversion,
    # so none of them can produce the old vacuous-pass warnings.
    assert [w for w in data["warnings"] if w.get("code") == "non_node_key"] == []


def test_ui_export_surfaces_real_problems(runner, tmp_path):
    """Acceptance: the converted graph is really validated — an unknown node
    type surfaces as `valid:false` (not a vacuous pass), while still flagging
    the file as UI-converted."""
    bad = {
        "nodes": [{"id": 1, "type": "TotallyMadeUpNode", "mode": 0, "inputs": [], "outputs": [], "widgets_values": []}],
        "links": [],
    }
    wf = _write(tmp_path, "bad_ui.json", bad)

    result = _validate(runner, wf)

    assert result.exit_code == 1
    data = _envelope(result)["data"]
    assert data["valid"] is False
    assert data["converted_from_ui"] is True
    assert any(e["code"] == "unknown_class_type" for e in data["errors"])


def test_ui_export_that_converts_to_nothing_is_rejected(runner, tmp_path):
    """A UI file whose nodes carry no usable `type` converts to zero executable
    nodes → structured `workflow_not_api_format` error, exit 1, message naming
    the conversion."""
    empty_convert = {"nodes": [{"id": 1, "mode": 0, "inputs": [], "outputs": []}], "links": []}
    wf = _write(tmp_path, "no_exec_ui.json", empty_convert)

    result = _validate(runner, wf)

    assert result.exit_code == 1
    error = _envelope(result)["error"]
    assert error["code"] == "workflow_not_api_format"
    assert "convert" in error["message"].lower()


def test_api_format_unchanged(runner, tmp_path):
    """An API-format file behaves exactly as before: validated directly, no
    `converted_from_ui` key in the payload. The fixture is a complete sd15
    txt2img graph — it carries a SaveImage output node, as any real API-format
    export does, so it clears the server-parity no-outputs check (BE-3357)."""
    api = {
        "4": {
            "class_type": "CheckpointLoaderSimple",
            "inputs": {"ckpt_name": "v1-5-pruned-emaonly-fp16.safetensors"},
        },
        "6": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["4", 1], "text": "a cat"}},
        "7": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["4", 1], "text": "blurry"}},
        "5": {"class_type": "EmptyLatentImage", "inputs": {"width": 512, "height": 512, "batch_size": 1}},
        "3": {
            "class_type": "KSampler",
            "inputs": {
                "model": ["4", 0],
                "positive": ["6", 0],
                "negative": ["7", 0],
                "latent_image": ["5", 0],
                "seed": 42,
                "steps": 20,
                "cfg": 8.0,
                "sampler_name": "euler",
                "scheduler": "simple",
                "denoise": 1.0,
            },
        },
        "8": {"class_type": "VAEDecode", "inputs": {"samples": ["3", 0], "vae": ["4", 2]}},
        "9": {"class_type": "SaveImage", "inputs": {"images": ["8", 0], "filename_prefix": "ComfyUI"}},
    }
    wf = _write(tmp_path, "api.json", api)

    result = _validate(runner, wf)

    assert result.exit_code == 0
    data = _envelope(result)["data"]
    assert data["valid"] is True
    assert "converted_from_ui" not in data


def test_non_dict_payload_unchanged(runner, tmp_path):
    """A non-dict JSON payload keeps its existing `workflow_not_api_format`
    error (the UI-detection branch never runs for it)."""
    wf = _write(tmp_path, "list.json", [1, 2, 3])

    result = _validate(runner, wf)

    assert result.exit_code == 1
    assert _envelope(result)["error"]["code"] == "workflow_not_api_format"


def test_empty_dict_payload_not_converted_and_rejected(runner, tmp_path):
    """An empty dict is not UI format and is left to the existing validator
    (no conversion, no `converted_from_ui` key) — which now rejects it: the
    server refuses any prompt with zero output nodes, including a node-less
    one, so validate mirrors that as `prompt_no_outputs` (BE-3357)."""
    wf = _write(tmp_path, "empty.json", {})

    result = _validate(runner, wf)

    assert result.exit_code == 1
    data = _envelope(result)["data"]
    assert "converted_from_ui" not in data
    assert data["valid"] is False
    assert any(e["code"] == "prompt_no_outputs" for e in data["errors"])


# --- partner-node (paid) visibility (BE-4328) -----------------------------------
#
# `comfy validate` now previews credit spend: it reports `partner_nodes` (the
# partner-API nodes a workflow uses) and `spends_credits` in its payload, using
# the same detection `comfy run` gates on (authoritative `api_node: true`, with a
# `partner/...` category fallback). It stays advisory — no exit-code change.
#
# These tests build a minimal object_info per case (a partner node needs only to
# exist, be an output node, and have no required inputs to validate clean), so the
# detection is exercised without depending on any real partner node's full schema.


def _validate_against(runner: CliRunner, workflow: Path, object_info: Path):
    """`comfy --json validate` offline against a caller-supplied object_info."""
    return runner.invoke(
        app,
        ["--json", "validate", "--workflow", str(workflow), "--input", str(object_info)],
        env={"COMFY_WHERE": "local"},
    )


def _node_info(*, category: str, api_node: bool | None = None) -> dict:
    """A minimal output-node object_info entry with no required inputs — it
    validates clean, so a one-node workflow using it is `valid: true`."""
    info = {"input": {"required": {}}, "output": [], "output_node": True, "category": category}
    if api_node is not None:
        info["api_node"] = api_node
    return info


def test_partner_node_via_api_node_flag(runner, tmp_path):
    """A valid API workflow whose node carries the authoritative `api_node: true`
    flag → `partner_nodes` lists it, `spends_credits` is true, and the exit code
    is unchanged (0, since the workflow is otherwise valid)."""
    oi = _write(tmp_path, "oi.json", {"AcmePartnerImage": _node_info(category="image", api_node=True)})
    wf = _write(tmp_path, "wf.json", {"1": {"class_type": "AcmePartnerImage", "inputs": {}}})

    result = _validate_against(runner, wf, oi)

    assert result.exit_code == 0, result.stdout
    data = _envelope(result)["data"]
    assert data["valid"] is True
    assert data["partner_nodes"] == ["AcmePartnerImage"]
    assert data["spends_credits"] is True


def test_no_partner_nodes_is_empty_list(runner, tmp_path):
    """A workflow with no partner nodes → `partner_nodes: []` (always present)
    and `spends_credits: false`."""
    oi = _write(tmp_path, "oi.json", {"PlainSave": _node_info(category="image")})
    wf = _write(tmp_path, "wf.json", {"1": {"class_type": "PlainSave", "inputs": {}}})

    result = _validate_against(runner, wf, oi)

    assert result.exit_code == 0, result.stdout
    data = _envelope(result)["data"]
    assert data["valid"] is True
    assert data["partner_nodes"] == []
    assert data["spends_credits"] is False


def test_partner_node_via_category_prefix_fallback(runner, tmp_path):
    """A node with a `partner/...` category and no `api_node` flag is still
    detected via the category-prefix fallback."""
    oi = _write(tmp_path, "oi.json", {"AcmeVideo": _node_info(category="partner/video/Acme")})
    wf = _write(tmp_path, "wf.json", {"1": {"class_type": "AcmeVideo", "inputs": {}}})

    result = _validate_against(runner, wf, oi)

    assert result.exit_code == 0, result.stdout
    data = _envelope(result)["data"]
    assert data["partner_nodes"] == ["AcmeVideo"]
    assert data["spends_credits"] is True


def test_partner_detection_runs_on_converted_ui_graph(runner, tmp_path):
    """A UI-format workflow using a partner node: detection runs on the CONVERTED
    (API-format) graph, so the partner node still surfaces alongside
    `converted_from_ui: true`."""
    oi = _write(tmp_path, "oi.json", {"AcmePartnerImage": _node_info(category="image", api_node=True)})
    ui = {
        "nodes": [{"id": 1, "type": "AcmePartnerImage", "mode": 0, "inputs": [], "outputs": [], "widgets_values": []}],
        "links": [],
    }
    wf = _write(tmp_path, "ui.json", ui)

    result = _validate_against(runner, wf, oi)

    assert result.exit_code == 0, result.stdout
    data = _envelope(result)["data"]
    assert data["converted_from_ui"] is True
    assert data["partner_nodes"] == ["AcmePartnerImage"]
    assert data["spends_credits"] is True


def test_invalid_workflow_still_reports_partner_nodes(runner, tmp_path):
    """An invalid workflow that also uses a partner node: both `errors` and
    `partner_nodes` are present (visibility is independent of validity), and the
    exit code is 1 because of the error — not because of the partner node."""
    oi = _write(tmp_path, "oi.json", {"AcmePartnerImage": _node_info(category="image", api_node=True)})
    wf = _write(
        tmp_path,
        "wf.json",
        {
            "1": {"class_type": "AcmePartnerImage", "inputs": {}},
            "2": {"class_type": "TotallyUnknownNode", "inputs": {}},
        },
    )

    result = _validate_against(runner, wf, oi)

    assert result.exit_code == 1
    data = _envelope(result)["data"]
    assert data["valid"] is False
    assert any(e["code"] == "unknown_class_type" for e in data["errors"])
    assert data["partner_nodes"] == ["AcmePartnerImage"]
    assert data["spends_credits"] is True


def test_partner_node_name_with_markup_does_not_crash_pretty(runner, tmp_path):
    """Regression: a partner `class_type` containing Rich-markup metacharacters
    (e.g. `[/yellow]`) is escaped before it reaches `rprint`, so the pretty-mode
    advisory line can't raise `MarkupError` and crash the command."""
    name = "Acme[/yellow]Node"
    oi = _write(tmp_path, "oi.json", {name: _node_info(category="image", api_node=True)})
    wf = _write(tmp_path, "wf.json", {"1": {"class_type": name, "inputs": {}}})

    result = runner.invoke(
        app,
        ["--no-json", "validate", "--workflow", str(wf), "--input", str(oi)],
        env={"COMFY_WHERE": "local"},
    )

    assert result.exit_code == 0, result.stdout
    assert result.exception is None, result.exception
    # The name is still shown (escaped) in the paid-nodes advisory.
    assert name in result.stdout


# --- object_info target resolution (BE-6306) ------------------------------------
#
# `validate` used to resolve its local object_info server as
# `--host`/`--port` > COMFY_LOCAL_URL > 127.0.0.1:8188, skipping the
# `config.background` step `comfy run` honors via `host_port.resolve_host_port`.
# With ComfyUI running as a comfy-cli background server on a non-8188 port, that
# made validate consult a DIFFERENT server than the one `run` submits to, so its
# verdict was meaningless (BE-6299: validate passed a workflow `run` rejected for
# a missing node class). It now resolves through the same chain, and reports the
# resolved target in the payload as `object_info_source`.

BACKGROUND_PORT = 8388


@pytest.fixture
def no_background(monkeypatch):
    """Neutralize the developer's real `config.background` (and COMFY_LOCAL_URL)
    so the tests below observe only what they set themselves."""
    return _set_background(monkeypatch, None)


def _set_background(monkeypatch, background):
    """Point `host_port.resolve_host_port` at a synthetic `config.background`.

    `resolve_host_port` reads `ConfigManager().background`, and ConfigManager
    already drops a record whose pid is dead — so a tuple here stands for a LIVE
    background server.
    """

    class _FakeConfigManager:
        def __init__(self):
            self.background = background

    monkeypatch.setattr("comfy_cli.host_port.ConfigManager", _FakeConfigManager)
    monkeypatch.delenv("COMFY_LOCAL_URL", raising=False)


@pytest.fixture
def captured_target(monkeypatch, tmp_path):
    """Stub the live object_info fetch, recording the host/port it was handed.

    Returns the dict the engine call is recorded into (`{}` until the fetch
    runs), so a test can assert on the server validate actually queried.
    """
    seen: dict = {}

    def _fake_load_from_target(*, mode="local", host=None, port=None):
        seen.update(mode=mode, host=host, port=port)
        return {"PlainSave": _node_info(category="image")}

    monkeypatch.setattr("comfy_cli.cql.engine._load_from_target", _fake_load_from_target)
    return seen


def _valid_workflow(tmp_path: Path) -> Path:
    """A one-node API workflow that validates clean against `PlainSave`."""
    return _write(tmp_path, "wf.json", {"1": {"class_type": "PlainSave", "inputs": {}}})


def _validate_live(runner: CliRunner, workflow: Path, *args: str, json_mode: str = "--json"):
    """Invoke `comfy validate` against a (stubbed) LIVE server — no `--input`."""
    return runner.invoke(
        app,
        [json_mode, "validate", "--workflow", str(workflow), *args],
        env={"COMFY_WHERE": "local"},
    )


def test_local_target_honors_background_server(runner, tmp_path, monkeypatch, captured_target):
    """Acceptance: with a live `config.background` on a non-default port and no
    `--host`/`--port`/`COMFY_LOCAL_URL`, validate fetches object_info from the
    BACKGROUND server — the same one `comfy run` submits to — not 127.0.0.1:8188."""
    _set_background(monkeypatch, ("127.0.0.1", BACKGROUND_PORT, 4242))

    result = _validate_live(runner, _valid_workflow(tmp_path))

    assert result.exit_code == 0, result.stdout
    assert captured_target["port"] == BACKGROUND_PORT
    assert captured_target["host"] == "127.0.0.1"


def test_explicit_flags_beat_background_server(runner, tmp_path, monkeypatch, captured_target):
    """Precedence is unchanged: explicit `--host`/`--port` still win over a
    recorded background server."""
    _set_background(monkeypatch, ("127.0.0.1", BACKGROUND_PORT, 4242))

    result = _validate_live(runner, _valid_workflow(tmp_path), "--host", "127.0.0.1", "--port", "9000")

    assert result.exit_code == 0, result.stdout
    assert captured_target["port"] == 9000
    assert captured_target["host"] == "127.0.0.1"


def test_combined_host_port_flag_is_split(runner, tmp_path, monkeypatch, captured_target):
    """Sharing `comfy run`'s resolution also gives validate `run`'s combined
    `--host host:port` form. It previously flowed through unsplit and produced a
    bogus `http://[127.0.0.1:9100]:8188` URL."""
    _set_background(monkeypatch, ("127.0.0.1", BACKGROUND_PORT, 4242))

    result = _validate_live(runner, _valid_workflow(tmp_path), "--host", "127.0.0.1:9100")

    assert result.exit_code == 0, result.stdout
    assert (captured_target["host"], captured_target["port"]) == ("127.0.0.1", 9100)


def test_malformed_host_is_a_usage_error(runner, tmp_path, no_background, captured_target):
    """A URL-unsafe `--host` is now rejected as a usage error (exit 2) by the
    shared validator, the same way `comfy run` and `comfy upload` reject it,
    instead of being built into a URL and failing at fetch time."""
    result = _validate_live(runner, _valid_workflow(tmp_path), "--host", "evil.example.com/path")

    assert result.exit_code == 2
    assert captured_target == {}  # no fetch was attempted


def test_env_local_url_beats_background_server(runner, tmp_path, monkeypatch, captured_target):
    """`COMFY_LOCAL_URL` also still outranks the background server."""
    _set_background(monkeypatch, ("127.0.0.1", BACKGROUND_PORT, 4242))
    monkeypatch.setenv("COMFY_LOCAL_URL", "http://127.0.0.1:9500")

    result = _validate_live(runner, _valid_workflow(tmp_path))

    assert result.exit_code == 0, result.stdout
    assert captured_target["port"] == 9500


def test_no_background_still_defaults_to_8188(runner, tmp_path, no_background, captured_target):
    """With nothing recorded and no flags, the default target is unchanged."""
    result = _validate_live(runner, _valid_workflow(tmp_path))

    assert result.exit_code == 0, result.stdout
    assert (captured_target["host"], captured_target["port"]) == ("127.0.0.1", 8188)


def test_payload_names_resolved_local_source(runner, tmp_path, monkeypatch, captured_target):
    """The `--json` envelope names the object_info source, carrying the RESOLVED
    host/port (the background server), so an agent can see which server answered."""
    _set_background(monkeypatch, ("127.0.0.1", BACKGROUND_PORT, 4242))

    result = _validate_live(runner, _valid_workflow(tmp_path))

    assert result.exit_code == 0, result.stdout
    data = _envelope(result)["data"]
    assert data["object_info_source"] == {"mode": "local", "host": "127.0.0.1", "port": BACKGROUND_PORT}


def test_payload_names_file_source_under_input(runner, tmp_path, monkeypatch):
    """`--input` is offline mode: the source is the file, and no host/port
    resolution happens (a recorded background server is irrelevant)."""
    _set_background(monkeypatch, ("127.0.0.1", BACKGROUND_PORT, 4242))
    oi = _write(tmp_path, "oi.json", {"PlainSave": _node_info(category="image")})
    wf = _valid_workflow(tmp_path)

    result = _validate_against(runner, wf, oi)

    assert result.exit_code == 0, result.stdout
    data = _envelope(result)["data"]
    assert data["object_info_source"] == {"mode": "file", "path": str(oi)}


def test_payload_names_cloud_source(runner, tmp_path, monkeypatch, captured_target):
    """Cloud mode is unchanged — no local host/port resolution — and the payload
    names the source as `cloud` only."""
    _set_background(monkeypatch, ("127.0.0.1", BACKGROUND_PORT, 4242))

    result = runner.invoke(
        app,
        ["--json", "validate", "--workflow", str(_valid_workflow(tmp_path)), "--where", "cloud"],
    )

    assert result.exit_code == 0, result.stdout
    # The background server must not leak into a cloud fetch.
    assert captured_target["mode"] == "cloud"
    assert (captured_target["host"], captured_target["port"]) == (None, None)
    assert _envelope(result)["data"]["object_info_source"] == {"mode": "cloud"}


def test_pretty_mode_prints_resolved_source(runner, tmp_path, monkeypatch, captured_target):
    """Pretty mode prints one dim line naming the server that answered."""
    _set_background(monkeypatch, ("127.0.0.1", BACKGROUND_PORT, 4242))

    result = _validate_live(runner, _valid_workflow(tmp_path), json_mode="--no-json")

    assert result.exit_code == 0, result.stdout
    assert f"object_info from http://127.0.0.1:{BACKGROUND_PORT}" in result.stdout


# --- review follow-ups on the BE-6306 resolution block ---------------------------


def test_wildcard_background_host_is_canonicalized(runner, tmp_path, monkeypatch, captured_target):
    """`comfy launch -- --listen 0.0.0.0` records the wildcard BIND address in
    `config.background`. Passing it through as a destination trips the
    object_info loopback guard, so validate hard-failed with "Refusing to fetch
    object_info from non-loopback host" where it previously succeeded — the exact
    opposite of the `comfy run` parity this block is for (`run`'s own object_info
    fetch fails open). It is canonicalized to loopback instead."""
    _set_background(monkeypatch, ("0.0.0.0", BACKGROUND_PORT, 4242))

    result = _validate_live(runner, _valid_workflow(tmp_path))

    assert result.exit_code == 0, result.stdout
    assert (captured_target["host"], captured_target["port"]) == ("127.0.0.1", BACKGROUND_PORT)


def test_where_flag_is_normalized_before_routing(runner, tmp_path, monkeypatch, captured_target):
    """`--where` is normalized through the shared resolver, not string-compared
    raw: `--where LOCAL` must take the local path (splitting a combined `--host`,
    consulting the background server) rather than skipping the whole block and
    building `http://[127.0.0.1:9100]:8188`."""
    _set_background(monkeypatch, ("127.0.0.1", BACKGROUND_PORT, 4242))

    result = _validate_live(runner, _valid_workflow(tmp_path), "--where", "LOCAL", "--host", "127.0.0.1:9100")

    assert result.exit_code == 0, result.stdout
    assert (captured_target["host"], captured_target["port"]) == ("127.0.0.1", 9100)
    assert _envelope(result)["data"]["object_info_source"]["mode"] == "local"


@pytest.mark.parametrize("bad_where", ["bogus", "file"])
def test_invalid_where_emits_envelope_not_traceback(runner, tmp_path, no_background, bad_where):
    """An unknown `--where` used to escape as a bare ValueError traceback with no
    envelope at all. (`file` is doubly interesting: it is the offline sentinel
    `object_info_source.mode` also uses.)"""
    result = _validate_live(runner, _valid_workflow(tmp_path), "--where", bad_where)

    assert result.exit_code == 1
    assert _envelope(result)["error"]["code"] == "where_invalid"


def test_empty_host_flag_is_rejected(runner, tmp_path, no_background, captured_target):
    """`--host ""` is not "no host" — it must be rejected rather than silently
    resolving to COMFY_LOCAL_URL / the background server / 127.0.0.1."""
    result = _validate_live(runner, _valid_workflow(tmp_path), "--host", "")

    assert result.exit_code == 2
    assert captured_target == {}


@pytest.mark.parametrize("bad_port", ["0", "99999"])
def test_out_of_range_port_flag_is_rejected(runner, tmp_path, no_background, captured_target, bad_port):
    """An out-of-range `--port` is a usage error. `0` in particular used to read
    as "not passed" and silently resolve to some other server."""
    result = _validate_live(runner, _valid_workflow(tmp_path), "--port", bad_port)

    assert result.exit_code == 2
    assert captured_target == {}


def test_explicit_port_zero_is_not_overridden_by_combined_host(runner, tmp_path, no_background, captured_target):
    """The combined-host port merge tests `is None`, so an explicit (invalid)
    `--port 0` is reported as such instead of being quietly replaced by the port
    embedded in `--host h:p`."""
    result = _validate_live(runner, _valid_workflow(tmp_path), "--host", "127.0.0.1:9100", "--port", "0")

    assert result.exit_code == 2
    assert captured_target == {}


def test_ipv6_source_is_unbracketed_in_payload_and_bracketed_for_display(runner, tmp_path, no_background):
    """Brackets are a URL encoding, not part of the address: the payload carries
    the raw literal (matching `Target.host`) and only the display URL brackets it."""
    with patch("comfy_cli.cql.engine._load_from_target", return_value={"PlainSave": _node_info(category="image")}):
        result = _validate_live(runner, _valid_workflow(tmp_path), "--host", "::1", "--port", "9000")
        assert result.exit_code == 0, result.stdout
        assert _envelope(result)["data"]["object_info_source"] == {"mode": "local", "host": "::1", "port": 9000}

        pretty = _validate_live(
            runner, _valid_workflow(tmp_path), "--host", "::1", "--port", "9000", json_mode="--no-json"
        )
    assert "object_info from http://[::1]:9000" in pretty.stdout
