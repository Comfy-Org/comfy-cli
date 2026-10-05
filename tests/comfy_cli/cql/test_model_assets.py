"""A model file the catalog lacks but Cloud loads from its asset library is
not a catalog error (comfy_cli.cql.model_assets)."""

from __future__ import annotations

import json
import urllib.error
import urllib.parse
from pathlib import Path

import pytest
from typer.testing import CliRunner

from comfy_cli.cmdline import app
from comfy_cli.cql import model_assets
from comfy_cli.cql.engine import Graph
from comfy_cli.model_variants import resolve_workflow_models

INT8 = "minimax_h3_video_vae_int8_convrot.safetensors"
FP16 = "minimax_h3_video_vae_fp16.safetensors"
VAES = [FP16, "ae.safetensors", "wan_2.1_vae.safetensors"]


def _object_info() -> dict:
    return {
        "VAELoader": {
            "input": {"required": {"vae_name": [VAES, {}]}},
            "input_order": {"required": ["vae_name"]},
            "output": ["VAE"],
            "output_name": ["VAE"],
            "category": "loaders",
            "display_name": "Load VAE",
            "description": "",
            "output_node": False,
            "python_module": "nodes",
        },
        "PreviewAny": {
            "input": {"required": {"mode": [["fast", "slow"], {}], "source": ["VAE", {}]}},
            "input_order": {"required": ["mode", "source"]},
            "output": [],
            "output_name": [],
            "category": "utils",
            "display_name": "Preview",
            "description": "",
            "output_node": True,
            "python_module": "nodes",
        },
    }


class Lookup:
    """A recording lookup over a fixed set of asset names."""

    def __init__(self, *names: str, raises: Exception | None = None):
        self.names = set(names)
        self.calls: list[str] = []
        self.raises = raises

    def __call__(self, name: str) -> bool:
        self.calls.append(name)
        if self.raises:
            raise self.raises
        return name in self.names


@pytest.fixture
def graph():
    return Graph.from_object_info(_object_info())


def _vae_port(graph):
    return next(p for p in graph.node("VAELoader").inputs if p.name == "vae_name")


def test_catalog_miss_resolved_by_asset_is_not_a_finding(graph):
    lookup = Lookup(INT8)
    model_assets.set_lookup(lookup)
    assert _vae_port(graph).validate_catalog(INT8) == []
    assert lookup.calls == [INT8]


def test_catalog_miss_with_no_asset_is_still_unknown_enum_value(graph):
    model_assets.set_lookup(Lookup())
    findings = _vae_port(graph).validate_catalog(INT8)
    assert [f["code"] for f in findings] == ["unknown_enum_value"]
    assert findings[0].get("precision_sibling") == FP16, "the existing precision offer is unchanged"


def test_no_lookup_installed_keeps_todays_finding(graph):
    # The suite default (conftest): off cloud there is no asset library.
    findings = _vae_port(graph).validate_catalog(INT8)
    assert [f["code"] for f in findings] == ["unknown_enum_value"]


def test_lookup_failure_keeps_the_finding(graph):
    model_assets.set_lookup(Lookup(INT8, raises=urllib.error.URLError("down")))
    findings = _vae_port(graph).validate_catalog(INT8)
    assert [f["code"] for f in findings] == ["unknown_enum_value"], "a failed lookup never accepts"


def test_catalog_hit_never_looks_up(graph):
    lookup = Lookup(FP16)
    model_assets.set_lookup(lookup)
    assert _vae_port(graph).validate_catalog(FP16) == []
    assert lookup.calls == []


def test_only_model_ports_and_model_file_values_are_looked_up(graph):
    lookup = Lookup("turbo.safetensors", "nope")
    model_assets.set_lookup(lookup)
    mode = next(p for p in graph.node("PreviewAny").inputs if p.name == "mode")
    # A filename never satisfies a dropdown that is not a model list.
    assert [f["code"] for f in mode.validate_catalog("turbo.safetensors")] == ["unknown_enum_value"]
    # A non-file value on a model port is not a model to look up.
    assert [f["code"] for f in _vae_port(graph).validate_catalog("nope")] == ["unknown_enum_value"]
    assert lookup.calls == []


def test_each_name_is_looked_up_once(graph):
    lookup = Lookup(INT8)
    model_assets.set_lookup(lookup)
    port = _vae_port(graph)
    for _ in range(3):
        assert port.validate_catalog(INT8) == []
    assert lookup.calls == [INT8]


def _template(vae: str) -> dict:
    return {
        "nodes": [{"id": 1, "type": "VAELoader", "widgets_values": [vae], "inputs": [], "outputs": []}],
        "links": [],
        "last_node_id": 1,
        "last_link_id": 0,
        "version": 0.4,
    }


def test_template_keeps_an_asset_backed_model_instead_of_swapping_precision(graph):
    model_assets.set_lookup(Lookup(INT8))
    wf = _template(INT8)
    subs, unavailable = resolve_workflow_models(wf, graph)
    assert subs == [] and unavailable == []
    assert wf["nodes"][0]["widgets_values"] == [INT8], "the template's own build is kept"


def test_template_still_swaps_when_no_asset_backs_the_model(graph):
    model_assets.set_lookup(Lookup())
    wf = _template(INT8)
    subs, _ = resolve_workflow_models(wf, graph)
    assert [s["code"] for s in subs] == ["normalized_value"]
    assert wf["nodes"][0]["widgets_values"] == [FP16]


# --- the CLI surface: validate and set-widget -------------------------------


def _api_workflow(vae: str) -> dict:
    return {
        "1": {"class_type": "VAELoader", "inputs": {"vae_name": vae}},
        "2": {"class_type": "PreviewAny", "inputs": {"mode": "fast", "source": ["1", 0]}},
    }


def _validate(tmp_path: Path, vae: str) -> dict:
    oi = tmp_path / "object_info.json"
    oi.write_text(json.dumps(_object_info()))
    wf = tmp_path / "wf.json"
    wf.write_text(json.dumps(_api_workflow(vae)))
    result = CliRunner().invoke(app, ["--json", "validate", "--workflow", str(wf), "--input", str(oi)])
    return json.loads(result.stdout)


def test_validate_passes_an_asset_backed_model(tmp_path):
    model_assets.set_lookup(Lookup(INT8))
    env = _validate(tmp_path, INT8)
    assert env["ok"] is True, env
    assert env["data"]["error_count"] == 0, env


def test_validate_still_rejects_a_model_with_no_asset(tmp_path):
    model_assets.set_lookup(Lookup())
    env = _validate(tmp_path, INT8)
    codes = [e["code"] for e in env["data"]["errors"]] if env.get("data") else [env["error"]["code"]]
    assert "unknown_enum_value" in codes, env


# --- the HTTP lookup ----------------------------------------------------------


class _Target:
    is_cloud = True
    auth_token = "t"
    api_key = None

    def url(self, *parts):
        return "https://cloud.example/api/" + "/".join(parts)


def _patch_request(monkeypatch, body=None, raises=None):
    seen = {}

    def fake(url, target, **kw):
        seen["url"] = url
        if raises:
            raise raises
        return 200, body

    monkeypatch.setattr("comfy_cli.http.request_json", fake)
    return seen


def test_asset_lookup_matches_the_exact_name_only(monkeypatch):
    seen = _patch_request(
        monkeypatch,
        {"assets": [{"name": "x_" + INT8}, {"name": INT8.upper()}, {"name": INT8}]},
    )
    assert model_assets.asset_name_exists(_Target(), INT8) is True
    assert "include_tags=models" in seen["url"] and "include_public=true" in seen["url"]
    assert "name_contains=" + INT8 in seen["url"]


def test_asset_lookup_rejects_near_matches(monkeypatch):
    _patch_request(monkeypatch, {"assets": [{"name": "x_" + INT8}, {"name": INT8.upper()}]})
    assert model_assets.asset_name_exists(_Target(), INT8) is False


@pytest.mark.parametrize(
    "body,raises",
    [
        (None, urllib.error.HTTPError("u", 500, "boom", {}, None)),
        (None, urllib.error.URLError("down")),
        (["not", "an", "object"], None),
        (None, None),
    ],
)
def test_asset_lookup_failure_is_not_found(monkeypatch, body, raises):
    _patch_request(monkeypatch, body, raises)
    assert model_assets.asset_name_exists(_Target(), INT8) is False


def test_lookup_resolves_off_for_a_local_target(monkeypatch):
    model_assets.reset()

    class Local:
        is_cloud = False
        auth_token = None
        api_key = None

    monkeypatch.setattr("comfy_cli.target.resolve_target", lambda **kw: Local())
    assert model_assets.model_asset_exists(INT8) is False


def test_lookup_resolves_off_without_a_cloud_credential(monkeypatch):
    model_assets.reset()

    class NoCred(_Target):
        auth_token = None

    monkeypatch.setattr("comfy_cli.target.resolve_target", lambda **kw: NoCred())
    called = []
    monkeypatch.setattr(model_assets, "asset_name_exists", lambda *a: called.append(a) or True)
    assert model_assets.model_asset_exists(INT8) is False
    assert called == []


def test_disable_env_turns_the_lookup_off(monkeypatch):
    model_assets.reset()
    monkeypatch.setenv(model_assets.DISABLE_ENV, "1")
    monkeypatch.setattr("comfy_cli.target.resolve_target", lambda **kw: _Target())
    monkeypatch.setattr(model_assets, "asset_name_exists", lambda *a: True)
    assert model_assets.model_asset_exists(INT8) is False


def test_cloud_target_with_credential_uses_the_asset_lookup(monkeypatch):
    model_assets.reset()
    monkeypatch.setattr("comfy_cli.target.resolve_target", lambda **kw: _Target())
    monkeypatch.setattr(model_assets, "asset_name_exists", lambda t, name: name == INT8)
    assert model_assets.model_asset_exists(INT8) is True
    assert model_assets.model_asset_exists("other.safetensors") is False


# --- exact value, paging, routing ---------------------------------------------


def test_the_value_is_looked_up_untrimmed(graph):
    lookup = Lookup(INT8)
    model_assets.set_lookup(lookup)
    findings = _vae_port(graph).validate_catalog(f" {INT8}")
    assert [f["code"] for f in findings] == ["unknown_enum_value"], "the server matches the value as is"
    assert lookup.calls == [f" {INT8}"]


def _paged(monkeypatch, pages: list[list[str]], total: int):
    calls = []

    def fake(url, target, **kw):
        q = urllib.parse.parse_qs(urllib.parse.urlsplit(url).query)
        offset = int(q["offset"][0])
        calls.append(offset)
        idx = offset // model_assets._PAGE
        rows = pages[idx] if idx < len(pages) else []
        return 200, {"assets": [{"name": n} for n in rows], "total": total}

    monkeypatch.setattr("comfy_cli.http.request_json", fake)
    return calls


def test_asset_lookup_reads_later_pages(monkeypatch):
    first = [f"a{i}_{INT8}" for i in range(model_assets._PAGE)]
    calls = _paged(monkeypatch, [first, [INT8]], total=model_assets._PAGE + 1)
    assert model_assets.asset_name_exists(_Target(), INT8) is True
    assert calls == [0, model_assets._PAGE]


def test_asset_lookup_stops_when_the_listing_is_exhausted(monkeypatch):
    calls = _paged(monkeypatch, [["x_" + INT8]], total=1)
    assert model_assets.asset_name_exists(_Target(), INT8) is False
    assert calls == [0]


def test_asset_lookup_is_bounded(monkeypatch):
    many = [f"a{i}_{INT8}" for i in range(model_assets._PAGE)]
    calls = _paged(monkeypatch, [many] * 50, total=10**6)
    assert model_assets.asset_name_exists(_Target(), INT8) is False
    assert len(calls) == model_assets._MAX_PAGES


def test_the_commands_where_routes_the_lookup(monkeypatch):
    model_assets.reset()
    seen = []

    def resolve(**kw):
        seen.append(kw["where"])
        return _Target() if kw["where"] == "cloud" else type("L", (), {"is_cloud": False})()

    monkeypatch.setattr("comfy_cli.target.resolve_target", resolve)
    monkeypatch.setattr(model_assets, "asset_name_exists", lambda t, name: True)
    model_assets.use_where("cloud")
    assert model_assets.model_asset_exists(INT8) is True
    # A different route drops the resolved lookup and its cached answers.
    model_assets.use_where("local")
    assert model_assets.model_asset_exists(INT8) is False
    assert seen == ["cloud", "local"]


def test_an_installed_lookup_survives_a_route(graph):
    lookup = Lookup(INT8)
    model_assets.set_lookup(lookup)
    model_assets.use_where("cloud")
    assert _vae_port(graph).validate_catalog(INT8) == []


def test_the_workflow_graph_loader_routes_the_lookup(tmp_path, monkeypatch):
    from comfy_cli.command import workflow

    model_assets.reset()
    oi = tmp_path / "object_info.json"
    oi.write_text(json.dumps(_object_info()))
    workflow._get_graph(str(oi), None, None, where="cloud")
    assert model_assets._where == "cloud"


def _route_stubs(monkeypatch):
    """resolve_target honours the route it is given; every asset exists."""
    model_assets.reset()
    seen = []

    class Local:
        is_cloud = False
        auth_token = None
        api_key = None

    def resolve(**kw):
        seen.append(kw["where"])
        return _Target() if kw["where"] == "cloud" else Local()

    monkeypatch.setattr("comfy_cli.target.resolve_target", resolve)
    monkeypatch.setattr(model_assets, "asset_name_exists", lambda t, name: True)
    return seen


def test_a_default_route_change_drops_cloud_answers(monkeypatch):
    seen = _route_stubs(monkeypatch)
    monkeypatch.setenv("COMFY_WHERE", "cloud")
    model_assets.use_where(None)
    assert model_assets.model_asset_exists(INT8) is True
    monkeypatch.setenv("COMFY_WHERE", "local")
    model_assets.use_where(None)
    assert model_assets.model_asset_exists(INT8) is False, "a cloud answer never carries over to local"
    assert seen == ["cloud", "local"]


def test_an_unresolvable_route_means_no_lookup(monkeypatch):
    _route_stubs(monkeypatch)
    monkeypatch.setenv("COMFY_WHERE", "not-a-route")
    model_assets.use_where(None)
    assert model_assets._where == "local"
    assert model_assets.model_asset_exists(INT8) is False


@pytest.mark.parametrize("flag,expect_ok", [("local", False), ("cloud", True)])
def test_validate_binds_its_own_route_over_the_default(tmp_path, monkeypatch, flag, expect_ok):
    _route_stubs(monkeypatch)
    monkeypatch.setenv("COMFY_WHERE", "cloud")
    oi = tmp_path / "object_info.json"
    oi.write_text(json.dumps(_object_info()))
    wf = tmp_path / "wf.json"
    wf.write_text(json.dumps(_api_workflow(INT8)))
    result = CliRunner().invoke(app, ["--json", "validate", "--workflow", str(wf), "--input", str(oi), "--where", flag])
    env = json.loads(result.stdout)
    assert env["ok"] is expect_ok, env
