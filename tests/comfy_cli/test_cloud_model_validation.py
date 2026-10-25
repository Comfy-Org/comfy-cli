"""Uploaded model assets may be absent from Cloud's static enum snapshot."""

from __future__ import annotations

import copy
import json
import urllib.error
from unittest.mock import Mock
from urllib.parse import parse_qs, urlsplit

import pytest

from comfy_cli.cql.engine import Graph
from comfy_cli.target import Target

DIGEST = "a" * 64
NAME = "private.safetensors"


def object_info():
    return {
        "Source": {"input": {"required": {}}, "output": ["MODEL"]},
        "LoraLoaderModelOnly": {
            "input": {
                "required": {"model": ["MODEL"], "lora_name": [["public.safetensors"]], "strength_model": ["FLOAT"]}
            },
            "output": ["MODEL"],
        },
        "Capture": {"input": {"required": {"model": ["MODEL"]}}, "output": [], "output_node": True},
    }


def workflow():
    return {
        "1": {"class_type": "Source", "inputs": {}},
        "2": {
            "class_type": "LoraLoaderModelOnly",
            "inputs": {"model": ["1", 0], "lora_name": NAME, "strength_model": 1.0},
        },
        "3": {"class_type": "Capture", "inputs": {"model": ["2", 0]}},
    }


def validation(wf=None):
    return Graph.from_object_info(object_info()).validate_workflow(wf or workflow())


def asset(**changes):
    return {
        "id": "asset-private",
        "name": NAME,
        "hash": DIGEST + ".safetensors",
        "tags": ["models", "loras"],
        **changes,
    }


@pytest.fixture
def service(monkeypatch):
    from comfy_cli import cloud_model_validation as cv

    monkeypatch.delenv("COMFY_OBJECT_INFO_FILE", raising=False)
    monkeypatch.setattr(
        cv,
        "resolve_target",
        lambda **kw: Target(kind="cloud", base_url="https://cloud.test", path_prefix="/api", auth_token="test-token"),
    )
    request = Mock(return_value=(200, {"assets": [asset()], "has_more": False}))
    monkeypatch.setattr(cv, "request_json", request)
    return cv, request


def test_authenticated_exact_model_asset_resolves_real_graph_error(service):
    cv, request = service
    result = validation()
    before = copy.deepcopy(result)
    assert [e["code"] for e in result["errors"]] == ["unknown_enum_value"]
    resolved = cv.resolve_cloud_model_errors(result, workflow(), where="cloud")
    assert resolved["valid"] is True
    assert resolved["errors"] == []
    assert resolved["warnings"][-1]["code"] == "cloud_model_asset_verified"
    assert resolved["warnings"][-1]["asset_id"] == "asset-private"
    assert result == before
    query = parse_qs(urlsplit(request.call_args.args[0]).query)
    assert query["include_public"] == ["false"]
    assert query["include_tags"] == ["models"]
    assert query["name_contains"] == [NAME]
    assert request.call_args.kwargs["max_bytes"] <= 4 * 1024 * 1024


@pytest.mark.parametrize(
    "row",
    [
        asset(name="prefix-" + NAME),
        asset(tags=["models", "vae"]),
        asset(tags=["loras"]),
        asset(hash="unknown"),
        asset(hash=None),
        asset(id=None),
        asset(name="other", loader_path="other"),
        asset(tags="models,loras"),
    ],
)
def test_fuzzy_wrong_folder_or_unverifiable_asset_retains_error(service, row):
    cv, request = service
    request.return_value = (200, {"assets": [row], "has_more": False})
    result = validation()
    assert cv.resolve_cloud_model_errors(result, workflow(), where="cloud") == result


def test_exact_server_loader_path_is_accepted(service):
    cv, request = service
    request.return_value = (200, {"assets": [asset(name="display name", loader_path=NAME)], "has_more": False})
    assert cv.resolve_cloud_model_errors(validation(), workflow(), where="cloud")["valid"] is True


@pytest.mark.parametrize(
    "body",
    [
        None,
        {},
        {"assets": None, "has_more": False},
        {"assets": [asset()]},
        {"assets": [asset()], "has_more": True},
        {"assets": [asset(), "malformed"], "has_more": False},
    ],
)
def test_incomplete_or_malformed_result_retains_original_diagnostics(service, body):
    cv, request = service
    request.return_value = (200, body)
    result = validation()
    assert cv.resolve_cloud_model_errors(result, workflow(), where="cloud") == result


@pytest.mark.parametrize(
    "error",
    [
        OSError("offline"),
        ValueError("bad response"),
        TimeoutError("timeout"),
        urllib.error.HTTPError("https://cloud.test", 403, "forbidden", {}, None),
    ],
)
def test_unavailable_lookup_retains_original_diagnostics(service, error):
    cv, request = service
    request.side_effect = error
    result = validation()
    assert cv.resolve_cloud_model_errors(result, workflow(), where="cloud") == result


@pytest.mark.parametrize(
    "where,offline,env", [("local", False, None), ("cloud", True, None), ("cloud", False, "/offline/catalog.json")]
)
def test_offline_or_local_validation_does_not_resolve_target_or_network(service, monkeypatch, where, offline, env):
    cv, request = service
    target = Mock(side_effect=AssertionError("must stay offline"))
    monkeypatch.setattr(cv, "resolve_target", target)
    if env:
        monkeypatch.setenv("COMFY_OBJECT_INFO_FILE", env)
    result = validation()
    assert cv.resolve_cloud_model_errors(result, workflow(), where=where, offline=offline) == result
    target.assert_not_called()
    request.assert_not_called()


@pytest.mark.parametrize("change", ["nonmodel", "otherfield", "othercode", "valid"])
def test_arbitrary_enums_and_other_errors_do_not_start_network(service, monkeypatch, change):
    cv, request = service
    wf = workflow()
    result = validation()
    if change == "nonmodel":
        wf["2"]["class_type"] = "KSampler"
    elif change == "otherfield":
        result["errors"][0]["field"] = "strength_model"
    elif change == "othercode":
        result["errors"][0]["code"] = "no_options_available"
    else:
        result = {"valid": True, "errors": [], "warnings": []}
    target = Mock(side_effect=AssertionError("unrelated error must not start network"))
    monkeypatch.setattr(cv, "resolve_target", target)
    assert cv.resolve_cloud_model_errors(result, wf, where="cloud") == result
    request.assert_not_called()
    target.assert_not_called()


def test_unauthenticated_target_keeps_error_without_assets_read(service, monkeypatch):
    cv, request = service
    monkeypatch.setattr(cv, "resolve_target", lambda **kw: Target(kind="cloud", base_url="https://cloud.test"))
    result = validation()
    assert cv.resolve_cloud_model_errors(result, workflow(), where="cloud") == result
    request.assert_not_called()


def test_conflicting_exact_asset_hashes_are_ambiguous(service):
    cv, request = service
    request.return_value = (200, {"assets": [asset(), asset(id="asset-other", hash="b" * 64)], "has_more": False})
    result = validation()
    assert cv.resolve_cloud_model_errors(result, workflow(), where="cloud") == result


def test_only_resolved_error_removed_other_diagnostics_unchanged(service):
    cv, _ = service
    result = validation()
    unrelated = {"node_id": "2", "field": "model", "code": "shape_mismatch", "message": "test invalid shape"}
    result["errors"].append(unrelated)
    resolved = cv.resolve_cloud_model_errors(result, workflow(), where="cloud")
    assert resolved["valid"] is False
    assert resolved["errors"] == [unrelated]


def test_repeated_same_model_uses_one_bounded_lookup(service):
    cv, request = service
    wf = workflow()
    wf["4"] = copy.deepcopy(wf["2"])
    result = validation(wf)
    assert len(result["errors"]) == 2
    assert cv.resolve_cloud_model_errors(result, wf, where="cloud")["valid"] is True
    assert request.call_count == 1


def test_run_preflight_uses_authenticated_asset_verification(service):
    from comfy_cli.command.run.preflight import _preflight_validate

    _, request = service
    renderer = Mock()
    renderer.is_pretty.return_value = False
    _preflight_validate(renderer, workflow(), object_info(), where="cloud", target_label="cloud")
    renderer.error.assert_not_called()
    assert request.call_count == 1


def test_workflow_validate_uses_same_asset_check_and_offline_override(service, monkeypatch, tmp_path):
    from comfy_cli.command import workflow as cmd
    from comfy_cli.cql import loader

    _, request = service
    renderer = Mock()
    renderer.is_pretty.return_value = False
    monkeypatch.setattr(cmd, "get_renderer", lambda: renderer)
    monkeypatch.setattr(loader, "resilient_load_object_info", lambda **kw: object_info())
    path = tmp_path / "workflow.json"
    path.write_text(json.dumps(workflow()))
    cmd.validate_api_workflow(str(path), where="cloud")
    assert renderer.emit.call_args.args[0]["valid"] is True
    request.assert_called_once()
    request.reset_mock()
    with pytest.raises(Exception):
        cmd.validate_api_workflow(str(path), where="cloud", input_path="offline-catalog.json")
    request.assert_not_called()


def test_actual_authenticated_http_transport_uses_supported_models_filter(service, monkeypatch):
    import io

    from comfy_cli import http

    cv, _ = service
    seen = []

    class Response(io.BytesIO):
        status = 200

    def open_request(request, *, timeout):
        query = parse_qs(urlsplit(request.full_url).query)
        assert query["include_tags"] == ["models"]
        assert query["include_public"] == ["false"]
        assert query["name_contains"] == [NAME]
        assert request.get_method() == "GET"
        assert request.get_header("Authorization") == "Bearer test-token"
        assert timeout == 10
        seen.append(request.full_url)
        return Response(json.dumps({"assets": [asset()], "has_more": False}).encode())

    monkeypatch.setattr(http._AUTHED_OPENER, "open", open_request)
    monkeypatch.setattr(cv, "request_json", http.request_json)
    assert cv.resolve_cloud_model_errors(validation(), workflow(), where="cloud")["valid"] is True
    assert len(seen) == 1


def test_bounded_http_body_preserves_original_error(service, monkeypatch):
    import io

    from comfy_cli import http

    cv, _ = service

    class Response(io.BytesIO):
        status = 200

    monkeypatch.setattr(http._AUTHED_OPENER, "open", lambda *args, **kw: Response(b"x" * (2 * 1024 * 1024 + 1)))
    monkeypatch.setattr(cv, "request_json", http.request_json)
    result = validation()
    assert cv.resolve_cloud_model_errors(result, workflow(), where="cloud") == result


@pytest.mark.parametrize("hash_value", [DIGEST, DIGEST + ".safetensors", "blake3:" + DIGEST])
def test_supported_server_hash_forms(service, hash_value):
    cv, request = service
    request.return_value = (200, {"assets": [asset(hash=hash_value)], "has_more": False})
    resolved = cv.resolve_cloud_model_errors(validation(), workflow(), where="cloud")
    assert resolved["valid"] is True
    assert resolved["warnings"][-1]["content_hash"] == "blake3:" + DIGEST


def test_unique_lookups_are_bounded_and_unchecked_errors_remain(service):
    cv, request = service
    wf = {}
    errors = []
    for i in range(17):
        wf[str(i)] = {"class_type": "LoraLoaderModelOnly", "inputs": {"lora_name": f"{i}.safetensors"}}
        errors.append({"node_id": str(i), "field": "lora_name", "code": "unknown_enum_value"})

    def lookup(url, *args, **kwargs):
        name = parse_qs(urlsplit(url).query)["name_contains"][0]
        return 200, {"assets": [asset(name=name)], "has_more": False}

    request.side_effect = lookup
    result = cv.resolve_cloud_model_errors({"valid": False, "errors": errors, "warnings": []}, wf, where="cloud")
    assert request.call_count == 16
    assert result["valid"] is False
    assert result["errors"] == [errors[-1]]
