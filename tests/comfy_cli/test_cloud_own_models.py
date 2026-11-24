"""A model the workspace uploaded to its Comfy Cloud library is a valid loader value.

Cloud's ``object_info`` lists Cloud's own models only, so the CLI's enum check
refused every upload before ``comfy run --where cloud`` could submit it, although
Cloud runs it. The run preflight now asks the library (``/api/assets``, the lookup
``comfy workflow check`` uses) about each refused model value.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest
import typer

from comfy_cli import cloud_own_models
from comfy_cli.cloud_own_models import OwnModels, admit_refused, note_refusals
from comfy_cli.command.run.preflight import _preflight_validate
from comfy_cli.cql.engine import Graph
from comfy_cli.output.renderer import reset_renderer_for_testing


def _node(inputs: dict, outputs: list[str], *, output_node: bool = False) -> dict:
    return {
        "input": {"required": inputs},
        "input_order": {"required": list(inputs)},
        "output": outputs,
        "output_name": outputs,
        "output_node": output_node,
        "category": "test",
        "display_name": "",
        "description": "",
        "python_module": "nodes",
    }


CLOUD_CKPTS = ["sd_xl_base_1.0.safetensors", "cardosAnime_v20.safetensors"]

OBJECT_INFO = {
    "CheckpointLoaderSimple": _node({"ckpt_name": [list(CLOUD_CKPTS), {}]}, ["MODEL", "CLIP", "VAE"]),
    # Cloud ships no file for this folder at all: a declared, empty option list.
    "UNETLoader": _node({"unet_name": [[], {}]}, ["MODEL"]),
    "KSamplerLite": _node({"model": ["MODEL", {}], "sampler_name": [["euler", "dpmpp_2m"], {}]}, ["LATENT"]),
    "ModelSink": _node({"model": ["MODEL", {}]}, [], output_node=True),
}

UPLOAD = {"name": "my_finetune_v1.safetensors", "tags": ["models", "checkpoints"], "size": 7, "hash": None}


def _workflow(ckpt: str, *, sampler: str = "euler") -> dict:
    return {
        "1": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": ckpt}},
        "2": {"class_type": "KSamplerLite", "inputs": {"model": ["1", 0], "sampler_name": sampler}},
        "3": {"class_type": "ModelSink", "inputs": {"model": ["1", 0]}},
    }


class Library:
    """A fake ``/api/assets`` lookup that records what it was asked."""

    def __init__(self, *rows: dict, fail: Exception | None = None):
        self.rows = list(rows)
        self.fail = fail
        self.asked: list[str] = []

    def __call__(self, name: str) -> list[dict]:
        self.asked.append(name)
        if self.fail is not None:
            raise self.fail
        return [r for r in self.rows if name in r["name"]]


@pytest.fixture(autouse=True)
def _reset_renderer():
    reset_renderer_for_testing()
    yield
    reset_renderer_for_testing()


def _renderer():
    r = MagicMock()
    r.is_pretty.return_value = False
    return r


def _refusal(renderer) -> dict:
    renderer.error.assert_called_once()
    return renderer.error.call_args.kwargs


# --- OwnModels: Cloud's own matching rule, one lookup per file name ----------


@pytest.mark.parametrize(
    ("value", "rows", "expected"),
    [
        ("my_finetune_v1.safetensors", [UPLOAD], ("present", "my_finetune_v1.safetensors", [])),
        # Cloud runs the saved value when exactly one upload ends in "/<value>".
        ("m.safetensors", [{"name": "kept/m.safetensors"}], ("present", "kept/m.safetensors", [])),
        # Same file name under another path: the saved value would have to change.
        ("ckpt/m.safetensors", [{"name": "m.safetensors"}], ("remap", "m.safetensors", [])),
        (
            "m.safetensors",
            [{"name": "a/m.safetensors"}, {"name": "b/m.safetensors"}],
            ("missing", None, ["a/m.safetensors", "b/m.safetensors"]),
        ),
        ("my_finetune_v1.safetensors", [], ("missing", None, [])),
    ],
)
def test_place_follows_clouds_matching_rule(value, rows, expected):
    assert OwnModels(Library(*rows)).place(value) == expected


def test_place_asks_once_per_file_name_and_never_for_a_non_model():
    lib = Library(UPLOAD)
    own = OwnModels(lib)
    assert own.holds("my_finetune_v1.safetensors")
    assert own.holds("my_finetune_v1.safetensors")
    assert own.place("euler") == ("not_a_model", None, [])
    assert lib.asked == ["my_finetune_v1.safetensors"]


def test_a_failed_lookup_is_recorded_not_raised():
    own = OwnModels(Library(fail=OSError("connection reset")))
    assert own.place("my_finetune_v1.safetensors") == ("unknown", None, [])
    assert "connection reset" in own.refusal_note("my_finetune_v1.safetensors")


def test_for_cloud_resolves_credentials_only_on_the_first_lookup():
    target = object()
    with (
        patch("comfy_cli.target.resolve_target", return_value=target) as resolve,
        patch("comfy_cli.command.workflow_check._fetch_model_assets", return_value=[UPLOAD]) as fetch,
    ):
        own = cloud_own_models.for_cloud()
        resolve.assert_not_called()
        assert own.holds("my_finetune_v1.safetensors")
        assert own.holds("other.safetensors") is False
    resolve.assert_called_once_with(where="cloud")
    assert fetch.call_args_list[0].args == (target, "my_finetune_v1.safetensors")
    assert fetch.call_args_list[0].kwargs == {"include_public": False}


# --- Graph.admit_combo_value ----------------------------------------------------


def test_admit_combo_value_widens_one_graph_only():
    info = json.loads(json.dumps(OBJECT_INFO))
    graph = Graph.from_object_info(info)
    assert graph.admit_combo_value("CheckpointLoaderSimple", "ckpt_name", "my_finetune_v1.safetensors")
    assert graph.validate_workflow(_workflow("my_finetune_v1.safetensors"))["valid"]
    # The catalog payload, and so a fresh Graph over it, is untouched.
    assert info["CheckpointLoaderSimple"]["input"]["required"]["ckpt_name"][0] == CLOUD_CKPTS
    assert not Graph.from_object_info(info).validate_workflow(_workflow("my_finetune_v1.safetensors"))["valid"]


def test_admit_combo_value_refuses_what_is_not_a_combo():
    graph = Graph.from_object_info(OBJECT_INFO)
    assert not graph.admit_combo_value("NoSuchNode", "ckpt_name", "x.safetensors")
    assert not graph.admit_combo_value("CheckpointLoaderSimple", "no_such_input", "x.safetensors")
    assert not graph.admit_combo_value("KSamplerLite", "model", "x.safetensors")


def test_admit_refused_admits_only_held_model_values():
    graph = Graph.from_object_info(OBJECT_INFO)
    wf = _workflow("my_finetune_v1.safetensors", sampler="not_a_sampler")
    errors = graph.validate_workflow(wf)["errors"]
    lib = Library(UPLOAD)
    admitted = admit_refused(graph, wf, errors, OwnModels(lib))
    assert admitted == [
        {
            "node_id": "1",
            "class_type": "CheckpointLoaderSimple",
            "field": "ckpt_name",
            "value": "my_finetune_v1.safetensors",
        }
    ]
    # The sampler is not a model file: never looked up, still refused.
    assert lib.asked == ["my_finetune_v1.safetensors"]
    left = graph.validate_workflow(wf)["errors"]
    assert [(e["node_id"], e["field"]) for e in left] == [("2", "sampler_name")]


def test_note_refusals_says_what_the_library_said():
    graph = Graph.from_object_info(OBJECT_INFO)
    wf = _workflow("ckpt/my_finetune_v1.safetensors")
    errors = graph.validate_workflow(wf)["errors"]
    own = OwnModels(Library(UPLOAD))
    assert admit_refused(graph, wf, errors, own) == []
    note_refusals(wf, errors, own)
    assert "has this file as 'my_finetune_v1.safetensors': set that" in errors[0]["hint"]


# --- run preflight ---------------------------------------------------------------


def test_preflight_accepts_an_upload_the_library_holds():
    renderer = _renderer()
    lib = Library(UPLOAD)
    _preflight_validate(
        renderer, _workflow("my_finetune_v1.safetensors"), OBJECT_INFO, where="cloud", own_models=OwnModels(lib)
    )
    renderer.error.assert_not_called()
    assert lib.asked == ["my_finetune_v1.safetensors"]


def test_preflight_accepts_an_upload_where_cloud_ships_no_option():
    renderer = _renderer()
    wf = {
        "1": {"class_type": "UNETLoader", "inputs": {"unet_name": "mine.safetensors"}},
        "2": {"class_type": "ModelSink", "inputs": {"model": ["1", 0]}},
    }
    _preflight_validate(
        renderer, wf, OBJECT_INFO, where="cloud", own_models=OwnModels(Library({"name": "mine.safetensors"}))
    )
    renderer.error.assert_not_called()


def test_preflight_still_refuses_a_model_nobody_holds():
    renderer = _renderer()
    with pytest.raises(typer.Exit):
        _preflight_validate(
            renderer,
            _workflow("nobody_has_this.safetensors"),
            OBJECT_INFO,
            where="cloud",
            own_models=OwnModels(Library(UPLOAD)),
        )
    err = _refusal(renderer)
    assert err["code"] == "workflow_unknown_nodes"
    (entry,) = err["details"]["errors"]
    assert entry["code"] == "unknown_enum_value"
    assert "no model in your Comfy Cloud model library is named 'nobody_has_this.safetensors'" in entry["hint"]


def test_preflight_refuses_when_the_library_cannot_be_read():
    renderer = _renderer()
    with pytest.raises(typer.Exit):
        _preflight_validate(
            renderer,
            _workflow("my_finetune_v1.safetensors"),
            OBJECT_INFO,
            where="cloud",
            own_models=OwnModels(Library(fail=OSError("timed out"))),
        )
    (entry,) = _refusal(renderer)["details"]["errors"]
    assert "could not check your Comfy Cloud model library: OSError: timed out" in entry["hint"]


def test_preflight_keeps_other_refusals_after_admitting_an_upload():
    renderer = _renderer()
    with pytest.raises(typer.Exit):
        _preflight_validate(
            renderer,
            _workflow("my_finetune_v1.safetensors", sampler="not_a_sampler"),
            OBJECT_INFO,
            where="cloud",
            own_models=OwnModels(Library(UPLOAD)),
        )
    errors = _refusal(renderer)["details"]["errors"]
    assert [(e["node_id"], e["field"]) for e in errors] == [("2", "sampler_name")]


def test_preflight_without_a_library_is_unchanged():
    renderer = _renderer()
    with pytest.raises(typer.Exit):
        _preflight_validate(renderer, _workflow("my_finetune_v1.safetensors"), OBJECT_INFO, where="local")
    (entry,) = _refusal(renderer)["details"]["errors"]
    assert entry["code"] == "unknown_enum_value"
    assert "library" not in entry["hint"]


def test_preflight_never_asks_about_clouds_own_models():
    renderer = _renderer()
    lib = Library(UPLOAD)
    _preflight_validate(renderer, _workflow(CLOUD_CKPTS[0]), OBJECT_INFO, where="cloud", own_models=OwnModels(lib))
    renderer.error.assert_not_called()
    assert lib.asked == []


# --- comfy run --where cloud, end to end ----------------------------------------


def _execute_cloud(tmp_path, ckpt: str, assets: list[dict]):
    from comfy_cli.comfy_client import SubmitResult
    from comfy_cli.command.run import execute_cloud
    from comfy_cli.target import Target

    path = tmp_path / "wf.json"
    path.write_text(json.dumps(_workflow(ckpt)))
    target = Target(
        kind="cloud",
        base_url="https://cloud.example.com",
        path_prefix="/api",
        history_path="history_v2",
        jobs_path="jobs",
        api_key="test-api-key",
    )
    client = MagicMock()
    client.submit_prompt.return_value = SubmitResult(prompt_id="p-1", number=1, node_errors={})
    with (
        patch("comfy_cli.target.resolve_target", return_value=target),
        patch("comfy_cli.cql.engine._load_from_target", return_value=OBJECT_INFO),
        patch("comfy_cli.cql.loader.resilient_load_object_info", return_value=OBJECT_INFO),
        patch("comfy_cli.command.workflow_check._fetch_model_assets", return_value=assets) as fetch,
        patch("comfy_cli.comfy_client.Client", return_value=client),
        patch("comfy_cli.command.run._spawn_watcher"),
        patch("comfy_cli.command.run._tail_state_file"),
    ):
        try:
            execute_cloud(str(path), wait=False)
        except typer.Exit as e:
            return client, fetch, e
    return client, fetch, None


def test_cloud_run_submits_a_workflow_naming_an_upload(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    client, fetch, exited = _execute_cloud(tmp_path, "my_finetune_v1.safetensors", [UPLOAD])
    assert exited is None
    client.submit_prompt.assert_called_once()
    assert fetch.call_args.kwargs == {"include_public": False}


def test_cloud_run_refuses_a_model_nobody_holds(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    client, _fetch, exited = _execute_cloud(tmp_path, "nobody_has_this.safetensors", [])
    assert exited is not None and exited.exit_code == 1
    client.submit_prompt.assert_not_called()


def test_cloud_run_with_clouds_own_model_reads_no_library(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    client, fetch, exited = _execute_cloud(tmp_path, CLOUD_CKPTS[1], [UPLOAD])
    assert exited is None
    client.submit_prompt.assert_called_once()
    fetch.assert_not_called()
