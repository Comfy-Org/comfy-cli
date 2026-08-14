"""Tests for ``comfy nodes widget-catalog`` (the widget-catalog producer).

WHY THIS COMMAND EXISTS: the CRDT doc host (cloud ``services/agent/dochost``)
and the applier (``@comfyorg/comfy-multi-player``) convert between the CRDT
doc's NAME-keyed widget maps and the workflow JSON's POSITIONAL
``widgets_values`` array. That conversion needs one derived projection of
``object_info`` — ``{types: {<class_type>: {widget_order, autogrow_templates}}}``
— and the widget order it needs is exactly what ``cql.engine.Graph`` already
computes for every edit primitive in this CLI. Emitting it here (rather than
recomputing it in Go) keeps a single source of ComfyUI widget semantics.

THE CONTRACT IS THE ENGINE: for every class, ``types[c].widget_order`` must be
byte-identical to ``Graph.widget_order(c)``. If those two ever diverge, the
applier writes a widget value into the wrong index and the user's canvas
silently corrupts — so the tests below assert equality against the engine
itself, never against a hand-written expectation.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from comfy_cli.caller import Caller
from comfy_cli.command import nodes as nodes_cmd
from comfy_cli.cql.engine import Graph
from comfy_cli.output.renderer import OutputMode, Renderer, reset_renderer_for_testing, set_renderer


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


# ---------------------------------------------------------------------------
# Fixture object_info — one class per interesting widget-order shape.
# ---------------------------------------------------------------------------


def _object_info() -> dict[str, Any]:
    return {
        # control_after_generate: the engine injects a synthetic widget right
        # after the seed, so a naive "list the non-link inputs" projection
        # mis-indexes every widget after it.
        "KSampler": {
            "input": {
                "required": {
                    "model": ["MODEL"],
                    "seed": ["INT", {"default": 0, "control_after_generate": True}],
                    "steps": ["INT", {"default": 20}],
                    "cfg": ["FLOAT", {"default": 8.0}],
                    "sampler_name": [["euler", "dpmpp_2m"]],
                    "denoise": ["FLOAT", {"default": 1.0}],
                }
            },
            "input_order": {"required": ["model", "seed", "steps", "cfg", "sampler_name", "denoise"]},
            "output": ["LATENT"],
            "output_name": ["LATENT"],
            "category": "sampling",
            "display_name": "KSampler",
            "python_module": "nodes",
        },
        "CLIPTextEncode": {
            "input": {"required": {"text": ["STRING", {"multiline": True}], "clip": ["CLIP"]}},
            "input_order": {"required": ["text", "clip"]},
            "output": ["CONDITIONING"],
            "output_name": ["CONDITIONING"],
            "category": "conditioning",
            "display_name": "CLIP Text Encode",
            "python_module": "nodes",
        },
        # Zero widgets — a real, load-bearing state. The applier must be able to
        # tell "this class has no widgets" from "this class is unknown".
        "VAEDecode": {
            "input": {"required": {"samples": ["LATENT"], "vae": ["VAE"]}},
            "input_order": {"required": ["samples", "vae"]},
            "output": ["IMAGE"],
            "output_name": ["IMAGE"],
            "category": "latent",
            "display_name": "VAE Decode",
            "python_module": "nodes",
        },
        # V3 autogrow WITH a schema-declared naming template.
        "BatchImagesNode": {
            "input": {
                "required": {
                    "images": ["COMFY_AUTOGROW_V3", {"template": {"prefix": "image", "min": 1, "max": 50}}],
                }
            },
            "input_order": {"required": ["images"]},
            "output": ["IMAGE"],
            "output_name": ["IMAGE"],
            "category": "image",
            "display_name": "Batch Images",
            "python_module": "nodes",
        },
        # V3 autogrow with NO template — the catalog must still say the input is
        # autogrow, falling back to the same pluralization the edit path uses.
        "UntemplatedGrowNode": {
            "input": {"required": {"masks": ["COMFY_AUTOGROW_V3"]}},
            "input_order": {"required": ["masks"]},
            "output": ["MASK"],
            "output_name": ["MASK"],
            "category": "mask",
            "display_name": "Untemplated Grow",
            "python_module": "nodes",
        },
        # kijai `inputcount` family: NOT autogrow-typed; fixed `{elem}_N` inputs
        # plus an INT `inputcount` widget the node reads at runtime.
        "ImageBatchMulti": {
            "input": {
                "required": {
                    "inputcount": ["INT", {"default": 2, "min": 2, "max": 1000}],
                    "image_1": ["IMAGE"],
                    "image_2": ["IMAGE"],
                }
            },
            "input_order": {"required": ["inputcount", "image_1", "image_2"]},
            "output": ["IMAGE"],
            "output_name": ["IMAGE"],
            "category": "image",
            "display_name": "Image Batch Multi",
            "python_module": "custom_nodes.KJNodes",
        },
        # Dynamic combo: the selector expands key-dependent sub-widgets, and the
        # catalog must carry the expanded order (model, model.resolution, seed).
        "DynNode": {
            "input": {
                "required": {
                    "model": [
                        "COMFY_DYNAMICCOMBO_V3",
                        {
                            "options": [
                                {"key": "a", "inputs": {"required": {"resolution": ["INT", {"default": 512}]}}},
                                {"key": "b", "inputs": {"required": {}}},
                            ]
                        },
                    ],
                    "seed": ["INT", {"default": 0}],
                }
            },
            "input_order": {"required": ["model", "seed"]},
            "output": ["IMAGE"],
            "output_name": ["IMAGE"],
            "category": "api node",
            "display_name": "Dyn Node",
            "python_module": "nodes",
        },
    }


def _graph(data: dict[str, Any] | None = None) -> Graph:
    return Graph.from_object_info(data if data is not None else _object_info())


@pytest.fixture
def patched_loader(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(nodes_cmd, "_get_graph", lambda *a, **kw: _graph())


def _run(args: list[str], capsys) -> dict[str, Any]:
    _force_json_renderer()
    runner = CliRunner()
    result = runner.invoke(nodes_cmd.app, args, standalone_mode=False)
    captured = capsys.readouterr().out
    if not captured.strip():
        captured = result.stdout or ""
    for line in reversed(captured.strip().splitlines()):
        try:
            return json.loads(line)
        except json.JSONDecodeError:
            continue
    raise AssertionError(f"no JSON envelope (rc={result.exit_code}, exc={result.exception}, out={captured[:600]})")


# ---------------------------------------------------------------------------
# widget_order — graded against the engine, class by class
# ---------------------------------------------------------------------------


class TestWidgetOrder:
    def test_every_class_matches_the_engine(self, patched_loader, capsys):
        env = _run(["widget-catalog"], capsys)
        assert env["ok"] is True
        types = env["data"]["types"]
        graph = _graph()
        assert set(types) == {m.id for m in graph.all_nodes()}
        for class_type, entry in types.items():
            # The catalog publishes the FRESH-node order (dynamic combos expanded
            # at their first key), which is what a consumer can address before it
            # has a node to read a selection from.
            assert entry["widget_order"] == graph.widget_order_default(class_type), class_type

    def test_control_after_generate_is_in_the_order(self, patched_loader, capsys):
        """The synthetic widget the frontend injects after a seed occupies a
        real `widgets_values` slot — omitting it shifts every later index."""
        types = _run(["widget-catalog"], capsys)["data"]["types"]
        assert types["KSampler"]["widget_order"] == [
            "seed",
            "control_after_generate",
            "steps",
            "cfg",
            "sampler_name",
            "denoise",
        ]

    def test_link_only_class_keeps_an_empty_order(self, patched_loader, capsys):
        types = _run(["widget-catalog"], capsys)["data"]["types"]
        assert types["VAEDecode"]["widget_order"] == []
        assert "VAEDecode" in types, "a widget-less class must still be present, not dropped"

    def test_dynamic_combo_sub_widgets_expand(self, patched_loader, capsys):
        types = _run(["widget-catalog"], capsys)["data"]["types"]
        assert types["DynNode"]["widget_order"] == ["model", "model.resolution", "seed"]


# ---------------------------------------------------------------------------
# autogrow / inputcount families
# ---------------------------------------------------------------------------


class TestGrowFamilies:
    def test_schema_declared_autogrow_template(self, patched_loader, capsys):
        types = _run(["widget-catalog"], capsys)["data"]["types"]
        assert types["BatchImagesNode"]["autogrow_templates"] == {"images": {"prefix": "image"}}

    def test_untemplated_autogrow_falls_back_to_the_edit_paths_naming(self, patched_loader, capsys):
        """No template in object_info still means "this input autogrows" — the
        catalog says so, using the same singularization `_autogrow_elem_name`
        applies when the schema is silent."""
        types = _run(["widget-catalog"], capsys)["data"]["types"]
        assert types["UntemplatedGrowNode"]["autogrow_templates"] == {"masks": {"prefix": "mask"}}

    def test_non_growing_class_carries_no_template_key(self, patched_loader, capsys):
        types = _run(["widget-catalog"], capsys)["data"]["types"]
        assert "autogrow_templates" not in types["KSampler"]

    def test_inputcount_family_is_reported(self, patched_loader, capsys):
        types = _run(["widget-catalog"], capsys)["data"]["types"]
        assert types["ImageBatchMulti"]["inputcount"] == {"widget": "inputcount", "elements": ["image"]}
        assert "inputcount" not in types["BatchImagesNode"], "autogrow is a different family"


# ---------------------------------------------------------------------------
# catalog_version
# ---------------------------------------------------------------------------


class TestCatalogVersion:
    def test_stable_across_runs_for_identical_input(self, patched_loader, capsys):
        first = _run(["widget-catalog"], capsys)["data"]
        second = _run(["widget-catalog"], capsys)["data"]
        assert first == second
        assert first["catalog_version"] == second["catalog_version"]
        assert first["catalog_version"].startswith("sha256:")
        assert len(first["catalog_version"]) == len("sha256:") + 64

    def test_changes_when_the_input_changes(self, monkeypatch, capsys):
        base = _object_info()
        monkeypatch.setattr(nodes_cmd, "_get_graph", lambda *a, **kw: _graph(base))
        before = _run(["widget-catalog"], capsys)["data"]["catalog_version"]

        drifted = _object_info()
        # One extra widget on one class — the smallest change that must move the
        # version, because it moves every later widget's index.
        drifted["KSampler"]["input"]["required"]["scheduler"] = [["normal", "karras"]]
        drifted["KSampler"]["input_order"]["required"].insert(4, "scheduler")
        monkeypatch.setattr(nodes_cmd, "_get_graph", lambda *a, **kw: _graph(drifted))
        after = _run(["widget-catalog"], capsys)["data"]["catalog_version"]

        assert after != before

    def test_version_is_independent_of_class_iteration_order(self, monkeypatch, capsys):
        """Reordering object_info's keys is not a catalog change — a pin that
        flapped on dict order would be useless as a cache key."""
        base = _object_info()
        monkeypatch.setattr(nodes_cmd, "_get_graph", lambda *a, **kw: _graph(base))
        before = _run(["widget-catalog"], capsys)["data"]["catalog_version"]

        shuffled = dict(reversed(list(_object_info().items())))
        monkeypatch.setattr(nodes_cmd, "_get_graph", lambda *a, **kw: _graph(shuffled))
        after = _run(["widget-catalog"], capsys)["data"]["catalog_version"]

        assert after == before

    def test_version_excludes_itself_and_the_class_count(self, patched_loader, capsys):
        """The hash covers the `types` map only, so a consumer can recompute it
        from the catalog it stored without carrying the envelope metadata."""
        import hashlib

        data = _run(["widget-catalog"], capsys)["data"]
        canonical = json.dumps(data["types"], sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        assert data["catalog_version"] == "sha256:" + hashlib.sha256(canonical.encode("utf-8")).hexdigest()
        assert data["class_count"] == len(data["types"])


# ---------------------------------------------------------------------------
# offline + projection
# ---------------------------------------------------------------------------


class TestOfflineAndSelect:
    def test_offline_via_input_dump(self, tmp_path, capsys):
        dump = tmp_path / "object_info.json"
        dump.write_text(json.dumps(_object_info()), encoding="utf-8")
        env = _run(["widget-catalog", "--input", str(dump)], capsys)
        assert env["ok"] is True
        assert env["data"]["types"]["KSampler"]["widget_order"][0] == "seed"

    def test_offline_via_comfy_object_info_file_env(self, tmp_path, monkeypatch, capsys):
        """The hermetic path the agent's sandbox uses: no --input, no server, no
        credential — just the baked dump every other object_info consumer reads."""
        dump = tmp_path / "object_info.json"
        dump.write_text(json.dumps(_object_info()), encoding="utf-8")
        monkeypatch.setenv("COMFY_OBJECT_INFO_FILE", str(dump))
        monkeypatch.setattr(
            "comfy_cli.cql.engine._load_from_target",
            lambda **_: (_ for _ in ()).throw(AssertionError("must not touch the network")),
        )
        env = _run(["widget-catalog"], capsys)
        assert env["ok"] is True
        assert env["data"]["types"]["BatchImagesNode"]["autogrow_templates"] == {"images": {"prefix": "image"}}

    def test_select_projects_the_payload(self, patched_loader, capsys):
        env = _run(["widget-catalog", "--select", "catalog_version"], capsys)
        assert env["ok"] is True
        assert isinstance(env["data"], str) and env["data"].startswith("sha256:")


class TestSchemaContract:
    def test_payload_validates_against_the_registered_schema(self, patched_loader, capsys):
        """`comfy discover` hands agents this schema; the payload has to match it."""
        import jsonschema

        from comfy_cli.discovery import COMMAND_SCHEMAS

        assert COMMAND_SCHEMAS["comfy nodes widget-catalog"] == "widget_catalog"
        schema = json.loads(
            (Path(nodes_cmd.__file__).resolve().parents[1] / "schemas" / "widget_catalog.json").read_text()
        )
        jsonschema.Draft202012Validator.check_schema(schema)
        jsonschema.Draft202012Validator(schema).validate(_run(["widget-catalog"], capsys)["data"])
