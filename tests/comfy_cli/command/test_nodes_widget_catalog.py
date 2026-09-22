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
        assert types["DynNode"]["widget_order"] == ["model", "model.resolution", "seed", "control_after_generate"]


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
    @pytest.mark.parametrize("mode", ["creative", "faithful", "flexible"])
    def test_regression_magnific_branch_widget_changes_catalog_pin(self, mode, tmp_path, capsys):
        """Non-default modes must not disappear from the pinned catalog.

        Fixture captured from /object_info/MagnificImageSkinEnhancerNode at
        ComfyUI b1693ecba9f5b65f8c80ab36b195ab963ec92413 on 2026-09-22.
        Regression: https://github.com/Comfy-Org/comfy-cli/pull/914
        Creative is the positive control: the existing first-choice projection
        sees its added widget, but misses the same change in the other modes.
        """
        fixture = Path(__file__).parents[1] / "fixtures" / "magnific_skin_enhancer_object_info.json"
        before = _run(["widget-catalog", "--input", str(fixture)], capsys)["data"]
        changed = json.loads(fixture.read_text(encoding="utf-8"))
        options = changed["MagnificImageSkinEnhancerNode"]["input"]["required"]["mode"][1]["options"]
        branch = next(option for option in options if option["key"] == mode)
        branch["inputs"]["required"]["extra_detail"] = ["INT", {"default": 17}]
        dump = tmp_path / "changed_object_info.json"
        dump.write_text(json.dumps(changed), encoding="utf-8")
        after = _run(["widget-catalog", "--input", str(dump)], capsys)["data"]

        assert after["catalog_version"] != before["catalog_version"], mode
        assert after["types"] != before["types"], mode

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


class TestSerializedLayout:
    @pytest.mark.parametrize("width", [16, 64])
    def test_regression_layout_does_not_rescan_sibling_options(self, width):
        # https://github.com/Comfy-Org/comfy-cli/pull/914#discussion_r4068275342
        # Count option-key reads, not elapsed time, so shared CI load cannot
        # turn the quadratic traversal regression into a flaky timing test.
        key_reads = 0

        class CountedOption(dict):
            def get(self, key, default=None):
                nonlocal key_reads
                if key == "key":
                    key_reads += 1
                return super().get(key, default)

        options = [CountedOption(key=f"branch-{i}", inputs={"required": {f"value-{i}": ["INT"]}}) for i in range(width)]
        graph = _graph(
            {
                "WideSelector": {
                    "input": {"required": {"mode": ["COMFY_DYNAMICCOMBO_V3", {"options": options}]}},
                    "output": [],
                }
            }
        )
        key_reads = 0
        layout = graph.widget_layout("WideSelector")

        # A skipped or reordered branch must not count as a faster traversal.
        assert layout == [
            {
                "name": "mode",
                "identity": [["field", "mode"]],
                "options": [
                    {
                        "key": f"branch-{i}",
                        "widgets": [
                            {
                                "name": f"mode.value-{i}",
                                "identity": [["field", "mode"], ["choice", f"branch-{i}"], ["field", f"value-{i}"]],
                            }
                        ],
                    }
                    for i in range(width)
                ],
            }
        ]
        # Allow multiple linear validation passes, but not one sibling scan
        # for every emitted branch (136 and 2080 probes at these widths).
        assert key_reads <= 4 * width

    @pytest.mark.parametrize("nested", [False, True])
    @pytest.mark.parametrize("selected", [False, True])
    def test_regression_structural_combo_orders_include_branch_slots(self, nested, selected):
        # https://github.com/Comfy-Org/comfy-cli/pull/914#discussion_r4068275324
        from comfy_cli.cql.widget_catalog import build_catalog

        data = _object_info()
        required = data["DynNode"]["input"]["required"]
        selector = required["model"]
        selector[0] = "COMBO"
        selector[1]["options"][1]["inputs"]["required"] = {
            "width": ["INT", {"default": 73}],
            "height": ["INT", {"default": 91}],
        }
        if nested:
            required["model"] = [
                "COMFY_DYNAMICCOMBO_V3",
                {"options": [{"key": "outer", "inputs": {"required": {"child": selector}}}]},
            ]
        graph = _graph(data)
        prefix = "model.child" if nested else "model"
        names = ["model", "model.child"] if nested else ["model"]
        names += [f"{prefix}.width", f"{prefix}.height"] if selected else [f"{prefix}.resolution"]
        names += ["seed", "control_after_generate"]

        # The value-independent API intentionally lists no branch fields.
        assert graph.widget_order("DynNode") == ["model", "seed", "control_after_generate"]
        if selected:
            values = (["outer"] if nested else []) + ["b", 73, 91, 37, "fixed"]
            assert graph.widget_order_for_node("DynNode", values) == names
        else:
            catalog = build_catalog(graph)["types"]["DynNode"]
            assert catalog["widget_order"] == names
            assert graph.widget_order_default("DynNode") == names

    def test_magnific_carries_every_branch_and_suffix(self):
        from comfy_cli.cql.widget_catalog import build_catalog

        fixture = Path(__file__).parents[1] / "fixtures" / "magnific_skin_enhancer_object_info.json"
        data = json.loads(fixture.read_text(encoding="utf-8"))
        node = data["MagnificImageSkinEnhancerNode"]
        node["input"]["required"]["suffix"] = ["INT", {"default": 7}]
        node["input_order"]["required"].append("suffix")
        entry = build_catalog(_graph(data))["types"]["MagnificImageSkinEnhancerNode"]

        assert entry["widget_layout"] == [
            {"name": "sharpen", "identity": [["field", "sharpen"]]},
            {"name": "smart_grain", "identity": [["field", "smart_grain"]]},
            {
                "name": "mode",
                "identity": [["field", "mode"]],
                "options": [
                    {"key": "creative", "widgets": []},
                    {
                        "key": "faithful",
                        "widgets": [
                            {
                                "name": "mode.skin_detail",
                                "identity": [["field", "mode"], ["choice", "faithful"], ["field", "skin_detail"]],
                            }
                        ],
                    },
                    {
                        "key": "flexible",
                        "widgets": [
                            {
                                "name": "mode.optimized_for",
                                "identity": [["field", "mode"], ["choice", "flexible"], ["field", "optimized_for"]],
                            }
                        ],
                    },
                ],
            },
            {"name": "suffix", "identity": [["field", "suffix"]]},
        ]

    def test_nested_branch_names_and_seed_companions_have_distinct_identity(self):
        from comfy_cli.cql.widget_catalog import build_catalog

        data = _object_info()
        options = data["DynNode"]["input"]["required"]["model"][1]["options"]
        options[0]["inputs"]["required"] = {
            "detail": [
                "COMFY_DYNAMICCOMBO_V3",
                {
                    "options": [
                        {
                            "key": "fine",
                            "inputs": {
                                "required": {
                                    "seed": ["INT", {"default": 11}],
                                    "noise_seed": ["INT", {"default": 29}],
                                    "resolution": ["INT", {"default": 7}],
                                }
                            },
                        }
                    ]
                },
            ],
        }
        options[1]["inputs"]["required"] = {"detail": ["STRING", {"default": "other"}]}
        layout = build_catalog(_graph(data))["types"]["DynNode"]["widget_layout"]
        branch_a, branch_b = layout[0]["options"]
        detail = branch_a["widgets"][0]
        assert detail["name"] == branch_b["widgets"][0]["name"] == "model.detail"
        assert detail["identity"] == [["field", "model"], ["choice", "a"], ["field", "detail"]]
        assert branch_b["widgets"][0]["identity"] == [["field", "model"], ["choice", "b"], ["field", "detail"]]
        widgets = detail["options"][0]["widgets"]
        assert [w["name"] for w in widgets] == [
            "model.detail.seed",
            "control_after_generate",
            "model.detail.noise_seed",
            "control_after_generate",
            "model.detail.resolution",
        ]
        assert [w["identity"] for w in widgets[1:4:2]] == [
            [
                ["field", "model"],
                ["choice", "a"],
                ["field", "detail"],
                ["choice", "fine"],
                ["field", "seed"],
                ["companion", "control_after_generate"],
            ],
            [
                ["field", "model"],
                ["choice", "a"],
                ["field", "detail"],
                ["choice", "fine"],
                ["field", "noise_seed"],
                ["companion", "control_after_generate"],
            ],
        ]

    @pytest.mark.parametrize(
        "class_name,inputs,expected",
        [
            ("LoadImage", {"image": [["x.png"], {"image_upload": True}]}, ["image"]),
            ("LoadAudio", {"audio": [["x.wav"], {"audio_upload": True}]}, ["audio"]),
            ("SaveGLB", {"filename_prefix": ["STRING"]}, ["filename_prefix", "image"]),
        ],
    )
    def test_only_serialized_frontend_slots_participate(self, class_name, inputs, expected):
        from comfy_cli.cql.widget_catalog import build_catalog

        data = {class_name: {"input": {"required": inputs}, "output": []}}
        layout = build_catalog(_graph(data))["types"][class_name]["widget_layout"]
        assert [w["name"] for w in layout] == expected

    @pytest.mark.parametrize("declared_image", [False, True])
    def test_regression_injected_image_is_read_only_but_declared_image_is_not(self, declared_image):
        # https://github.com/Comfy-Org/comfy-cli/pull/914#discussion_r4068275348
        # The same serialized identity can name an injected viewport or a
        # schema-backed field. Consumers cannot infer permission from its name.
        import jsonschema

        from comfy_cli.cql.widget_catalog import build_catalog

        inputs = {"filename_prefix": ["STRING"]}
        if declared_image:
            inputs["image"] = ["STRING"]
        catalog = build_catalog(_graph({"SaveGLB": {"input": {"required": inputs}, "output": []}}))
        expected_image = {"name": "image", "identity": [["field", "image"]]}
        if not declared_image:
            expected_image["read_only"] = True
        assert catalog["types"]["SaveGLB"]["widget_layout"] == [
            {"name": "filename_prefix", "identity": [["field", "filename_prefix"]]},
            expected_image,
        ]
        schema = json.loads(
            (Path(nodes_cmd.__file__).resolve().parents[1] / "schemas" / "widget_catalog.json").read_text()
        )
        jsonschema.Draft202012Validator(schema).validate(catalog)

    def test_regression_injected_buttons_are_read_only_but_seed_companion_is_not(self):
        # https://github.com/Comfy-Org/comfy-cli/pull/914#discussion_r4068275348
        # Both use companion identities, but only the seed control is writable.
        import jsonschema

        from comfy_cli.cql.widget_catalog import build_catalog

        inputs = {"model_file": [["scene.glb"]], "image": ["LOAD_3D"], "seed": ["INT"]}
        catalog = build_catalog(
            _graph(
                {
                    "Load3D": {
                        "input": {"required": inputs},
                        "input_order": {"required": ["model_file", "image", "seed"]},
                        "output": [],
                    }
                }
            )
        )
        layout = catalog["types"]["Load3D"]["widget_layout"]
        assert [(w["name"], w.get("read_only", False)) for w in layout] == [
            ("model_file", False),
            ("upload 3d model", True),
            ("upload extra resources", True),
            ("clear", True),
            ("image", False),
            ("seed", False),
            ("control_after_generate", False),
        ]
        assert [w["identity"] for w in layout[1:4]] == [
            [["field", "image"], ["companion", "upload 3d model"]],
            [["field", "image"], ["companion", "upload extra resources"]],
            [["field", "image"], ["companion", "clear"]],
        ]
        assert layout[-1]["identity"] == [["field", "seed"], ["companion", "control_after_generate"]]
        schema = json.loads(
            (Path(nodes_cmd.__file__).resolve().parents[1] / "schemas" / "widget_catalog.json").read_text()
        )
        jsonschema.Draft202012Validator(schema).validate(catalog)

    @pytest.mark.parametrize("keys", [[1, True], ["a", "a"]])
    def test_unsupported_selector_keys_fail_with_class_and_field(self, keys):
        from comfy_cli.cql.widget_catalog import build_catalog

        data = _object_info()
        options = data["DynNode"]["input"]["required"]["model"][1]["options"]
        for option, key in zip(options, keys):
            option["key"] = key
        with pytest.raises(ValueError, match=r"DynNode.*model.*unique string"):
            build_catalog(_graph(data))

    @pytest.mark.parametrize("type_name", ["COMFY_DYNAMICCOMBO_V3", "COMBO"])
    @pytest.mark.parametrize("nested", [False, True])
    @pytest.mark.parametrize("malformed", [{"inputs": {"required": {"lost": ["INT"]}}}, None])
    def test_regression_malformed_selector_branches_cannot_disappear(self, type_name, nested, malformed):
        # https://github.com/Comfy-Org/comfy-cli/pull/914#discussion_r4068215356
        from comfy_cli.cql.widget_catalog import build_catalog

        data = _object_info()
        selector = [type_name, {"options": [{"key": "valid", "inputs": {}}, malformed]}]
        required = data["DynNode"]["input"]["required"]
        if nested:
            required["model"][1]["options"][1]["inputs"]["required"] = {"child": selector}
        else:
            required["model"] = selector
        field = r"model\.child" if nested else "model"
        with pytest.raises(ValueError, match=rf"DynNode.*{field}.*unique string"):
            build_catalog(_graph(data))

    @pytest.mark.parametrize("nested", [False, True])
    @pytest.mark.parametrize("declaration", [None, {}, {"options": None}, {"options": "remote"}, {"options": []}])
    def test_regression_unavailable_selector_choices_are_not_declared_empty(self, nested, declaration):
        # https://github.com/Comfy-Org/comfy-cli/pull/914#discussion_r4068275339
        from comfy_cli.cql.widget_catalog import build_catalog

        data = _object_info()
        selector = ["COMFY_DYNAMICCOMBO_V3"]
        if declaration is not None:
            selector.append(declaration)
        required = data["DynNode"]["input"]["required"]
        if nested:
            required["model"][1]["options"][1]["inputs"]["required"] = {"child": selector}
        else:
            required["model"] = selector

        if declaration != {"options": []}:
            field = r"model\.child" if nested else "model"
            with pytest.raises(ValueError, match=rf"DynNode.*{field}.*explicit options list"):
                build_catalog(_graph(data))
        else:
            layout = build_catalog(_graph(data))["types"]["DynNode"]["widget_layout"]
            entry = layout[0]["options"][1]["widgets"][0] if nested else layout[0]
            assert entry == {
                "name": "model.child" if nested else "model",
                "identity": [["field", "model"], ["choice", "b"], ["field", "child"]]
                if nested
                else [["field", "model"]],
                "options": [],
            }
