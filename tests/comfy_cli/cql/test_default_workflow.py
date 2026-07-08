"""Offline unit tests for the bundled default text2img workflow injector.

No server, no object_info, no apply_slots — the injector is a pure direct
API-format write over the pinned graph (BE-2535).
"""

from __future__ import annotations

import pytest

from comfy_cli.cql.default_workflow import (
    CHECKPOINT_LOADER_ID,
    NEGATIVE_PROMPT_ID,
    POSITIVE_PROMPT_ID,
    PromptInjectionError,
    build_default_workflow,
    load_default_workflow,
)


class TestBundledGraph:
    def test_loads_expected_seven_nodes(self):
        wf = load_default_workflow()
        classes = sorted(n["class_type"] for n in wf.values())
        assert classes == [
            "CLIPTextEncode",
            "CLIPTextEncode",
            "CheckpointLoaderSimple",
            "EmptyLatentImage",
            "KSampler",
            "SaveImage",
            "VAEDecode",
        ]

    def test_is_api_format_not_subgraphed(self):
        wf = load_default_workflow()
        # API format: every value is a node dict with class_type + inputs; no
        # UI-format `nodes`/`links` keys, no subgraph wrappers.
        assert "nodes" not in wf and "links" not in wf
        for node in wf.values():
            assert "class_type" in node
            assert isinstance(node["inputs"], dict)

    def test_pinned_ids_point_at_expected_classes(self):
        wf = load_default_workflow()
        assert wf[CHECKPOINT_LOADER_ID]["class_type"] == "CheckpointLoaderSimple"
        assert wf[POSITIVE_PROMPT_ID]["class_type"] == "CLIPTextEncode"
        assert wf[NEGATIVE_PROMPT_ID]["class_type"] == "CLIPTextEncode"

    def test_fresh_copy_each_call(self):
        a = load_default_workflow()
        a[POSITIVE_PROMPT_ID]["inputs"]["text"] = "mutated"
        b = load_default_workflow()
        assert b[POSITIVE_PROMPT_ID]["inputs"]["text"] == ""


class TestPromptInjection:
    def test_prompt_sets_positive_text_only(self):
        wf = build_default_workflow(prompt="a red fox in snow")
        assert wf[POSITIVE_PROMPT_ID]["inputs"]["text"] == "a red fox in snow"
        # Negative prompt is left untouched.
        assert wf[NEGATIVE_PROMPT_ID]["inputs"]["text"] == ""

    def test_no_prompt_no_overrides_is_default_graph(self):
        wf = build_default_workflow()
        assert wf[POSITIVE_PROMPT_ID]["inputs"]["text"] == ""
        assert wf[CHECKPOINT_LOADER_ID]["inputs"]["ckpt_name"] == "v1-5-pruned-emaonly.ckpt"


class TestSetOverrides:
    def test_checkpoint_alias(self):
        wf = build_default_workflow(overrides=["checkpoint=sd_xl.safetensors"])
        assert wf[CHECKPOINT_LOADER_ID]["inputs"]["ckpt_name"] == "sd_xl.safetensors"

    def test_raw_node_field_form(self):
        wf = build_default_workflow(overrides=["4.ckpt_name=sd_xl.safetensors"])
        assert wf[CHECKPOINT_LOADER_ID]["inputs"]["ckpt_name"] == "sd_xl.safetensors"

    def test_alias_and_raw_form_are_equivalent(self):
        a = build_default_workflow(overrides=["checkpoint=model.ckpt"])
        b = build_default_workflow(overrides=["4.ckpt_name=model.ckpt"])
        assert a == b

    def test_negative_alias(self):
        wf = build_default_workflow(overrides=["negative=blurry, low quality"])
        assert wf[NEGATIVE_PROMPT_ID]["inputs"]["text"] == "blurry, low quality"

    def test_seed_coerced_to_int(self):
        wf = build_default_workflow(overrides=["seed=42"])
        assert wf["3"]["inputs"]["seed"] == 42
        assert isinstance(wf["3"]["inputs"]["seed"], int)

    def test_cfg_coerced_to_float(self):
        wf = build_default_workflow(overrides=["cfg=7.5"])
        assert wf["3"]["inputs"]["cfg"] == 7.5
        assert isinstance(wf["3"]["inputs"]["cfg"], float)

    def test_width_height_coerced_to_int(self):
        wf = build_default_workflow(overrides=["width=768", "height=1024"])
        assert wf["5"]["inputs"]["width"] == 768
        assert wf["5"]["inputs"]["height"] == 1024

    def test_later_override_wins(self):
        wf = build_default_workflow(overrides=["seed=1", "seed=2"])
        assert wf["3"]["inputs"]["seed"] == 2

    def test_prompt_and_set_combine(self):
        wf = build_default_workflow(prompt="fox", overrides=["negative=blurry", "seed=9"])
        assert wf[POSITIVE_PROMPT_ID]["inputs"]["text"] == "fox"
        assert wf[NEGATIVE_PROMPT_ID]["inputs"]["text"] == "blurry"
        assert wf["3"]["inputs"]["seed"] == 9

    def test_value_containing_equals_sign(self):
        wf = build_default_workflow(overrides=["negative=a=b"])
        assert wf[NEGATIVE_PROMPT_ID]["inputs"]["text"] == "a=b"


class TestErrors:
    def test_unknown_alias(self):
        with pytest.raises(PromptInjectionError) as e:
            build_default_workflow(overrides=["bogus=x"])
        assert e.value.code == "prompt_rejected"

    def test_unknown_node_id_raw_form(self):
        with pytest.raises(PromptInjectionError) as e:
            build_default_workflow(overrides=["99.foo=1"])
        assert e.value.code == "prompt_rejected"

    def test_missing_equals(self):
        with pytest.raises(PromptInjectionError) as e:
            build_default_workflow(overrides=["seed"])
        assert e.value.code == "prompt_rejected"

    def test_non_integer_for_int_field(self):
        with pytest.raises(PromptInjectionError) as e:
            build_default_workflow(overrides=["seed=notanumber"])
        assert e.value.code == "prompt_rejected"

    def test_empty_field_in_raw_form(self):
        with pytest.raises(PromptInjectionError):
            build_default_workflow(overrides=["4.=x"])

    def test_unknown_raw_field_is_rejected(self):
        # A typo in the raw form must fail fast rather than silently write a junk
        # key while the real input keeps its default.
        with pytest.raises(PromptInjectionError) as e:
            build_default_workflow(overrides=["4.ckpt_naem=x"])
        assert e.value.code == "prompt_rejected"

    def test_connection_edge_target_is_rejected(self):
        # `3.positive` holds a wired connection (["6", 0]); overwriting it with a
        # scalar would corrupt the graph topology.
        with pytest.raises(PromptInjectionError):
            build_default_workflow(overrides=["3.positive=cat"])

    def test_rewire_attempt_is_rejected(self):
        with pytest.raises(PromptInjectionError):
            build_default_workflow(overrides=['9.images=["8", 0]'])

    @pytest.mark.parametrize("bad", ["cfg=nan", "cfg=inf", "cfg=-inf", "cfg=Infinity"])
    def test_non_finite_float_is_rejected(self, bad):
        with pytest.raises(PromptInjectionError):
            build_default_workflow(overrides=[bad])


class TestBundleUnavailable:
    def test_missing_bundle_surfaces_controlled_error(self, monkeypatch):
        import comfy_cli.cql.default_workflow as mod

        def boom(*_a, **_k):
            raise FileNotFoundError("no such resource")

        monkeypatch.setattr(mod.resources, "files", boom)
        with pytest.raises(PromptInjectionError) as e:
            build_default_workflow(prompt="fox")
        assert e.value.code == "default_workflow_unavailable"

    def test_corrupt_bundle_surfaces_controlled_error(self, monkeypatch):
        import comfy_cli.cql.default_workflow as mod

        monkeypatch.setattr(mod.json, "loads", lambda *_a, **_k: (_ for _ in ()).throw(ValueError("bad json")))
        with pytest.raises(PromptInjectionError) as e:
            load_default_workflow()
        assert e.value.code == "default_workflow_unavailable"
