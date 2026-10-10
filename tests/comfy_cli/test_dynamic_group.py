import copy
import json
from pathlib import Path

import pytest

from comfy_cli import workflow_ops, workflow_to_api
from comfy_cli.cql.engine import Graph
from comfy_cli.cql.widget_catalog import build_catalog

FIXTURE = json.loads((Path(__file__).parent / "fixtures" / "dynamic_group.json").read_text())


@pytest.mark.parametrize("case_name", ["empty", "populated"])
def test_dynamic_group_conversion_matches_frontend(case_name):
    case = FIXTURE["cases"][case_name]

    prompt = workflow_to_api.convert_ui_to_api(case["workflow"], FIXTURE["object_info"])

    assert prompt["1"]["inputs"] == case["expected_inputs"]


def test_dynamic_group_field_edit_preserves_other_widgets():
    case = FIXTURE["cases"]["populated"]
    graph = Graph.from_object_info(FIXTURE["object_info"])

    edited, _ = workflow_ops.set_widget(copy.deepcopy(case["workflow"]), graph, 1, "loras.1.strength", 0.7)
    prompt = workflow_to_api.convert_ui_to_api(edited, FIXTURE["object_info"])

    assert prompt["1"]["inputs"] == {**case["expected_inputs"], "loras.1.strength": 0.7}
    assert edited["nodes"][0]["widgets_values_named"]["loras.1.strength"] == 0.7


def test_named_widget_values_keep_group_fields_when_edited():
    case = FIXTURE["cases"]["populated"]
    workflow = copy.deepcopy(case["workflow"])
    workflow["nodes"][0]["widgets_values"] = workflow["nodes"][0]["widgets_values_named"].copy()
    assert workflow_to_api.convert_ui_to_api(workflow, FIXTURE["object_info"])["1"]["inputs"] == case["expected_inputs"]
    graph = Graph.from_object_info(FIXTURE["object_info"])
    edited, _ = workflow_ops.set_widget(workflow, graph, 1, "loras.1.strength", 0.7)
    assert workflow_to_api.convert_ui_to_api(edited, FIXTURE["object_info"])["1"]["inputs"] == {
        **case["expected_inputs"],
        "loras.1.strength": 0.7,
    }


def test_dynamic_group_resize_preserves_surviving_rows_and_trailing_widgets():
    case = FIXTURE["cases"]["populated"]
    graph = Graph.from_object_info(FIXTURE["object_info"])
    workflow = copy.deepcopy(case["workflow"])

    grown, op = workflow_ops.set_widget(workflow, graph, 1, "loras", 3)
    replayed = workflow_ops.apply_op(copy.deepcopy(case["workflow"]), op, graph)
    expected = {
        **case["expected_inputs"],
        "loras.2.lora_name": "A.safetensors",
        "loras.2.strength": 1.0,
        "loras.2.enabled": True,
    }
    assert workflow_to_api.convert_ui_to_api(grown, FIXTURE["object_info"])["1"]["inputs"] == expected
    assert workflow_ops.canonical(replayed) == workflow_ops.canonical(grown)

    shrunk, _ = workflow_ops.set_widget(grown, graph, 1, "loras", 0, base_version=1)
    node = shrunk["nodes"][0]
    assert node["widgets_values"] == ["head", 0, "tail"]
    assert node["widgets_values_named"] == {"before": "head", "loras": 0, "after": "tail"}
    assert not any(slot["name"].startswith("loras.") for slot in node["inputs"])
    assert workflow_to_api.convert_ui_to_api(shrunk, FIXTURE["object_info"])["1"]["inputs"] == {
        "before": "head",
        "after": "tail",
    }


@pytest.mark.parametrize("count", [-1, 4, True, 1.5])
def test_dynamic_group_resize_rejects_invalid_count(count):
    workflow = copy.deepcopy(FIXTURE["cases"]["populated"]["workflow"])
    graph = Graph.from_object_info(FIXTURE["object_info"])
    with pytest.raises(ValueError):
        workflow_ops.set_widget(workflow, graph, 1, "loras", count)


def test_dynamic_group_catalog_describes_row_layout_and_defaults():
    graph = Graph.from_object_info(FIXTURE["object_info"])
    node = build_catalog(graph)["types"]["DevToolsNodeWithDynamicGroup"]
    assert node["widget_order"] == ["before", "loras", "after"]
    assert node["dynamic_groups"] == {
        "loras": {
            "min": 0,
            "max": 3,
            "widgets": ["lora_name", "strength", "enabled"],
            "defaults": {"lora_name": "A.safetensors", "strength": 1.0, "enabled": True},
        }
    }


def test_dynamic_group_discovery_and_slots_expose_editable_fields():
    graph = Graph.from_object_info(FIXTURE["object_info"])
    details = graph.morphism_to_dict(graph.node("DevToolsNodeWithDynamicGroup"))
    group = next(p for p in details["inputs"] if p["name"] == "loras")
    assert not group["is_link"]
    assert group["dynamic_group"]["template"]["required"]["strength"][0] == "FLOAT"
    slots = graph.get_template_schema("example", FIXTURE["cases"]["populated"]["workflow"])["slots"]
    values = {s["address"]: s["current_value"] for s in slots}
    assert values["1.loras.1.strength"] == 0.5
    assert values["1.after"] == "tail"


def test_dynamic_group_fresh_node_seeds_minimum_rows():
    info = copy.deepcopy(FIXTURE["object_info"])
    info["DevToolsNodeWithDynamicGroup"]["input"]["required"]["loras"][1]["min"] = 1
    graph = Graph.from_object_info(info)
    workflow, _ = workflow_ops.add_node({"nodes": [], "links": []}, graph, "DevToolsNodeWithDynamicGroup")
    prompt = workflow_to_api.convert_ui_to_api(workflow, info)
    assert next(iter(prompt.values()))["inputs"] == {
        "before": "first",
        "after": "last",
        "loras.0.lora_name": "A.safetensors",
        "loras.0.strength": 1,
        "loras.0.enabled": True,
    }


def test_empty_group_without_siblings_does_not_submit_controller():
    info = copy.deepcopy(FIXTURE["object_info"])
    schema = info["DevToolsNodeWithDynamicGroup"]
    del schema["input"]["required"]["before"]
    del schema["input"]["required"]["after"]
    workflow = copy.deepcopy(FIXTURE["cases"]["empty"]["workflow"])
    workflow["nodes"][0]["widgets_values"] = [0]
    assert workflow_to_api.convert_ui_to_api(workflow, info)["1"]["inputs"] == {}


def test_dynamic_group_resize_keeps_connected_rows_until_disconnected():
    workflow = copy.deepcopy(FIXTURE["cases"]["populated"]["workflow"])
    workflow["nodes"][0]["inputs"][-1]["link"] = 10
    original = copy.deepcopy(workflow)
    graph = Graph.from_object_info(FIXTURE["object_info"])
    with pytest.raises(ValueError, match="disconnect"):
        workflow_ops.set_widget(workflow, graph, 1, "loras", 1)
    assert workflow["nodes"] == original["nodes"]
    assert workflow["links"] == original["links"]


def test_conversion_keeps_saved_rows_above_new_schema_limit():
    info = copy.deepcopy(FIXTURE["object_info"])
    info["DevToolsNodeWithDynamicGroup"]["input"]["required"]["loras"][1]["max"] = 1
    case = FIXTURE["cases"]["populated"]
    assert workflow_to_api.convert_ui_to_api(case["workflow"], info)["1"]["inputs"] == case["expected_inputs"]


def test_row_seed_companion_stays_with_its_row_when_resized():
    info = copy.deepcopy(FIXTURE["object_info"])
    info["DevToolsNodeWithDynamicGroup"]["input"]["required"]["loras"][1]["template"] = {
        "required": {"seed": ["INT", {"default": 5, "control_after_generate": True}]}
    }
    graph = Graph.from_object_info(info)
    workflow = copy.deepcopy(FIXTURE["cases"]["empty"]["workflow"])
    grown, _ = workflow_ops.set_widget(workflow, graph, 1, "loras", 2)
    assert grown["nodes"][0]["widgets_values"] == ["first", 2, 5, "fixed", 5, "fixed", "last"]
    assert graph.widget_order_for_node("DevToolsNodeWithDynamicGroup", grown["nodes"][0]["widgets_values"]) == [
        "before",
        "loras",
        "loras.0.seed",
        "loras.0.seed.0",
        "loras.1.seed",
        "loras.1.seed.0",
        "after",
    ]
    assert workflow_to_api.convert_ui_to_api(grown, info)["1"]["inputs"] == {
        "before": "first",
        "loras.0.seed": 5,
        "loras.1.seed": 5,
        "after": "last",
    }


@pytest.mark.parametrize("shape", ["array", "object", "explicit_form", "above_max"])
def test_two_groups_with_dynamic_combo_and_seed_preserve_field_identity(shape):
    case = FIXTURE["cases"]["composite"]
    workflow = copy.deepcopy(case["workflow"])
    info = copy.deepcopy(FIXTURE["object_info"])
    node = workflow["nodes"][0]
    if shape == "object":
        node["widgets_values"] = node["widgets_values_named"].copy()
    elif shape == "explicit_form":
        node["widgets_values_form"] = {"order": case["widget_order"].copy()}
    elif shape == "above_max":
        info["CompositeDynamicGroup"]["input"]["required"]["loras"][1]["max"] = 1
    graph = Graph.from_object_info(info)

    assert workflow_to_api.convert_ui_to_api(workflow, info)["1"]["inputs"] == case["expected_inputs"]
    slots = graph.get_template_schema("example", workflow)["slots"]
    values = {slot["address"]: slot["current_value"] for slot in slots}
    assert values["1.weights.0.weight"] == 0.25
    assert values["1.mode.steps"] == 7
    assert values["1.after"] == "tail"

    edited, op = workflow_ops.set_widget(workflow, graph, 1, "weights.0.weight", 0.9)
    replayed = workflow_ops.apply_op(copy.deepcopy(case["workflow"]), op, graph)
    expected = {**case["expected_inputs"], "weights.0.weight": 0.9}
    assert workflow_to_api.convert_ui_to_api(edited, info)["1"]["inputs"] == expected
    assert workflow_to_api.convert_ui_to_api(replayed, info)["1"]["inputs"] == expected
    assert edited["nodes"][0]["widgets_values_named"] == {
        **case["workflow"]["nodes"][0]["widgets_values_named"],
        "weights.0.weight": 0.9,
    }
    if shape == "explicit_form":
        assert node["widgets_values_form"] == {"order": case["widget_order"]}

    catalog = build_catalog(graph)["types"]["CompositeDynamicGroup"]
    assert catalog["widget_order"] == ["before", "loras", "mode", "weights", "after"]
    assert catalog["dynamic_combos"]["mode"]["options"]["advanced"]["widgets"] == ["mode.steps"]
    assert catalog["dynamic_groups"]["weights"] == {
        "min": 0,
        "max": 3,
        "widgets": ["seed", "seed.0", "weight"],
        "defaults": {"seed": 5, "seed.0": "fixed", "weight": 1},
    }
    assert set(catalog["dynamic_groups"]) == {"loras", "weights"}


@pytest.mark.parametrize("operation", ["conversion", "edit", "slots"])
@pytest.mark.parametrize("mismatch", ["reordered", "unknown_key", "wrong_length", "missing_name"])
def test_dynamic_group_refuses_incompatible_explicit_form_before_changing_values(operation, mismatch):
    workflow = copy.deepcopy(FIXTURE["cases"]["populated"]["workflow"])
    node = workflow["nodes"][0]
    order = list(node["widgets_values_named"])
    node["widgets_values_form"] = {"order": order}
    if mismatch == "reordered":
        order[0], order[-1] = order[-1], order[0]
        node["widgets_values"][0], node["widgets_values"][-1] = node["widgets_values"][-1], node["widgets_values"][0]
    elif mismatch == "unknown_key":
        node["widgets_values_form"]["row_ids"] = []
    elif mismatch == "wrong_length":
        node["widgets_values"].pop()
    elif mismatch == "missing_name":
        node["widgets_values"] = node["widgets_values_named"].copy()
        del node["widgets_values"]["after"]
    original = copy.deepcopy(workflow)
    graph = Graph.from_object_info(FIXTURE["object_info"])

    error_type = workflow_to_api.WorkflowConversionError if operation == "conversion" else ValueError
    with pytest.raises(error_type, match="widgets_values_form"):
        if operation == "conversion":
            workflow_to_api.convert_ui_to_api(workflow, FIXTURE["object_info"])
        elif operation == "edit":
            workflow_ops.set_widget(workflow, graph, 1, "loras.0.strength", 0.9)
        else:
            graph.get_template_schema("example", workflow)
    assert workflow_ops.canonical(workflow) == workflow_ops.canonical(original)


@pytest.mark.parametrize("operation", ["conversion", "edit", "catalog"])
def test_dynamic_group_generated_name_collision_is_refused(operation):
    info = copy.deepcopy(FIXTURE["object_info"])
    schema = info["DevToolsNodeWithDynamicGroup"]
    schema["input"]["required"]["loras.0.strength"] = ["FLOAT", {"default": 2}]
    schema["input_order"]["required"].insert(1, "loras.0.strength")
    workflow = copy.deepcopy(FIXTURE["cases"]["populated"]["workflow"])
    workflow["nodes"][0]["widgets_values"].insert(1, 2)
    original = copy.deepcopy(workflow)
    graph = Graph.from_object_info(info)

    error_type = workflow_to_api.WorkflowConversionError if operation == "conversion" else ValueError
    with pytest.raises(error_type, match="collision"):
        if operation == "conversion":
            workflow_to_api.convert_ui_to_api(workflow, info)
        elif operation == "edit":
            workflow_ops.set_widget(workflow, graph, 1, "loras.0.strength", 0.9)
        else:
            build_catalog(graph)
    assert workflow_ops.canonical(workflow) == workflow_ops.canonical(original)


def test_dynamic_group_ambiguous_named_edit_does_not_pick_first_occurrence():
    info = copy.deepcopy(FIXTURE["object_info"])
    info["DevToolsNodeWithDynamicGroup"]["input"]["optional"] = {"after": ["STRING", {"default": "other"}]}
    graph = Graph.from_object_info(info)
    workflow = copy.deepcopy(FIXTURE["cases"]["populated"]["workflow"])
    workflow["nodes"][0]["widgets_values"].append("other")
    original = copy.deepcopy(workflow)

    with pytest.raises(ValueError, match="ambiguous"):
        workflow_ops.set_widget(workflow, graph, 1, "after", "changed")
    assert workflow_ops.canonical(workflow) == workflow_ops.canonical(original)


@pytest.mark.parametrize("changed_field", ["added", "removed"])
def test_dynamic_group_schema_width_skew_is_refused_instead_of_shifting_trailing_widget(changed_field):
    info = copy.deepcopy(FIXTURE["object_info"])
    fields = info["DevToolsNodeWithDynamicGroup"]["input"]["required"]["loras"][1]["template"]["optional"]
    if changed_field == "added":
        fields["extra"] = ["STRING", {"default": "new"}]
    else:
        del fields["enabled"]
    workflow = copy.deepcopy(FIXTURE["cases"]["populated"]["workflow"])
    graph = Graph.from_object_info(info)

    with pytest.raises(workflow_to_api.WorkflowConversionError, match="DynamicGroup"):
        workflow_to_api.convert_ui_to_api(workflow, info)
    with pytest.raises(ValueError, match="DynamicGroup"):
        workflow_ops.set_widget(workflow, graph, 1, "loras.0.strength", 0.9)


@pytest.mark.parametrize("operation", ["conversion", "edit", "catalog"])
def test_unknown_dynamic_group_schema_version_is_refused(operation):
    info = copy.deepcopy(FIXTURE["object_info"])
    info["DevToolsNodeWithDynamicGroup"]["input"]["required"]["loras"][0] = "COMFY_DYNAMICGROUP_V4"
    workflow = copy.deepcopy(FIXTURE["cases"]["populated"]["workflow"])
    graph = Graph.from_object_info(info)

    error_type = workflow_to_api.WorkflowConversionError if operation == "conversion" else ValueError
    with pytest.raises(error_type, match="unsupported DynamicGroup"):
        if operation == "conversion":
            workflow_to_api.convert_ui_to_api(workflow, info)
        elif operation == "edit":
            workflow_ops.set_widget(workflow, graph, 1, "before", "changed")
        else:
            build_catalog(graph)


def test_matching_explicit_form_remains_valid_after_standalone_resize():
    workflow = copy.deepcopy(FIXTURE["cases"]["populated"]["workflow"])
    workflow["nodes"][0]["widgets_values_form"] = {"order": list(workflow["nodes"][0]["widgets_values_named"])}
    graph = Graph.from_object_info(FIXTURE["object_info"])

    resized, _ = workflow_ops.set_widget(workflow, graph, 1, "loras", 3)

    node = resized["nodes"][0]
    assert node["widgets_values_form"]["order"] == graph.widget_order_for_node(node["type"], node["widgets_values"])
    assert workflow_to_api.convert_ui_to_api(resized, FIXTURE["object_info"])["1"]["inputs"] == {
        **FIXTURE["cases"]["populated"]["expected_inputs"],
        "loras.2.lora_name": "A.safetensors",
        "loras.2.strength": 1,
        "loras.2.enabled": True,
    }


def test_named_group_ignores_rows_beyond_saved_count():
    case = FIXTURE["cases"]["populated"]
    workflow = copy.deepcopy(case["workflow"])
    node = workflow["nodes"][0]
    node["widgets_values"] = {**node["widgets_values_named"], "loras": 1}
    expected = {key: value for key, value in case["expected_inputs"].items() if not key.startswith("loras.1.")}

    assert workflow_to_api.convert_ui_to_api(workflow, FIXTURE["object_info"])["1"]["inputs"] == expected
    graph = Graph.from_object_info(FIXTURE["object_info"])
    edited, _ = workflow_ops.set_widget(workflow, graph, 1, "loras.0.strength", 0.9)
    assert workflow_to_api.convert_ui_to_api(edited, FIXTURE["object_info"])["1"]["inputs"] == {
        **expected,
        "loras.0.strength": 0.9,
    }


def test_named_group_optional_fields_do_not_occupy_required_map_keys():
    case = FIXTURE["cases"]["populated"]
    workflow = copy.deepcopy(case["workflow"])
    node = workflow["nodes"][0]
    node["widgets_values"] = {
        key: value for key, value in node["widgets_values_named"].items() if not key.endswith(".enabled")
    }
    expected = {key: value for key, value in case["expected_inputs"].items() if not key.endswith(".enabled")}

    assert workflow_to_api.convert_ui_to_api(workflow, FIXTURE["object_info"])["1"]["inputs"] == expected
    graph = Graph.from_object_info(FIXTURE["object_info"])
    slots = graph.get_template_schema("example", workflow)["slots"]
    values = {slot["address"]: slot["current_value"] for slot in slots}
    assert values["1.loras.1.strength"] == 0.5
    assert values["1.after"] == "tail"


def test_named_group_edit_preserves_omitted_optional_and_seed_companion():
    case = FIXTURE["cases"]["composite"]
    workflow = copy.deepcopy(case["workflow"])
    node = workflow["nodes"][0]
    omitted = {"loras.1.enabled", "weights.0.seed.0"}
    node["widgets_values"] = {
        name: value for name, value in node["widgets_values_named"].items() if name not in omitted
    }
    saved_values = node["widgets_values"].copy()
    expected = {name: value for name, value in case["expected_inputs"].items() if name not in omitted}
    graph = Graph.from_object_info(FIXTURE["object_info"])

    assert workflow_to_api.convert_ui_to_api(workflow, FIXTURE["object_info"])["1"]["inputs"] == expected
    edited, _ = workflow_ops.set_widget(workflow, graph, 1, "before", "edited")

    assert edited["nodes"][0]["widgets_values"] == {**saved_values, "before": "edited"}
    assert workflow_to_api.convert_ui_to_api(edited, FIXTURE["object_info"])["1"]["inputs"] == {
        **expected,
        "before": "edited",
    }


def test_named_group_resize_preserves_existing_omissions():
    case = FIXTURE["cases"]["composite"]
    workflow = copy.deepcopy(case["workflow"])
    node = workflow["nodes"][0]
    node["widgets_values"] = {
        name: value for name, value in node["widgets_values_named"].items() if name != "loras.1.enabled"
    }
    saved_values = node["widgets_values"].copy()
    graph = Graph.from_object_info(FIXTURE["object_info"])

    grown, _ = workflow_ops.set_widget(workflow, graph, 1, "loras", 3)

    assert grown["nodes"][0]["widgets_values"] == {
        **saved_values,
        "loras": 3,
        "loras.2.lora_name": "A.safetensors",
        "loras.2.strength": 1,
        "loras.2.enabled": True,
    }


@pytest.mark.parametrize("method", ["edit", "replay"])
@pytest.mark.parametrize(
    ("widget", "value"),
    [("loras.0.lora_name", "missing.safetensors"), ("loras.0.strength", 3), ("weights.0.seed.0", "wrong")],
)
def test_group_catalog_rejection_precedes_writes(method, widget, value):
    case = FIXTURE["cases"]["composite"]
    workflow = copy.deepcopy(case["workflow"])
    original = copy.deepcopy(workflow)
    graph = Graph.from_object_info(FIXTURE["object_info"])
    with pytest.raises(ValueError):
        if method == "edit":
            workflow_ops.set_widget(workflow, graph, 1, widget, value)
        else:
            _, op = workflow_ops.set_widget(copy.deepcopy(workflow), graph, 1, "before", "edited")
            op.update(widget=widget, value=value)
            workflow_ops.apply_op(workflow, op, graph)
    assert workflow["nodes"] == original["nodes"]


def test_group_seed_companion_accepts_supported_control_mode_without_prompt_input():
    workflow = copy.deepcopy(FIXTURE["cases"]["composite"]["workflow"])
    graph = Graph.from_object_info(FIXTURE["object_info"])
    edited, _ = workflow_ops.set_widget(workflow, graph, 1, "weights.0.seed.0", "randomize")
    assert edited["nodes"][0]["widgets_values_named"]["weights.0.seed.0"] == "randomize"
    assert (
        workflow_to_api.convert_ui_to_api(edited, FIXTURE["object_info"])["1"]["inputs"]
        == FIXTURE["cases"]["composite"]["expected_inputs"]
    )


def test_saved_group_seed_companion_refuses_positional_shift():
    workflow = copy.deepcopy(FIXTURE["cases"]["composite"]["workflow"])
    case = FIXTURE["cases"]["composite"]
    workflow["nodes"][0]["widgets_values"][case["widget_order"].index("weights.0.seed.0")] = "wrong"
    with pytest.raises(workflow_to_api.WorkflowConversionError, match="companion"):
        workflow_to_api.convert_ui_to_api(workflow, FIXTURE["object_info"])


@pytest.mark.parametrize("shadow", [None, [], "invalid"])
@pytest.mark.parametrize("widget", ["before", "loras"])
def test_group_malformed_named_shadow_is_refused_before_mutation(shadow, widget):
    workflow = copy.deepcopy(FIXTURE["cases"]["populated"]["workflow"])
    workflow["nodes"][0]["widgets_values_named"] = shadow
    original = copy.deepcopy(workflow)
    graph = Graph.from_object_info(FIXTURE["object_info"])
    with pytest.raises(ValueError, match="widgets_values_named"):
        workflow_ops.set_widget(workflow, graph, 1, widget, 1 if widget == "loras" else "edited")
    assert workflow["nodes"] == original["nodes"]


def test_group_resize_with_malformed_inputs_is_refused_before_mutation():
    workflow = copy.deepcopy(FIXTURE["cases"]["populated"]["workflow"])
    workflow["nodes"][0]["inputs"] = None
    original = copy.deepcopy(workflow)
    with pytest.raises(ValueError, match="inputs"):
        workflow_ops.set_widget(workflow, Graph.from_object_info(FIXTURE["object_info"]), 1, "loras", 1)
    assert workflow["nodes"] == original["nodes"]


def test_named_group_without_controller_refuses_silent_row_loss():
    workflow = copy.deepcopy(FIXTURE["cases"]["populated"]["workflow"])
    node = workflow["nodes"][0]
    node["widgets_values"] = {name: value for name, value in node["widgets_values_named"].items() if name != "loras"}
    with pytest.raises(workflow_to_api.WorkflowConversionError, match="controller"):
        workflow_to_api.convert_ui_to_api(workflow, FIXTURE["object_info"])


@pytest.mark.parametrize("minimum", [0, 1])
def test_class_discovery_uses_default_group_rows_without_saved_workflow(minimum):
    info = copy.deepcopy(FIXTURE["object_info"])
    info["DevToolsNodeWithDynamicGroup"]["input"]["required"]["loras"][1]["min"] = minimum
    graph = Graph.from_object_info(info)
    expected = ["before", "loras"]
    if minimum:
        expected += ["loras.0.lora_name", "loras.0.strength", "loras.0.enabled"]
    expected += ["after"]
    assert graph.editable_widget_names("DevToolsNodeWithDynamicGroup") == expected
    assert graph.widget_order_for_node("DevToolsNodeWithDynamicGroup", None) == expected


def test_empty_named_group_has_no_prompt_inputs():
    info = copy.deepcopy(FIXTURE["object_info"])
    info["DevToolsNodeWithDynamicGroup"]["input"]["required"] = {
        "loras": info["DevToolsNodeWithDynamicGroup"]["input"]["required"]["loras"]
    }
    workflow = copy.deepcopy(FIXTURE["cases"]["empty"]["workflow"])
    workflow["nodes"][0]["widgets_values"] = {}

    assert workflow_to_api.convert_ui_to_api(workflow, info)["1"]["inputs"] == {}
    slots = Graph.from_object_info(info).get_template_schema("example", workflow)["slots"]
    assert {slot["address"]: slot["current_value"] for slot in slots} == {"1.loras": 0}


def test_controllerless_positional_group_is_refused():
    workflow = copy.deepcopy(FIXTURE["cases"]["empty"]["workflow"])
    workflow["nodes"][0]["widgets_values"] = []

    with pytest.raises(workflow_to_api.WorkflowConversionError, match="DynamicGroup"):
        workflow_to_api.convert_ui_to_api(workflow, FIXTURE["object_info"])


@pytest.mark.parametrize("count", [1, 1_000_000])
def test_named_group_controller_requires_represented_rows(count):
    workflow = copy.deepcopy(FIXTURE["cases"]["empty"]["workflow"])
    workflow["nodes"][0]["widgets_values"] = {
        "before": "head",
        "loras": count,
        "after": "tail",
        "unrelated_1": "extra",
        "unrelated_2": "extra",
    }

    with pytest.raises(workflow_to_api.WorkflowConversionError, match="DynamicGroup"):
        workflow_to_api.convert_ui_to_api(workflow, FIXTURE["object_info"])


@pytest.mark.parametrize("successor", ["same_row", "next_row", "trailing"])
def test_row_seed_keeps_successor_control_keyword(successor):
    info = copy.deepcopy(FIXTURE["object_info"])
    group = info["DevToolsNodeWithDynamicGroup"]["input"]["required"]["loras"][1]
    seed = ["INT", {"default": 5}]
    mode = [["fixed", "randomize"], {}]
    if successor == "same_row":
        fields = {"variation_seed": seed, "mode": mode}
        values = ["head", 1, 5, "fixed", "tail"]
        expected = {"before": "head", "loras.0.variation_seed": 5, "loras.0.mode": "fixed", "after": "tail"}
    elif successor == "next_row":
        fields = {"mode": mode, "variation_seed": seed}
        values = ["head", 2, "fixed", 5, "randomize", 6, "tail"]
        expected = {
            "before": "head",
            "loras.0.mode": "fixed",
            "loras.0.variation_seed": 5,
            "loras.1.mode": "randomize",
            "loras.1.variation_seed": 6,
            "after": "tail",
        }
    else:
        fields = {"variation_seed": seed}
        values = ["head", 1, 5, "fixed"]
        expected = {"before": "head", "loras.0.variation_seed": 5, "after": "fixed"}
    group["template"] = {"required": fields}
    workflow = copy.deepcopy(FIXTURE["cases"]["empty"]["workflow"])
    workflow["nodes"][0]["widgets_values"] = values

    assert workflow_to_api.convert_ui_to_api(workflow, info)["1"]["inputs"] == expected


@pytest.mark.parametrize("shape", ["array", "named", "explicit_form"])
def test_combo_edit_keeps_group_form_and_named_values_in_sync(shape):
    case = FIXTURE["cases"]["composite"]
    workflow = copy.deepcopy(case["workflow"])
    node = workflow["nodes"][0]
    if shape == "named":
        node["widgets_values"] = node["widgets_values_named"].copy()
    elif shape == "explicit_form":
        node["widgets_values_form"] = {"order": case["widget_order"].copy()}
    graph = Graph.from_object_info(FIXTURE["object_info"])

    edited, _ = workflow_ops.set_widget(workflow, graph, 1, "mode", "basic")

    node = edited["nodes"][0]
    expected_named = {
        key: value for key, value in case["workflow"]["nodes"][0]["widgets_values_named"].items() if key != "mode.steps"
    }
    expected_named["mode"] = "basic"
    assert node["widgets_values_named"] == expected_named
    if shape == "named":
        assert node["widgets_values"] == expected_named
    elif shape == "explicit_form":
        assert node["widgets_values_form"] == {"order": [name for name in case["widget_order"] if name != "mode.steps"]}
    expected = {key: value for key, value in case["expected_inputs"].items() if key != "mode.steps"}
    expected["mode"] = "basic"
    assert workflow_to_api.convert_ui_to_api(edited, FIXTURE["object_info"])["1"]["inputs"] == expected
    slots = graph.get_template_schema("example", edited)["slots"]
    assert {slot["address"]: slot["current_value"] for slot in slots}["1.after"] == "tail"

    edited, _ = workflow_ops.set_widget(edited, graph, 1, "weights.0.weight", 0.9, base_version=1)
    restored, _ = workflow_ops.set_widget(edited, graph, 1, "mode", "advanced", base_version=2)
    assert workflow_to_api.convert_ui_to_api(restored, FIXTURE["object_info"])["1"]["inputs"] == {
        **case["expected_inputs"],
        "mode.steps": 4,
        "weights.0.weight": 0.9,
    }
    assert restored["nodes"][0]["widgets_values_named"]["mode.steps"] == 4
    if shape == "explicit_form":
        assert restored["nodes"][0]["widgets_values_form"] == {"order": case["widget_order"]}


@pytest.mark.parametrize("linked_position", ["after", "named_after", "before", "unlinked"])
def test_group_shrink_preserves_surviving_linked_input_indices(linked_position):
    workflow = copy.deepcopy(FIXTURE["cases"]["populated"]["workflow"])
    node = workflow["nodes"][0]
    if linked_position == "named_after":
        node["widgets_values"] = node["widgets_values_named"].copy()
    extra = {
        "name": "before" if linked_position == "before" else "after",
        "type": "STRING",
        "widget": {"name": "before" if linked_position == "before" else "after"},
        "link": None if linked_position == "unlinked" else 10,
    }
    if linked_position == "before":
        node["inputs"].insert(0, extra)
    else:
        node["inputs"].append(extra)
    if extra["link"] is not None:
        workflow["links"] = [[10, 2, 0, 1, node["inputs"].index(extra), "STRING"]]
    original = copy.deepcopy(workflow)
    graph = Graph.from_object_info(FIXTURE["object_info"])

    if linked_position in ("after", "named_after"):
        with pytest.raises(ValueError, match="disconnect"):
            workflow_ops.set_widget(workflow, graph, 1, "loras", 1)
        assert workflow["nodes"] == original["nodes"]
        assert workflow["links"] == original["links"]
    else:
        shrunk, _ = workflow_ops.set_widget(workflow, graph, 1, "loras", 1)
        assert [slot["name"] for slot in shrunk["nodes"][0]["inputs"]] == [
            slot["name"] for slot in original["nodes"][0]["inputs"] if not slot["name"].startswith("loras.1.")
        ]
        assert shrunk["links"] == original["links"]
        assert shrunk["nodes"][0]["widgets_values"][-1] == "tail"


@pytest.mark.parametrize(
    "field_spec",
    [
        ["IMAGE", {}],
        ["COMBO", {"options": ["A"], "forceInput": True}],
        ["COMFY_DYNAMICCOMBO_V3", {"options": [{"key": "basic", "inputs": {"required": {}}}]}],
        ["COMFY_AUTOGROW_V3", {"template": {"input": {"required": {"image": ["IMAGE", {}]}}}}],
        ["COMFY_DYNAMICGROUP_V3", {"template": {"required": {"value": ["INT", {}]}}}],
    ],
    ids=["socket", "forced_widget", "combo", "autogrow", "group"],
)
@pytest.mark.parametrize("operation", ["catalog", "conversion"])
def test_group_rejects_unsupported_template_fields(field_spec, operation):
    info = copy.deepcopy(FIXTURE["object_info"])
    info["DevToolsNodeWithDynamicGroup"]["input"]["required"]["loras"][1]["template"] = {
        "required": {"value": field_spec}
    }
    error_type = workflow_to_api.WorkflowConversionError if operation == "conversion" else ValueError

    with pytest.raises(error_type, match="template"):
        if operation == "catalog":
            build_catalog(Graph.from_object_info(info))
        else:
            workflow_to_api.convert_ui_to_api(FIXTURE["cases"]["empty"]["workflow"], info)


@pytest.mark.parametrize("operation", ["catalog", "array_conversion", "named_conversion"])
def test_group_inside_dynamic_combo_is_explicitly_unsupported_by_cli(operation):
    info = copy.deepcopy(FIXTURE["object_info"])
    schema = info["DevToolsNodeWithDynamicGroup"]
    group = schema["input"]["required"].pop("loras")
    schema["input"]["required"]["mode"] = [
        "COMFY_DYNAMICCOMBO_V3",
        {"options": [{"key": "stack", "inputs": {"required": {"loras": group}}}]},
    ]
    schema["input_order"]["required"] = ["before", "mode", "after"]
    workflow = copy.deepcopy(FIXTURE["cases"]["populated"]["workflow"])
    node = workflow["nodes"][0]
    node["widgets_values"] = ["head", "stack", 1, "A.safetensors", 1, True, "tail"]
    if operation == "named_conversion":
        node["widgets_values"] = {
            "before": "head",
            "mode": "stack",
            "mode.loras": 1,
            "mode.loras.0.lora_name": "A.safetensors",
            "mode.loras.0.strength": 1,
            "mode.loras.0.enabled": True,
            "after": "tail",
        }
    error_type = ValueError if operation == "catalog" else workflow_to_api.WorkflowConversionError
    with pytest.raises(error_type, match="CLI.*DynamicCombo"):
        if operation == "catalog":
            build_catalog(Graph.from_object_info(info))
        else:
            workflow_to_api.convert_ui_to_api(workflow, info)


@pytest.mark.parametrize("template", ["invalid", {"required": ["invalid"]}])
def test_malformed_group_template_returns_validation_error(template):
    info = copy.deepcopy(FIXTURE["object_info"])
    info["DevToolsNodeWithDynamicGroup"]["output_node"] = True
    info["DevToolsNodeWithDynamicGroup"]["input"]["required"]["loras"][1]["template"] = template
    prompt = {"1": {"class_type": "DevToolsNodeWithDynamicGroup", "inputs": {"before": "head", "after": "tail"}}}
    result = Graph.from_object_info(info).validate_workflow(prompt)
    assert not result["valid"]
    assert any(error["code"] == "invalid_dynamic_group_schema" for error in result["errors"])


@pytest.mark.parametrize(
    ("minimum", "maximum"),
    [(21, 21), (True, 3), (0.5, 3), (-1, 3), (2, 1), (0, 21), (0, 0)],
)
def test_group_metadata_bounds_are_refused_before_default_expansion(minimum, maximum):
    info = copy.deepcopy(FIXTURE["object_info"])
    options = info["DevToolsNodeWithDynamicGroup"]["input"]["required"]["loras"][1]
    options.update(min=minimum, max=maximum)
    with pytest.raises(ValueError, match="bounds"):
        Graph.from_object_info(info).widget_defaults("DevToolsNodeWithDynamicGroup")


@pytest.mark.parametrize("malformation", ["force_input", "dotted_field"])
def test_group_refuses_unsupported_controller_and_generated_companion_collision(malformation):
    info = copy.deepcopy(FIXTURE["object_info"])
    options = info["DevToolsNodeWithDynamicGroup"]["input"]["required"]["loras"][1]
    if malformation == "force_input":
        options["forceInput"] = True
    else:
        options["template"] = {
            "required": {"seed": ["INT", {"control_after_generate": True}], "seed.0": ["STRING", {}]}
        }
    with pytest.raises(ValueError, match="DynamicGroup"):
        build_catalog(Graph.from_object_info(info))


@pytest.mark.parametrize(
    ("inputs", "minimum", "error_field"),
    [
        ({}, 0, None),
        ({}, 1, "loras"),
        ({"loras.2.lora_name": "C.safetensors", "loras.2.strength": 0.5}, 1, None),
        ({"loras.2.strength": 0.5}, 0, "loras.2.lora_name"),
        ({"loras.3.lora_name": "A.safetensors", "loras.3.strength": 1}, 0, "loras.3.lora_name"),
        ({"loras.1000000.strength": 1}, 0, "loras.1000000.strength"),
        ({"loras.01.strength": 1}, 0, "loras.01.strength"),
        ({"loras.0.unknown": 1}, 0, "loras.0.unknown"),
        ({"loras.0.lora_name": "A.safetensors", "loras.0.strength": 3}, 0, "loras.0.strength"),
    ],
)
def test_dynamic_group_prompt_validation_checks_only_submitted_rows(inputs, minimum, error_field):
    info = copy.deepcopy(FIXTURE["object_info"])
    schema = info["DevToolsNodeWithDynamicGroup"]
    schema["output_node"] = True
    schema["input"]["required"]["loras"][1]["min"] = minimum
    graph = Graph.from_object_info(info)
    prompt = {
        "1": {"class_type": "DevToolsNodeWithDynamicGroup", "inputs": {"before": "head", "after": "tail", **inputs}}
    }

    result = graph.validate_workflow(prompt)

    if error_field is None:
        assert result["valid"], result
    else:
        assert not result["valid"]
        assert error_field in {error["field"] for error in result["errors"]}
