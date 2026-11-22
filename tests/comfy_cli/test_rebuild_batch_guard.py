"""Producer admission only: destructive rebuilds must not erase batch writes."""

import copy

import pytest

from comfy_cli import workflow_ops as ops
from comfy_cli.cql.engine import DynamicGroupBindingError, Graph


def graph():
    return Graph.from_object_info(
        {
            "Gen": {
                "input": {
                    "required": {
                        "model": [
                            "COMFY_DYNAMICCOMBO_V3",
                            {
                                "options": [
                                    {"key": "base", "inputs": {"required": {}}},
                                    {
                                        "key": "max",
                                        "inputs": {
                                            "required": {
                                                "mode": [["balanced", "quality"], {}],
                                                "quality": ["STRING", {"default": "low"}],
                                            }
                                        },
                                    },
                                ]
                            },
                        ],
                        "axis": ["STRING", {"default": "x"}],
                        "axis.mode": ["STRING", {"default": "y"}],
                        "rows": [
                            "COMFY_DYNAMICGROUP_V3",
                            {"min": 0, "max": 4, "template": {"required": {"weight": ["FLOAT", {"default": 1.0}]}}},
                        ],
                    }
                },
                "output": [],
            },
        }
    )


def write(node, widget, value):
    return {"op": "set_widget", "node": node, "widget": widget, "value": value}


def fresh():
    g = graph()
    wf, op = ops.add_node({"nodes": [], "links": []}, g, "Gen")
    wf, _ = ops.set_widget(wf, g, op["node_id"], "model", "max")
    return g, wf, op["node_id"]


def test_successful_batch_preserves_in_place_workflow_and_node():
    g, wf, nid = fresh()
    node = wf["nodes"][0]
    result, stream, _ = ops.apply_specs(wf, g, [write(nid, "axis", "changed")])
    assert result is wf
    assert wf["nodes"][0] is node
    assert stream[0]["value"] == "changed"
    assert node["widgets_values"] != fresh()[1]["nodes"][0]["widgets_values"]


def test_batch_allocates_once_so_execution_cannot_remint_a_colliding_id(monkeypatch):
    g, wf, nid = fresh()
    allocations = []

    def allocate():
        value = nid + 1 if not allocations else nid
        allocations.append(value)
        return value

    monkeypatch.setattr(ops, "mint_id", allocate)
    _, stream, aliases = ops.apply_specs(
        wf,
        g,
        [
            {"op": "add_node", "class_type": "Gen", "as": "new"},
            write(nid, "model.mode", "quality"),
            write("$new", "model", "base"),
        ],
    )
    assert allocations == [nid + 1]
    assert aliases["new"] == nid + 1
    assert len(wf["nodes"]) == 2
    assert len(stream) == 3
    assert "quality" in wf["nodes"][0]["widgets_values"]


def test_refusal_restores_held_container_references():
    g, wf, nid = fresh()
    before = copy.deepcopy(wf)
    nodes, node = wf["nodes"], wf["nodes"][0]
    widgets, stamps = node["widgets_values"], wf["_widget_stamps"]
    with pytest.raises(ValueError, match="destructive.*rebuild"):
        ops.apply_specs(
            wf,
            g,
            [
                {"op": "add_node", "class_type": "Gen"},
                write(nid, "model.mode", "quality"),
                write(nid, "model", "base"),
            ],
        )
    assert wf == before
    assert wf["nodes"] is nodes
    assert nodes[0] is node
    assert node["widgets_values"] is widgets
    assert wf["_widget_stamps"] is stamps
    assert node == before["nodes"][0]


def test_collision_refusal_restores_the_actual_execution(monkeypatch):
    g, wf, nid = fresh()
    before = copy.deepcopy(wf)
    node = wf["nodes"][0]
    monkeypatch.setattr(ops, "mint_id", lambda: nid)
    with pytest.raises(ValueError, match="destructive.*rebuild"):
        ops.apply_specs(
            wf,
            g,
            [
                {"op": "add_node", "class_type": "Gen", "as": "new"},
                write(nid, "model.mode", "quality"),
                write("$new", "model", "base"),
            ],
        )
    assert wf == before
    assert wf["nodes"][0] is node


def test_success_mints_only_the_returned_op_ids(monkeypatch):
    g, wf, nid = fresh()
    mint = ops._new_op_id
    issued = []

    def issue():
        value = mint()
        issued.append(value)
        return value

    monkeypatch.setattr(ops, "_new_op_id", issue)
    _, stream, _ = ops.apply_specs(wf, g, [write(nid, "axis", "one"), write(nid, "axis", "two")])
    assert issued == [op["op_id"] for op in stream]


@pytest.mark.parametrize("unrelated_first", [True, False])
def test_duplicate_control_marker_uses_the_first_slots_actual_owner(unrelated_first):
    marker = ["INT", {"default": 1, "control_after_generate": True}]
    model = [
        "COMFY_DYNAMICCOMBO_V3",
        {
            "options": [
                {"key": "base", "inputs": {"required": {}}},
                {"key": "max", "inputs": {"required": {"child": marker}}},
            ]
        },
    ]
    order = ["aseed", "model"] if unrelated_first else ["model", "aseed"]
    g = Graph.from_object_info(
        {"Gen": {"input": {"required": {"aseed": marker, "model": model}}, "input_order": {"required": order}}}
    )
    wf, added = ops.add_node({"nodes": [], "links": []}, g, "Gen")
    nid = added["node_id"]
    ops.set_widget(wf, g, nid, "model", "max")
    before = copy.deepcopy(wf)
    specs = [write(nid, "control_after_generate", "randomize"), write(nid, "model", "base")]
    if unrelated_first:
        ops.apply_specs(wf, g, specs)
        assert wf["nodes"][0]["widgets_values"] == [1, "randomize", "base"]
    else:
        with pytest.raises(ValueError, match="destructive.*rebuild"):
            ops.apply_specs(wf, g, specs)
        assert wf == before


def test_failure_restores_modified_then_detached_node():
    g, wf, nid = fresh()
    before = copy.deepcopy(wf)
    node, widgets = wf["nodes"][0], wf["nodes"][0]["widgets_values"]
    with pytest.raises(ValueError, match="unknown op") as caught:
        ops.apply_specs(wf, g, [write(nid, "axis", "changed"), {"op": "delete_node", "node": nid}, {"op": "bad"}])
    assert caught.value.applied_count == 0
    assert wf == before
    assert wf["nodes"][0] is node
    assert node["widgets_values"] is widgets
    assert node == before["nodes"][0]


def test_interruption_rolls_back_without_replacing_the_exception(monkeypatch):
    g, wf, nid = fresh()
    before = copy.deepcopy(wf)
    node, widgets = wf["nodes"][0], wf["nodes"][0]["widgets_values"]
    interrupted = KeyboardInterrupt("interrupted after mutation")

    def interrupt(*args):
        raise interrupted

    monkeypatch.setattr(ops, "_admit_rebuild_write", interrupt)
    with pytest.raises(KeyboardInterrupt) as caught:
        ops.apply_specs(wf, g, [write(nid, "axis", "changed")])
    assert caught.value is interrupted
    assert wf == before
    assert wf["nodes"][0] is node
    assert node["widgets_values"] is widgets


@pytest.mark.parametrize("alias", [False, True])
def test_child_then_selector_refuses_atomically(alias):
    g, wf, nid = fresh()
    before = copy.deepcopy(wf)
    prefix = [{"op": "add_node", "class_type": "Gen", "as": "new"}] if alias else []
    node = "$new" if alias else str(nid)
    prefix += [write(node, "model", "max")] if alias else []
    specs = prefix + [
        write(node, "model.mode", "quality"),
        write("new" if alias else nid, "model", "base"),
        write(node, "model", "max"),
    ]
    with pytest.raises(ValueError, match="destructive.*rebuild") as caught:
        ops.apply_specs(wf, g, specs)
    assert "separate batches" not in str(caught.value)
    assert caught.value.spec_index == len(prefix) + 1
    assert caught.value.spec_op == "set_widget"
    assert caught.value.applied_count == 0
    assert wf == before


@pytest.mark.parametrize("count,refuse", [(0, True), (1, False), (2, False)])
def test_group_only_refuses_removed_written_rows(count, refuse):
    g, wf, nid = fresh()
    wf, _ = ops.set_widget(wf, g, nid, "rows", 1)
    before = copy.deepcopy(wf)
    specs = [write(nid, "rows.0.weight", 0.5), write(str(nid), "rows", count), write(nid, "rows", 2)]
    if refuse:
        with pytest.raises(ValueError, match="destructive.*rebuild"):
            ops.apply_specs(wf, g, specs)
        assert wf == before
    else:
        ops.apply_specs(wf, g, specs)


def test_group_marker_uses_its_saved_row_field_name_for_shrink_admission():
    marker = ["INT", {"default": 1, "control_after_generate": True}]
    g = Graph.from_object_info(
        {
            "Gen": {
                "input": {
                    "required": {
                        "rows": [
                            "COMFY_DYNAMICGROUP_V3",
                            {"min": 0, "max": 2, "template": {"required": {"seed": marker}}},
                        ]
                    }
                }
            }
        }
    )
    wf, added = ops.add_node({"nodes": [], "links": []}, g, "Gen")
    nid = added["node_id"]
    wf, _ = ops.set_widget(wf, g, nid, "rows", 1)
    before = copy.deepcopy(wf)

    with pytest.raises(ValueError, match=r"erase earlier batch write to 'rows\.0\.seed\.0'"):
        ops.apply_specs(
            wf,
            g,
            [write(nid, "rows.0.seed.0", "randomize"), write(nid, "rows", 0)],
        )

    assert wf == before


def test_count_then_group_child_passes():
    g, wf, nid = fresh()

    ops.apply_specs(wf, g, [write(nid, "rows", 1), write(nid, "rows.0.weight", 0.5)])

    assert 0.5 in wf["nodes"][0]["widgets_values"]


def test_noop_selector_scalar_prefix_and_selector_then_child_pass():
    g, wf, nid = fresh()
    ops.apply_specs(
        wf,
        g,
        [
            write(nid, "model.mode", "quality"),
            write(nid, "model", "max"),
            write(nid, "axis.mode", "z"),
            write(nid, "axis", "q"),
        ],
    )
    ops.apply_specs(
        wf, g, [write(nid, "model", "base"), write(nid, "model", "max"), write(nid, "model.mode", "quality")]
    )


@pytest.mark.parametrize("rebuild", ["model", "model.nested"])
def test_nested_schema_owners(rebuild):
    info = {
        "Gen": {
            "input": {
                "required": {
                    "model": [
                        "COMFY_DYNAMICCOMBO_V3",
                        {
                            "options": [
                                {"key": "base", "inputs": {"required": {}}},
                                {
                                    "key": "max",
                                    "inputs": {
                                        "required": {
                                            "nested": [
                                                "COMFY_DYNAMICCOMBO_V3",
                                                {
                                                    "options": [
                                                        {
                                                            "key": "a",
                                                            "inputs": {
                                                                "required": {"text": ["STRING", {"default": ""}]}
                                                            },
                                                        },
                                                        {"key": "b", "inputs": {"required": {}}},
                                                    ]
                                                },
                                            ],
                                        }
                                    },
                                },
                            ]
                        },
                    ],
                }
            },
            "output": [],
        },
    }
    g = Graph.from_object_info(info)
    wf, op = ops.add_node({"nodes": [], "links": []}, g, "Gen")
    nid = op["node_id"]
    wf, _ = ops.set_widget(wf, g, nid, "model", "max")
    before = copy.deepcopy(wf)
    with pytest.raises(ValueError, match="destructive.*rebuild"):
        ops.apply_specs(
            wf,
            g,
            [write(nid, "model.nested.text", "saved"), write(nid, rebuild, "base" if rebuild == "model" else "b")],
        )
    assert wf == before


@pytest.mark.parametrize("rebuild", ["model", "model.choice"])
def test_same_named_nested_selectors_keep_each_variants_ownership(rebuild):
    def nested(child):
        return [
            "COMFY_DYNAMICCOMBO_V3",
            {
                "options": [
                    {"key": "shown", "inputs": {"required": {child: ["STRING", {"default": ""}]}}},
                    {"key": "hidden", "inputs": {"required": {}}},
                ]
            },
        ]

    g = Graph.from_object_info(
        {
            "Gen": {
                "input": {
                    "required": {
                        "model": [
                            "COMFY_DYNAMICCOMBO_V3",
                            {
                                "options": [
                                    {"key": "first", "inputs": {"required": {"choice": nested("first_child")}}},
                                    {"key": "second", "inputs": {"required": {"choice": nested("second_child")}}},
                                ]
                            },
                        ]
                    }
                }
            }
        }
    )
    wf, added = ops.add_node({"nodes": [], "links": []}, g, "Gen")
    nid = added["node_id"]
    wf, _ = ops.set_widget(wf, g, nid, "model", "second")
    before = copy.deepcopy(wf)
    rebuild_value = "first" if rebuild == "model" else "hidden"

    with pytest.raises(ValueError, match=r"erase earlier batch write to 'model\.choice\.second_child'"):
        ops.apply_specs(
            wf,
            g,
            [write(nid, "model.choice.second_child", "saved"), write(nid, rebuild, rebuild_value)],
        )

    assert wf == before


def test_dynamic_group_inside_combo_remains_an_explicitly_unsupported_schema():
    g = Graph.from_object_info(
        {
            "Gen": {
                "input": {
                    "required": {
                        "model": [
                            "COMFY_DYNAMICCOMBO_V3",
                            {
                                "options": [
                                    {"key": "base", "inputs": {"required": {}}},
                                    {
                                        "key": "rows",
                                        "inputs": {
                                            "required": {
                                                "items": [
                                                    "COMFY_DYNAMICGROUP_V3",
                                                    {
                                                        "min": 0,
                                                        "max": 2,
                                                        "template": {
                                                            "required": {"weight": ["FLOAT", {"default": 1.0}]}
                                                        },
                                                    },
                                                ]
                                            }
                                        },
                                    },
                                ]
                            },
                        ]
                    }
                }
            }
        }
    )

    with pytest.raises(DynamicGroupBindingError, match="does not support DynamicGroup inside DynamicCombo"):
        ops.add_node({"nodes": [], "links": []}, g, "Gen")


def promoted_fresh():
    g, interior, nid = fresh()
    inner = interior["nodes"][0]
    inner["inputs"] = [
        {"name": "model", "type": "COMFY_DYNAMICCOMBO_V3", "widget": {"name": "model"}, "link": 1},
        {"name": "model.mode", "type": "COMBO", "widget": {"name": "model.mode"}, "link": 2},
    ]
    definition = {
        "id": "definition",
        "name": "Group",
        "inputs": [
            {"id": "selector", "name": "model", "type": "COMFY_DYNAMICCOMBO_V3", "linkIds": [1]},
            {"id": "child", "name": "model.mode", "type": "COMBO", "linkIds": [2]},
        ],
        "outputs": [],
        "nodes": [inner],
        "links": [
            {
                "id": 1,
                "origin_id": -10,
                "origin_slot": 0,
                "target_id": nid,
                "target_slot": 0,
                "type": "COMFY_DYNAMICCOMBO_V3",
            },
            {"id": 2, "origin_id": -10, "origin_slot": 1, "target_id": nid, "target_slot": 1, "type": "COMBO"},
        ],
    }
    host = {
        "id": 9,
        "type": "definition",
        "inputs": [
            {"name": "model", "type": "COMFY_DYNAMICCOMBO_V3", "widget": {"name": "model"}, "link": None},
            {"name": "model.mode", "type": "COMBO", "widget": {"name": "model.mode"}, "link": None},
        ],
        "outputs": [],
        "properties": {},
        "widgets_values": ["max", "balanced"],
    }
    return g, {"nodes": [host], "links": [], "definitions": {"subgraphs": [definition]}}, host["id"]


def test_promoted_child_then_promoted_selector_refuses_atomically():
    g, wf, nid = promoted_fresh()
    before = copy.deepcopy(wf)

    with pytest.raises(ValueError, match=r"erase earlier batch write to 'model\.mode'"):
        ops.apply_specs(wf, g, [write(nid, "model.mode", "quality"), write(nid, "model", "base")])

    assert wf == before


def test_graphless_legacy_primitive_batch_still_applies():
    wf = {
        "nodes": [
            {
                "id": 1,
                "type": "PrimitiveNode",
                "inputs": [],
                "outputs": [{"name": "STRING", "type": "STRING", "links": [], "widget": {"name": "value"}}],
                "widgets_values": ["old"],
            }
        ],
        "links": [],
    }

    result, stream, _ = ops.apply_specs(wf, None, [write(1, "value", "new")])

    assert result["nodes"][0]["widgets_values"] == ["new"]
    assert stream[0]["value"] == "new"


@pytest.mark.parametrize("stale", ["child", "selector"])
def test_stale_lww_writes_do_not_create_false_structural_conflicts(stale):
    g, wf, nid = fresh()
    stale_widget = "model.mode" if stale == "child" else "model"
    stale_value = "balanced" if stale == "child" else "max"
    wf, _ = ops.set_widget(wf, g, nid, stale_widget, stale_value, actor="peer", base_version=10)

    result, stream, _ = ops.apply_specs(
        wf,
        g,
        [write(nid, "model.mode", "quality"), write(nid, "model", "base")],
        base_version=5,
    )

    assert len(stream) == 2
    node = result["nodes"][0]
    selector = ops._widget_index(g, "Gen", "model", node["widgets_values"], node_id=nid, node=node)
    assert node["widgets_values"][selector] == ("base" if stale == "child" else "max")


def test_diagnostic_names_the_earliest_conflicting_write(monkeypatch):
    monkeypatch.setattr(ops, "mint_id", lambda: 1)
    g, wf, nid = fresh()

    with pytest.raises(ValueError, match=r"erase earlier batch write to 'model\.mode'"):
        ops.apply_specs(
            wf,
            g,
            [
                write(nid, "model.mode", "quality"),
                write(nid, "model.quality", "high"),
                write(nid, "model", "base"),
            ],
        )


def test_interior_identity_and_unrelated_instance():
    g, inner, nid = fresh()
    wf = {
        "nodes": [
            {"id": 9, "type": "def", "inputs": [], "outputs": [], "properties": {}},
            {"id": 10, "type": "def", "inputs": [], "outputs": [], "properties": {}},
        ],
        "links": [],
        "definitions": {
            "subgraphs": [
                {"id": "def", "name": "Group", "inputs": [], "outputs": [], "nodes": inner["nodes"], "links": []}
            ]
        },
    }
    before = copy.deepcopy(wf)
    with pytest.raises(ValueError, match="destructive.*rebuild"):
        ops.apply_specs(wf, g, [write(f"9/{nid}", "model.mode", "quality"), write(f"9:{nid}", "model", "base")])
    assert wf == before
    ops.apply_specs(wf, g, [write(f"9/{nid}", "model.mode", "quality"), write(f"10/{nid}", "model", "base")])


def test_cli_refusal_preserves_file_bytes(tmp_path, monkeypatch):
    import inspect
    import json

    import typer

    from comfy_cli.command import workflow

    workflow_edit = workflow._wedit

    g, wf, nid = fresh()
    path = tmp_path / "workflow.json"
    original = json.dumps(wf, indent=3).encode() + b"\n"
    path.write_bytes(original)
    recipe = tmp_path / "ops.json"
    recipe.write_text(
        json.dumps(
            [
                {"op": "add_node", "class_type": "Gen", "as": "new"},
                write(nid, "model.mode", "quality"),
                write(nid, "model", "base"),
                write(nid, "model", "max"),
            ]
        )
    )
    monkeypatch.setattr(workflow_edit, "_graph_or_exit", lambda *args: g)
    with pytest.raises(typer.Exit) as caught:
        inspect.unwrap(workflow_edit.apply_cmd)(str(path), str(recipe))
    assert caught.value.exit_code == 1
    assert path.read_bytes() == original
