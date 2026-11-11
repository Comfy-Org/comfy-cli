"""Producer admission only: destructive rebuilds must not erase batch writes."""

import copy

import pytest

from comfy_cli import workflow_ops as ops
from comfy_cli.cql.engine import Graph


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
