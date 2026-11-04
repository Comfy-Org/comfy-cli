"""``workflow print --format json``: the structured ``nodes`` records.

The records are taken while each line is rendered, so the main guard here is
parity: for every fixture workflow, each record says exactly what its printed
line says — the same argument names, the same literal values, and a link (or
an ``IN`` reference) wherever the line has a reference.
"""

from __future__ import annotations

import ast
import copy
import json
import re
from pathlib import Path

import pytest

from comfy_cli.cql.engine import Graph
from comfy_cli.workflow_print import PrintUnsupported, py_literal, render_py

FIXTURES = Path(__file__).parent / "fixtures"
_LINE_ID = re.compile(r"^\s*(\S+)")


@pytest.fixture(scope="module")
def sd15_graph() -> Graph:
    return Graph.from_object_info(json.loads((FIXTURES / "sd15_object_info.json").read_text()))


@pytest.fixture(scope="module")
def subgraph_graph() -> Graph:
    return Graph.from_object_info(json.loads((FIXTURES / "subgraph_object_info.json").read_text()))


def _by_id(res) -> dict[str, dict]:
    return {r["id"]: r for r in res.nodes}


def test_sd15_records(sd15_graph):
    wf = json.loads((FIXTURES / "sd15_ui_workflow.json").read_text())
    nodes = _by_id(render_py(wf, sd15_graph))
    k = nodes["3"]
    assert (k["name"], k["type"], k["mode"], k["title"]) == ("ksampler", "KSampler", "normal", "")
    assert k["links"]["model"] == {"node": "4", "output": "MODEL", "slot": 0}
    # A single-output source is still named, so a caller never guesses it.
    assert k["links"]["positive"] == {"node": "6", "output": "CONDITIONING", "slot": 0}
    assert k["widgets"]["seed"] == 685468484323813
    assert k["widgets"]["steps"] == 20
    assert k["widgets"]["sampler_name"] == "euler"
    assert set(nodes) == {"3", "4", "5", "6", "7", "8", "9"}
    assert nodes["4"]["outputs"] == ["MODEL", "CLIP", "VAE"]


def test_flags_mode_title_and_unlinked_socket(sd15_graph):
    wf = json.loads((FIXTURES / "sd15_ui_workflow.json").read_text())
    for n in wf["nodes"]:
        if n["id"] == 8:
            n["mode"] = 4
            n["flags"] = {"collapsed": True, "pinned": True}
            n["title"] = "Decode"
    wf["links"] = [ln for ln in wf["links"] if not (ln[3] == 8 and ln[4] == 1)]
    for n in wf["nodes"]:
        if n["id"] == 8:
            for inp in n["inputs"]:
                if inp["name"] == "vae":
                    inp["link"] = None
    rec = _by_id(render_py(wf, sd15_graph))["8"]
    assert (rec["mode"], rec["collapsed"], rec["pinned"], rec["title"]) == ("bypassed", True, True, "Decode")
    assert rec["links"]["vae"] is None


def test_multi_output_index_ref_is_a_link(sd15_graph):
    """``ckpt.out[1]`` (an output whose name is not an identifier) is a link,
    not widget text."""
    wf = json.loads((FIXTURES / "sd15_ui_workflow.json").read_text())
    for n in wf["nodes"]:
        if n["id"] == 4:
            n["outputs"][1]["name"] = "my clip"
    res = render_py(wf, sd15_graph)
    assert "checkpoint_loader_simple.out[1]" in res.source
    assert _by_id(res)["6"]["links"]["clip"] == {"node": "4", "output": "my clip", "slot": 1}


def test_subgraph_interior_once_per_instance(subgraph_graph):
    wf = json.loads((FIXTURES / "subgraph_template_ui.json").read_text())
    inst = next(n for n in wf["nodes"] if n["id"] == 10)
    twin = copy.deepcopy(inst)
    twin["id"] = 900
    for inp in twin.get("inputs") or []:
        inp["link"] = None
    for out in twin.get("outputs") or []:
        out["links"] = []
    wf["nodes"].append(twin)
    res = render_py(wf, subgraph_graph)
    nodes = _by_id(res)
    assert nodes["10"]["subgraph"] == inst["type"]
    interior_10 = {i[3:] for i in nodes if i.startswith("10/") and i.count("/") == 1}
    interior_900 = {i[4:] for i in nodes if i.startswith("900/") and i.count("/") == 1}
    assert interior_10 and interior_10 == interior_900
    for i in nodes:
        if i.startswith("900/"):
            rec = nodes[i]
            for link in rec["links"].values():
                assert link is None or link["node"].startswith("900/")
            for p in rec["promoted"].values():
                assert p["instance"] == rec["id"].rsplit("/", 1)[0]
    assert any(r["promoted"] for r in res.nodes if r["id"].startswith("10/"))
    # Every link points at a node the records hold.
    for r in res.nodes:
        for link in r["links"].values():
            assert link is None or link["node"] in nodes, (r["id"], link)


def _fixture_cases():
    cases = []
    for wf_path in sorted(FIXTURES.glob("*.json")):
        if "object_info" in wf_path.name:
            continue
        try:
            wf = json.loads(wf_path.read_text())
        except ValueError:
            continue
        if isinstance(wf, dict) and isinstance(wf.get("nodes"), list) and wf.get("links") is not None:
            cases.append(wf_path)
    return cases


@pytest.mark.parametrize("wf_path", _fixture_cases(), ids=lambda p: p.name)
def test_records_match_printed_lines(wf_path, sd15_graph, subgraph_graph):
    wf = json.loads(wf_path.read_text())
    graph = subgraph_graph if "subgraph" in wf_path.name else sd15_graph
    try:
        res = render_py(wf, graph)
    except PrintUnsupported:
        pytest.skip("fixture is refused by print")
    assert_records_match_source(res)


def _code_part(line: str) -> str:
    """The statement before the line's ``  # <id>`` comment; a prompt may
    itself contain ``  #``, so take the first split that parses."""
    parts = line.split("  #")
    for i in range(1, len(parts)):
        code = "  #".join(parts[:i])
        try:
            ast.parse(code)
        except SyntaxError:
            continue
        return code
    raise AssertionError(f"unparseable line: {line}")


def assert_records_match_source(res) -> None:
    """Each printed node line and its record carry the same arguments: names,
    literal text, and a link (to the node its reference names) or an ``IN``
    reference wherever the line has a reference."""
    # Lines are matched to records by binding name: top-level names are
    # unique, and an interior line is compared with its first instance copy.
    first = {}
    for r in res.nodes:
        key = r["name"] if "definition" not in r else (r["definition"], r["name"])
        first.setdefault(key, r)
    current_def = None
    checked = 0
    for raw in res.source.split("\n"):
        line = raw.strip()
        if raw.startswith("# subgraph "):
            current_def = raw.split()[2]
            continue
        if not line or line.startswith("#") or line.startswith("OUT"):
            continue
        code = _code_part(line)
        tree = ast.parse(code)
        stmt = tree.body[0]
        name = stmt.targets[0].id
        rec = first[(current_def, name) if raw.startswith("    ") else name]
        assert _LINE_ID.search(line[len(code) + 3 :]).group(1) == rec["id"].rsplit("/", 1)[-1]
        call = stmt.value
        kws = {}
        for kw in call.keywords:
            if kw.arg is None:
                for k, v in zip(kw.value.keys, kw.value.values):
                    kws[k.value] = v
            else:
                kws[kw.arg] = kw.value
        if "widgets_positional" in rec or rec.get("class_unknown"):
            kws.pop("widgets", None)
        assert set(kws) == set(rec["widgets"]) | set(rec["links"]) | set(rec["promoted"]), line
        for k, v in kws.items():
            if isinstance(v, (ast.Name, ast.Attribute, ast.Subscript)) and not (
                isinstance(v, ast.Name) and v.id in ("True", "False", "None")
            ):
                base = v
                while not isinstance(base, ast.Name):
                    base = base.value
                if base.id == "IN":
                    assert k in rec["promoted"], (line, k)
                    continue
                link = rec["links"].get(k)
                assert link is not None, (line, k)
                if not raw.startswith("    "):
                    assert link["node"] == res.bindings[base.id], (line, k)
                continue
            text = ast.get_source_segment(code, v)
            if text == "None" and k in rec["links"]:
                assert rec["links"][k] is None
                continue
            assert k in rec["widgets"], (line, k)
            assert py_literal(rec["widgets"][k]) == text, (line, k)
        checked += 1
    assert checked > 0


def test_open_autogrow_slot_is_an_empty_link():
    """The printer adds one open slot after the wired ones; the records hold
    it as an empty link, so a caller can wire it by that name."""
    from comfy_cli import workflow_ops

    graph = Graph.from_object_info(json.loads((FIXTURES / "object_info_nested_autogrow.json").read_text()))
    wf = {"last_node_id": 0, "last_link_id": 0, "nodes": [], "links": [], "groups": [], "version": 0.4}
    wf, op = workflow_ops.add_node(wf, graph, "LoadImage")
    a = op["node_id"]
    wf, op = workflow_ops.add_node(wf, graph, "GeminiNanoBanana2V2")
    b = op["node_id"]
    wf, _ = workflow_ops.connect(wf, graph, a, "IMAGE", b, "model.images.image_1")
    res = render_py(wf, graph)
    links = _by_id(res)[str(b)]["links"]
    assert links["model.images.image_1"] == {"node": str(a), "output": "IMAGE", "slot": 0}
    assert links["model.images.image_2"] is None
    assert_records_match_source(res)


@pytest.mark.parametrize(
    "wf_path",
    sorted(p for p in (FIXTURES / "gallery").glob("*.json") if "object_info" not in p.name),
    ids=lambda p: p.name,
)
def test_gallery_records_match_printed_lines(wf_path):
    graph = Graph.from_object_info(json.loads((FIXTURES / "object_info_subgraph_promoted.json").read_text()))
    try:
        res = render_py(json.loads(wf_path.read_text()), graph)
    except PrintUnsupported:
        pytest.skip("fixture is refused by print")
    assert_records_match_source(res)
