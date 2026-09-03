"""`workflow ls-nodes` must report a node's BYPASS/MUTE state.

ComfyUI lets a user disable a node without deleting it: mode 4 = bypass (passes
input through), mode 2 = mute/never (removed from execution). workflow_to_api
already understands both — it strips them at API conversion
(workflow_to_api.py:149) and has a bypassed-id helper (:550).

But ls-nodes emitted only id/type/title, so a caller inspecting the graph could
not tell a disabled node from a live one. The consequence is worse than a
cosmetic gap: the agent "repairs" a graph whose node is merely bypassed, or
reports a workflow ready to run when a required node is muted.
"""

from __future__ import annotations

import pytest
from test_workflow_edit import (  # type: ignore[import-not-found]
    _base_workflow,
    _graph,
    _run,
    _write,
    reset_singleton,  # noqa: F401  (autouse fixture)
)

from comfy_cli.command import workflow_edit

MODE_MUTED, MODE_BYPASS = 2, 4


@pytest.fixture
def patched_graph(monkeypatch):
    monkeypatch.setattr(workflow_edit, "_get_graph", lambda *a, **kw: _graph())


def _wf_with_modes() -> dict:
    wf = _base_workflow()
    wf["nodes"][0]["mode"] = MODE_BYPASS  # KSampler, id 3
    wf["nodes"][1]["mode"] = MODE_MUTED  # EmptyLatentImage, id 7
    wf["nodes"].append(
        {
            "id": 9,
            "type": "VAEDecode",
            "pos": [0, 0],
            "mode": 0,
            "inputs": [],
            "outputs": [],
            "widgets_values": [],
        }
    )
    return wf


def _rows(tmp_path, capsys) -> dict:
    path = _write(tmp_path, _wf_with_modes())
    env = _run(["ls-nodes", str(path)], capsys)
    assert env["ok"] is True, env
    return {r["id"]: r for r in env["data"]["nodes"]}


def test_ls_nodes_reports_bypassed_and_muted(patched_graph, tmp_path, capsys):
    rows = _rows(tmp_path, capsys)
    assert rows[3].get("mode") == "bypass", rows[3]
    assert rows[7].get("mode") == "mute", rows[7]


def test_ls_nodes_omits_mode_for_normal_nodes(patched_graph, tmp_path, capsys):
    """A live node must stay a single clean row — no mode noise on the 99% case."""
    rows = _rows(tmp_path, capsys)
    assert "mode" not in rows[9], rows[9]


def test_ls_nodes_unchanged_when_no_modes_set(patched_graph, tmp_path, capsys):
    path = _write(tmp_path, _base_workflow())
    env = _run(["ls-nodes", str(path)], capsys)
    assert all("mode" not in r for r in env["data"]["nodes"]), env["data"]["nodes"]


# ---------------------------------------------------------------------------
# subgraph interiors — `data.subgraph_nodes[]`
#
# `workflow["nodes"]` is the TOP LEVEL only: a subgraph instance is one opaque
# node whose `type` is its definition UUID, and the nodes it actually executes
# live under `definitions.subgraphs[].nodes`. `workflow_to_api` expands those
# interiors and then DROPS a muted/bypassed one — so the graph runs without the
# node and, before this, no reader of `ls-nodes` could tell.
#
# They are emitted under a NEW sibling key, never appended to `nodes[]`: the
# cloud agent renders every `nodes[]` entry as a model-visible line and pins
# that listing.
# ---------------------------------------------------------------------------

SG_UUID = "8f1e0a2c-0000-4000-8000-000000000001"
SG_UUID_2 = "8f1e0a2c-0000-4000-8000-000000000002"


def _sg_workflow(instance_ids=(10,), interior_mode=MODE_MUTED) -> dict:
    """Top-level EmptyLatentImage 7 plus one instance per id, all pointing at a
    single definition whose interior node 9 carries ``interior_mode``."""
    wf = {
        "last_node_id": 60,
        "last_link_id": 0,
        "nodes": [
            {"id": 7, "type": "EmptyLatentImage", "pos": [0, 0], "widgets_values": [512, 512, 1]},
        ],
        "links": [],
        "definitions": {
            "subgraphs": [
                {
                    "id": SG_UUID,
                    "name": "Text to Image",
                    "inputs": [],
                    "nodes": [
                        {"id": 9, "type": "CLIPTextEncode", "mode": interior_mode, "widgets_values": ["a cat"]},
                        {"id": 11, "type": "VAEDecode", "mode": 0},
                    ],
                    "links": [],
                }
            ]
        },
    }
    for nid in instance_ids:
        wf["nodes"].append({"id": nid, "type": SG_UUID, "pos": [100, 0]})
    return wf


def _env(tmp_path, wf: dict, capsys) -> dict:
    path = _write(tmp_path, wf)
    env = _run(["ls-nodes", str(path)], capsys)
    assert env["ok"] is True, env
    return env


def test_interior_node_mode_is_reported(patched_graph, tmp_path, capsys):
    """(i) a live instance 10 whose definition has interior node 9 with mode 2."""
    env = _env(tmp_path, _sg_workflow(), capsys)
    by_path = {r["path"]: r for r in env["data"]["subgraph_nodes"]}
    assert by_path["10/9"]["instance"] == "10"
    assert by_path["10/9"]["id"] == 9
    assert by_path["10/9"]["mode"] == "mute"
    assert by_path["10/9"]["type"] == "CLIPTextEncode"
    # a normally-executing interior stays label-free, same as the top-level rows
    assert "mode" not in by_path["10/11"], by_path["10/11"]


def test_top_level_nodes_unchanged_by_interiors(patched_graph, tmp_path, capsys):
    """`nodes[]` keeps its exact meaning — the instance stays ONE opaque row and
    no interior is appended. The cloud agent pins this listing."""
    env = _env(tmp_path, _sg_workflow(), capsys)
    assert [r["id"] for r in env["data"]["nodes"]] == [7, 10]
    assert env["data"]["count"] == 2
    assert env["data"]["subgraph_count"] == len(env["data"]["subgraph_nodes"]) == 2


def test_nested_instance_two_levels_deep(patched_graph, tmp_path, capsys):
    """(ii) 10/3/7, two levels deep, bypassed."""
    wf = _sg_workflow()
    wf["definitions"]["subgraphs"][0]["nodes"].append({"id": 3, "type": SG_UUID_2})
    wf["definitions"]["subgraphs"].append(
        {
            "id": SG_UUID_2,
            "name": "Inner",
            "inputs": [],
            "nodes": [{"id": 7, "type": "KSampler", "mode": MODE_BYPASS}],
            "links": [],
        }
    )
    env = _env(tmp_path, wf, capsys)
    by_path = {r["path"]: r for r in env["data"]["subgraph_nodes"]}
    assert by_path["10/3/7"]["mode"] == "bypass"
    assert by_path["10/3/7"]["instance"] == "10", "instance stays the TOP-LEVEL id"
    assert by_path["10/3/7"]["id"] == 7
    # the nested instance itself is a row too, so a reader can see the chain
    assert by_path["10/3"]["type"] == SG_UUID_2


def test_two_instances_of_same_definition_both_emit(patched_graph, tmp_path, capsys):
    """(iii) 10 and 11 share one definition — both must appear, addressed apart."""
    env = _env(tmp_path, _sg_workflow(instance_ids=(10, 11)), capsys)
    by_path = {r["path"]: r for r in env["data"]["subgraph_nodes"]}
    assert by_path["10/9"]["mode"] == by_path["11/9"]["mode"] == "mute"
    assert by_path["10/9"]["instance"] == "10"
    assert by_path["11/9"]["instance"] == "11"


def test_no_definitions_emits_empty_list(patched_graph, tmp_path, capsys):
    """(iv) the 99% workflow: the key is always present, and empty."""
    env = _env(tmp_path, _base_workflow(), capsys)
    assert env["data"]["subgraph_nodes"] == []
    assert env["data"]["subgraph_count"] == 0
    assert env["data"]["count"] == 2


def test_self_referencing_definition_terminates(patched_graph, tmp_path, capsys):
    """(v) a definition that contains an instance of ITSELF must not recurse
    forever. ComfyUI cannot author this; a hand-written or corrupt document can."""
    wf = _sg_workflow()
    wf["definitions"]["subgraphs"][0]["nodes"].append({"id": 5, "type": SG_UUID})
    env = _env(tmp_path, wf, capsys)
    paths = [r["path"] for r in env["data"]["subgraph_nodes"]]
    assert "10/5" in paths
    assert len(paths) == len(set(paths)), "addresses must stay unique"
    assert len(paths) < 100, f"cycle was not bounded: {len(paths)} rows"


def test_depth_cap_stops_a_long_nesting_chain(patched_graph, tmp_path, capsys):
    """A chain of distinct definitions deeper than `_MAX_SUBGRAPH_DEPTH` is
    truncated at the cap rather than walked to the bottom."""
    from comfy_cli.cql.engine import _MAX_SUBGRAPH_DEPTH

    depth = _MAX_SUBGRAPH_DEPTH + 5
    uuids = [f"8f1e0a2c-0000-4000-8000-{i:012d}" for i in range(depth)]
    subgraphs = []
    for i, u in enumerate(uuids):
        inner = [{"id": 1, "type": "VAEDecode"}]
        if i + 1 < depth:
            inner.append({"id": 2, "type": uuids[i + 1]})
        subgraphs.append({"id": u, "name": f"L{i}", "inputs": [], "nodes": inner, "links": []})
    wf = {
        "last_node_id": 60,
        "last_link_id": 0,
        "nodes": [{"id": 10, "type": uuids[0], "pos": [0, 0]}],
        "links": [],
        "definitions": {"subgraphs": subgraphs},
    }
    env = _env(tmp_path, wf, capsys)
    levels = {r["path"].count("/") for r in env["data"]["subgraph_nodes"]}
    assert max(levels) == _MAX_SUBGRAPH_DEPTH, sorted(levels)


def test_malformed_definitions_do_not_crash(patched_graph, tmp_path, capsys):
    """Non-dict entries in `nodes`/`subgraphs`, and an instance whose `type`
    resolves to nothing, are skipped rather than raising."""
    wf = _sg_workflow()
    wf["nodes"].append("not-a-node")
    wf["nodes"].append({"id": 12, "type": ["unhashable"]})
    wf["definitions"]["subgraphs"].append("not-a-subgraph")
    wf["definitions"]["subgraphs"][0]["nodes"].append(None)
    env = _env(tmp_path, wf, capsys)
    assert [r["path"] for r in env["data"]["subgraph_nodes"]] == ["10/9", "10/11"]


# ---------------------------------------------------------------------------
# Robustness of the interior walk (review of PR #845).
#
# `ls-nodes` is the one workflow command with no catalog and no validation gate
# in front of it — an agent points it at whatever JSON it was handed. Every case
# below raised an uncaught exception before the fix, so the CLI's contract (an
# error envelope, never a traceback) was broken by a corrupt file rather than a
# hostile one. Each was reproduced against the pre-fix walker.
# ---------------------------------------------------------------------------


def _wf_with_interior(interior_nodes) -> dict:
    """One live instance 10 whose definition's `nodes` is exactly what's given."""
    return {
        "last_node_id": 60,
        "last_link_id": 0,
        "nodes": [{"id": 10, "type": SG_UUID, "pos": [0, 0]}],
        "links": [],
        "definitions": {"subgraphs": [{"id": SG_UUID, "name": "S", "inputs": [], "nodes": interior_nodes}]},
    }


@pytest.mark.parametrize(
    "wf, why",
    [
        (_wf_with_interior(1), "a definition's `nodes` is a truthy scalar: survives `or []`, TypeError on iteration"),
        (
            _wf_with_interior([{"id": 9, "type": "X", "properties": ["a"]}]),
            "truthy non-dict `properties`: survives `or {}`, AttributeError on .get "
            "(a falsy `[]` never reached the bug)",
        ),
        (
            _wf_with_interior([{"id": 9, "type": "X", "mode": []}]),
            "unhashable `mode`: _MODE_LABELS.get hashes its argument -> TypeError",
        ),
        (
            {"nodes": [], "links": [], "definitions": 5},
            "`definitions` is not a dict: AttributeError inside _subgraph_defs_by_id",
        ),
        (
            {"nodes": [], "links": [], "definitions": {"subgraphs": 5}},
            "`subgraphs` is a truthy scalar: TypeError iterating it",
        ),
    ],
)
def test_malformed_shapes_emit_an_envelope_not_a_traceback(patched_graph, tmp_path, capsys, wf, why):
    env = _env(tmp_path, wf, capsys)
    assert env["ok"] is True, why
    # A malformed container is SKIPPED, never guessed at: the walk yields no row
    # for it rather than inventing one.
    assert isinstance(env["data"]["subgraph_nodes"], list), why


def test_unhashable_mode_on_a_top_level_node_does_not_crash(patched_graph, tmp_path, capsys):
    """The same hash hazard on the pre-existing top-level row builder: an
    unhashable `mode` must fall through to "no label", not abort the command."""
    wf = _base_workflow()
    wf["nodes"][0]["mode"] = []
    env = _env(tmp_path, wf, capsys)
    assert all("mode" not in r for r in env["data"]["nodes"]), env["data"]["nodes"]


def test_interior_path_has_no_leading_separator_when_instance_has_no_id(patched_graph, tmp_path, capsys):
    """An instance with no `id` gave `/9`, whose empty leading segment resolves
    as no slot address. Join like the slot walker: separator only after a
    non-empty prefix."""
    wf = _wf_with_interior([{"id": 9, "type": "CLIPTextEncode"}])
    del wf["nodes"][0]["id"]
    env = _env(tmp_path, wf, capsys)
    assert [r["path"] for r in env["data"]["subgraph_nodes"]] == ["9"]


def test_branching_definition_graph_is_bounded_and_flagged(patched_graph, tmp_path, capsys):
    """The depth cap and the per-path `seen_defs` set bound CYCLES and LINEAR
    nesting, but not BRANCHING: definitions that each hold two instances of the
    next repeat no definition on any path and stay under the depth cap, yet
    expand a few-KB file into ~2**depth rows. Only a total row budget bounds it,
    and the consumer has to be told the listing is short."""
    from comfy_cli.command.workflow_edit import _MAX_SUBGRAPH_INTERIOR_ROWS

    uuids = [f"8f1e0a2c-0000-4000-8000-{i:012d}" for i in range(20)]
    subgraphs = []
    for i, u in enumerate(uuids):
        inner = [{"id": 1, "type": "VAEDecode"}]
        if i + 1 < len(uuids):  # two instances of the NEXT definition -> 2**i growth
            inner += [{"id": 2, "type": uuids[i + 1]}, {"id": 3, "type": uuids[i + 1]}]
        subgraphs.append({"id": u, "name": f"L{i}", "inputs": [], "nodes": inner, "links": []})
    wf = {
        "last_node_id": 60,
        "last_link_id": 0,
        "nodes": [{"id": 10, "type": uuids[0], "pos": [0, 0]}],
        "links": [],
        "definitions": {"subgraphs": subgraphs},
    }
    env = _env(tmp_path, wf, capsys)
    assert len(env["data"]["subgraph_nodes"]) == _MAX_SUBGRAPH_INTERIOR_ROWS
    assert env["data"]["subgraph_count"] == _MAX_SUBGRAPH_INTERIOR_ROWS
    assert env["data"]["subgraph_truncated"] is True


def test_ordinary_workflow_carries_no_truncation_key(patched_graph, tmp_path, capsys):
    """`subgraph_truncated` is label-only-when-set, like `mode`."""
    env = _env(tmp_path, _sg_workflow(), capsys)
    assert "subgraph_truncated" not in env["data"], env["data"]


# ---------------------------------------------------------------------------
# Pretty mode: node text is workflow-file text, and Rich reads a `str` cell as
# MARKUP. `ls-nodes` is pointed at files an agent did not author (downloaded
# templates, a peer's export), so a title carrying `[/]` crashed the render with
# MarkupError and `[link=...]` rendered a live OSC 8 hyperlink. Same contract as
# tests/comfy_cli/command/test_pretty_print_sanitize.py, applied to both tables.
# ---------------------------------------------------------------------------

HOSTILE = "\x1b[2J\x1b]0;PWNED\x07boom [link=https://attacker.example]click[/link]"
UNBALANCED_TITLE = "title [/] oops"


@pytest.fixture
def pretty_stream(monkeypatch):
    """A pretty renderer over a stream Rich treats as a tty — `force_terminal`
    is what makes Rich emit OSC 8 at all, so without it half the hazard hides."""
    import io

    from comfy_cli.output.renderer import OutputMode, Renderer, set_renderer

    monkeypatch.setenv("FORCE_COLOR", "1")
    monkeypatch.setenv("COLUMNS", "300")
    stream = io.StringIO()
    r = Renderer.resolve(is_stdout_tty=True, env={}, caller=None)
    r.mode = OutputMode.PRETTY
    r.pretty_stream = stream
    set_renderer(r)
    return stream


def _render_pretty(tmp_path, wf: dict) -> str:
    from typer.testing import CliRunner

    from comfy_cli.command import workflow as workflow_cmd

    path = _write(tmp_path, wf)
    result = CliRunner().invoke(workflow_cmd.app, ["ls-nodes", str(path)], standalone_mode=False)
    assert result.exception is None, result.exception
    return result


def _assert_inert(out: str) -> None:
    import re

    assert "\x1b]8;" not in out, "markup was rendered into a live OSC 8 hyperlink"
    assert "\x1b]0;" not in out, "OSC 0 window-title sequence reached the terminal"
    assert "\x1b[2J" not in out, "CSI 2J screen-clear reached the terminal"
    residue = re.sub(r"\x1b\[[0-9;]*m", "", out)  # Rich's own SGR styling
    assert "\x1b" not in residue, f"escape byte survived: {residue!r}"


def test_pretty_top_level_row_text_is_inert(patched_graph, pretty_stream, tmp_path):
    wf = _base_workflow()
    wf["nodes"][0]["title"] = HOSTILE
    _render_pretty(tmp_path, wf)
    _assert_inert(pretty_stream.getvalue())


def test_pretty_interior_row_text_is_inert(patched_graph, pretty_stream, tmp_path):
    wf = _wf_with_interior([{"id": 9, "type": "CLIPTextEncode", "title": HOSTILE}])
    _render_pretty(tmp_path, wf)
    _assert_inert(pretty_stream.getvalue())


@pytest.mark.parametrize(
    "wf_factory",
    [
        pytest.param(lambda: _wf_with_interior([{"id": 9, "type": UNBALANCED_TITLE}]), id="interior-type"),
        pytest.param(
            lambda: _wf_with_interior([{"id": 9, "type": "X", "title": UNBALANCED_TITLE}]), id="interior-title"
        ),
        pytest.param(lambda: _wf_with_interior([{"id": UNBALANCED_TITLE, "type": "X"}]), id="interior-path"),
    ],
)
def test_pretty_unbalanced_markup_does_not_crash_the_render(patched_graph, pretty_stream, tmp_path, wf_factory):
    """An unbalanced `[/]` raises rich.errors.MarkupError mid-render, so the CLI
    dies while merely printing a table."""
    _render_pretty(tmp_path, wf_factory())
    assert "oops" in pretty_stream.getvalue()
