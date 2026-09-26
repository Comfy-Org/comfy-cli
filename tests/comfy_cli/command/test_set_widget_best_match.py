"""`set-widget` leads an unknown_enum_value refusal with its best_match.

See tests/comfy_cli/cql/test_combo_best_match.py for the motivating case
('16:9 (Landscape)' vs '16:9 (Widescreen)' on ResolutionSelector).
"""

from __future__ import annotations

import json

from test_workflow_edit import (  # type: ignore[import-not-found]
    _run,
    reset_singleton,  # noqa: F401  (autouse fixture)
)

from comfy_cli import workflow_ops
from comfy_cli.command import workflow_edit
from comfy_cli.cql.engine import Graph

RATIOS = [
    "1:1 (Square)",
    "4:3 (Standard)",
    "9:16 (Portrait Widescreen)",
    "16:9 (Widescreen)",
    "21:9 (Ultrawide)",
]


def test_set_widget_refusal_leads_with_the_best_match(tmp_path, capsys, monkeypatch):
    graph = Graph.from_object_info(
        {
            "ResolutionSelector": {
                "input": {"required": {"aspect_ratio": [RATIOS, {}]}},
                "input_order": {"required": ["aspect_ratio"]},
                "output": ["INT", "INT"],
                "output_name": ["width", "height"],
                "category": "utils",
                "display_name": "Resolution Selector",
                "description": "",
                "output_node": False,
                "python_module": "comfy_extras",
            }
        }
    )
    wf = {"nodes": [], "links": [], "last_node_id": 0, "last_link_id": 0, "version": 0.4}
    wf, op = workflow_ops.add_node(wf, graph, "ResolutionSelector")
    path = tmp_path / "wf.json"
    path.write_text(json.dumps(wf))
    before = path.read_text()

    monkeypatch.setattr(workflow_edit, "_get_graph", lambda *a, **kw: graph)
    env = _run(["set-widget", str(path), f"{op['node_id']}.aspect_ratio", "16:9 (Landscape)"], capsys)
    assert env["ok"] is False
    err = env["error"]
    assert err["code"] == "unknown_enum_value"
    assert err["details"]["best_match"] == "16:9 (Widescreen)", err
    assert err["hint"].startswith("use '16:9 (Widescreen)'"), err
    assert path.read_text() == before, "never auto-apply the guess"
