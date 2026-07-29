"""Server-supplied text must not reach the terminal live from *direct*
``rich.print`` sites (BE-4794).

``comfy_cli.output.sanitize`` covers everything routed through ``Renderer``,
but ``Renderer`` is not the only path to the terminal: command modules print
under their own ``is_pretty()`` gates, and those strings never touch the
renderer's helpers. Two distinct defects live there, both reachable from a
hostile ComfyUI host on ``comfy run --host/--port``:

  1. raw ANSI in the payload reaches the terminal (screen clear, window-title
     rewrite, repaint-to-spoof), and
  2. Rich *manufactures* new escapes from markup it finds — and an unbalanced
     ``[/]`` raises ``rich.errors.MarkupError``, hard-crashing the CLI while it
     is merely printing an error line.

These tests drive the real call sites in pretty mode and assert both. Each was
verified to fail before the corresponding ``sanitize_markup`` call was added.
"""

from __future__ import annotations

import io
import json
import re
import urllib.error
from unittest.mock import MagicMock, patch

import pytest
import typer

from comfy_cli.command.run.execution import WorkflowExecution
from comfy_cli.output.renderer import OutputMode, Renderer, reset_renderer_for_testing, set_renderer

# A payload carrying every hazard at once: CSI 2J clears the screen, OSC 0
# rewrites the window title, and the markup forges an OSC 8 hyperlink.
EVIL = "\x1b[2J\x1b]0;PWNED\x07boom [link=https://attacker.example]click[/link]"
UNBALANCED = "server said [/] oops"

# Rich's own styling. Anything left after stripping these came from the payload.
_RICH_SGR_RE = re.compile(r"\x1b\[[0-9;]*m")

# Rich's ``Live`` display additionally emits cursor show/hide, erase-line and
# cursor-up. Those are Rich's, not the payload's — a `Live`-backed site strips
# them too, but nothing wider: the screen-clear and OSC forms the attacker
# needs are deliberately absent from this allowlist.
_RICH_LIVE_RE = re.compile(r"\x1b\[(?:\?25[lh]|[0-9]*[AK])")


@pytest.fixture(autouse=True)
def reset_singleton():
    reset_renderer_for_testing()
    yield
    reset_renderer_for_testing()


@pytest.fixture
def pretty(monkeypatch):
    """Install a pretty renderer writing to a StringIO Rich treats as a tty.

    ``force_terminal`` matters: Rich only emits OSC 8 hyperlinks to a terminal,
    so without it the markup half of the attack is invisible. ``COLUMNS`` is
    pinned because the assertions look for un-wrapped substrings.
    """
    monkeypatch.setenv("FORCE_COLOR", "1")
    monkeypatch.setenv("COLUMNS", "300")
    stream = io.StringIO()
    r = Renderer.resolve(is_stdout_tty=True, env={}, caller=None)
    r.mode = OutputMode.PRETTY
    r.pretty_stream = stream
    set_renderer(r)
    return stream


def assert_inert(stream: io.StringIO, *, live: bool = False) -> str:
    """The payload rendered as literal text: no live escape, nothing executed.

    Set ``live=True`` for a site backed by Rich's ``Live``, which emits its own
    cursor/erase control codes on top of the SGR styling.
    """
    out = stream.getvalue()
    assert "\x1b]8;" not in out, "markup was rendered into a live OSC 8 hyperlink"
    assert "\x1b]0;" not in out, "OSC 0 window-title sequence reached the terminal"
    assert "\x1b[2J" not in out, "CSI 2J screen-clear reached the terminal"
    residue = _RICH_SGR_RE.sub("", out)
    if live:
        residue = _RICH_LIVE_RE.sub("", residue)
    assert "\x1b" not in residue, f"escape byte survived: {residue!r}"
    return residue


def _execution(workflow=None, *, verbose=False):
    return WorkflowExecution(
        workflow=workflow if workflow is not None else {},
        host="127.0.0.1",
        port=8188,
        verbose=verbose,
        local_paths=False,
        progress=None,
        timeout=30,
    )


def _http_error(status: int, body: bytes) -> urllib.error.HTTPError:
    return urllib.error.HTTPError(
        url="http://127.0.0.1:8188/prompt", code=status, msg="err", hdrs=None, fp=io.BytesIO(body)
    )


def _queue_against(ex, exc):
    """Run ``ex.queue()`` with ``/prompt`` raising ``exc``. Always exits 1."""
    with patch("comfy_cli.command.run.execution.request.urlopen", side_effect=exc):
        with pytest.raises(typer.Exit):
            ex.queue()


# ---------------------------------------------------------------------------
# execution.py — the HTTP error body the reviewer proved still leaked
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("status", [400, 500])
def test_http_error_body_is_inert(pretty, status):
    """`comfy run` against a hostile host: the body is printed verbatim two
    lines above the (already covered) error panel."""
    _queue_against(_execution(), _http_error(status, EVIL.encode()))
    assert "boom" in assert_inert(pretty)


@pytest.mark.parametrize("status", [400, 500])
def test_http_error_body_with_unbalanced_markup_does_not_crash(pretty, status):
    """An unbalanced ``[/]`` in the body used to raise ``MarkupError`` — a
    remote server could hard-crash the client."""
    _queue_against(_execution(), _http_error(status, UNBALANCED.encode()))
    assert "oops" in assert_inert(pretty)


def test_node_errors_json_dump_is_inert(pretty):
    """``json.dumps`` neutralizes the ESC byte but NOT ``[``/``]`` — the markup
    half of the attack survives the JSON encoding."""
    body = json.dumps({"node_errors": {"1": {"errors": [{"message": EVIL}], "class_type": "X"}}}).encode()
    _queue_against(_execution(), _http_error(400, body))
    assert "boom" in assert_inert(pretty)


def test_node_errors_json_dump_with_unbalanced_markup_does_not_crash(pretty):
    body = json.dumps({"node_errors": {"1": {"errors": [{"message": UNBALANCED}], "class_type": "X"}}}).encode()
    _queue_against(_execution(), _http_error(400, body))
    assert "oops" in assert_inert(pretty)


def test_connection_error_reason_is_inert(pretty):
    _queue_against(_execution(), urllib.error.URLError(EVIL))
    assert_inert(pretty)


def test_execution_error_payload_is_inert(pretty):
    """The ``execution_error`` WS frame is server-authored end to end."""
    ex = _execution()
    with pytest.raises(typer.Exit):
        ex.on_error({"node_id": "1", "exception_message": EVIL, "traceback": [UNBALANCED]})
    assert "boom" in assert_inert(pretty)


# ---------------------------------------------------------------------------
# execution.py — --verbose node log (node_id off the wire, title from workflow)
# ---------------------------------------------------------------------------


def test_log_node_known_node_is_inert(pretty):
    ex = _execution({"1": {"class_type": EVIL, "inputs": {}, "_meta": {"title": UNBALANCED}}}, verbose=True)
    ex.log_node("Executing", "1")
    assert_inert(pretty)


def test_log_node_unknown_node_id_is_inert(pretty):
    ex = _execution({}, verbose=True)
    ex.log_node("Executing", EVIL)
    assert_inert(pretty)


# ---------------------------------------------------------------------------
# preflight.py — validation warnings, field names derived from object_info
# ---------------------------------------------------------------------------


def test_preflight_warnings_are_inert(pretty):
    from comfy_cli.command.run import preflight

    graph = MagicMock()
    graph.validate_workflow.return_value = {
        "valid": True,
        "warnings": [{"field": UNBALANCED, "message": EVIL}],
    }
    with patch("comfy_cli.cql.engine.Graph") as g:
        g.from_object_info.return_value = graph
        preflight._preflight_validate(preflight.get_renderer(), {"1": {}}, {"X": {}})
    assert "boom" in assert_inert(pretty)


# ---------------------------------------------------------------------------
# watcher.py — output URLs echoed back from the job state file
# ---------------------------------------------------------------------------


def test_watcher_output_urls_are_inert(pretty):
    from comfy_cli.command.run import watcher

    state = MagicMock()
    state.status = "completed"
    state.is_terminal = True
    state.outputs = [EVIL, UNBALANCED]
    with patch("comfy_cli.jobs_state.read", return_value=state):
        watcher._tail_state_file("p" * 12, seconds=0.0)
    assert "boom" in assert_inert(pretty, live=True)


# ---------------------------------------------------------------------------
# models/search.py — the cloud asset catalog is server-supplied too
# ---------------------------------------------------------------------------


#
# `Table.add_row` parses markup in a `str` cell — the same sink the error
# panel's `_kv_table` already covers — so the list/search tables need it too.


def _hostile_asset(name: str) -> dict:
    return {
        "id": "a1",
        "name": name,
        "display_name": name,
        "tags": ["models", UNBALANCED],
        "size": None,
        "preview_url": EVIL,
        "metadata": {"base_model": EVIL, "repo_url": EVIL, "trained_words": [UNBALANCED]},
    }


@pytest.fixture
def cloud_catalog():
    """Point the models commands at a cloud target serving a hostile catalog."""
    target = MagicMock()
    target.is_cloud = True
    target.url.return_value = "https://cloud.example/assets"
    with (
        patch("comfy_cli.target.resolve_target", return_value=target),
        patch("comfy_cli.command.models.search._http_get_json") as get_json,
    ):
        yield get_json


def test_models_show_fields_are_inert(pretty, cloud_catalog):
    from comfy_cli.command.models.search import show_cmd

    cloud_catalog.return_value = {"assets": [_hostile_asset(EVIL)], "has_more": False}
    show_cmd(name=EVIL)
    assert "boom" in assert_inert(pretty)


def test_models_search_table_cells_are_inert(pretty, cloud_catalog):
    from comfy_cli.command.models.search import search_cmd

    cloud_catalog.return_value = {"assets": [_hostile_asset(EVIL)], "has_more": False}
    search_cmd(text=None, type_=None, limit=20, include_public=True, where=None)
    assert_inert(pretty)


def test_models_list_folders_table_cells_are_inert(pretty, cloud_catalog):
    from comfy_cli.command.models.search import list_folders_cmd

    cloud_catalog.return_value = [{"name": EVIL, "folders": [UNBALANCED]}]
    list_folders_cmd(where=None)
    assert_inert(pretty)
