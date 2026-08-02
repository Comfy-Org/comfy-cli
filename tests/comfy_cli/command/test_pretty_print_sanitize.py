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
from comfy_cli.output import get_renderer
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
    with patch("comfy_cli.command.run.execution.no_redirect_urlopen", side_effect=exc):
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
# nodes.py — `object_info` off the same --host/--port the run path uses
# ---------------------------------------------------------------------------

# Every string here is chosen by whoever answers /object_info.
HOSTILE_OBJECT_INFO = {
    EVIL: {
        "input": {"required": {UNBALANCED: ["IMAGE", {}]}},
        "output": [EVIL],
        "output_name": [EVIL],
        "name": EVIL,
        "display_name": UNBALANCED,
        "description": EVIL,
        "category": UNBALANCED,
    }
}


@pytest.fixture
def hostile_object_info():
    with patch("comfy_cli.cql.loader.resilient_load_object_info", return_value=HOSTILE_OBJECT_INFO):
        yield


def test_nodes_ls_is_inert(pretty, hostile_object_info):
    from comfy_cli.command.nodes import ls_cmd

    ls_cmd(
        produces=None,
        accepts=None,
        category=None,
        pack=None,
        label=None,
        cloud_disabled=False,
        api_only=False,
        output_only=False,
        exclude_deprecated=False,
        limit=None,
        input_path=None,
        host=None,
        port=None,
        where=None,
    )
    assert_inert(pretty)


def test_nodes_show_is_inert(pretty, hostile_object_info):
    """`comfy nodes show --host <hostile>` — same flag pair as `comfy run`."""
    from comfy_cli.command.nodes import show_cmd

    show_cmd(name=EVIL, where=None, input_path=None, host=None, port=None)
    assert "boom" in assert_inert(pretty)


def test_nodes_search_is_inert(pretty, hostile_object_info):
    from comfy_cli.command.nodes import search_cmd

    search_cmd(query="boom", limit=20, input_path=None, host=None, port=None, where=None)
    assert_inert(pretty)


def test_nodes_types_is_inert(pretty, hostile_object_info):
    from comfy_cli.command.nodes import types_cmd

    types_cmd(limit=None, input_path=None, host=None, port=None, where=None)
    assert_inert(pretty)


def test_nodes_categories_is_inert(pretty, hostile_object_info):
    from comfy_cli.command.nodes import categories_cmd

    categories_cmd(prefix=None, limit=None, input_path=None, host=None, port=None, where=None)
    assert_inert(pretty)


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


# ---------------------------------------------------------------------------
# models/models.py — the download-server error text (BE-5023)
# ---------------------------------------------------------------------------
#
# `comfy model download --url <host>` renders whatever the host put in a failed
# response. Only one branch of `guess_status_code_reason` echoes server text —
# the 401 one, which interpolates the response body's JSON `message` — so a
# CivitAI-shaped mirror or a MITM'd model host answering 401 chooses the string
# that lands on the terminal. These drive that real body through the real print
# sites: `download`'s `except DownloadException` line and the per-row error line
# `download-status` / `downloads` print under the table.


class _FakeErrorResponse:
    """An `httpx.stream` context manager that fails with a chosen body."""

    def __init__(self, status_code: int, body: bytes):
        self.status_code = status_code
        self.headers = {}
        self._body = body

    def read(self) -> bytes:
        return self._body

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def _hostile_401_body(message: str) -> bytes:
    return json.dumps({"message": message}).encode()


@pytest.fixture
def download_workspace(tmp_path, monkeypatch):
    """Point `models.get_workspace()` at a per-test directory."""
    from comfy_cli.command.models import models

    ws = tmp_path / "workspace"
    ws.mkdir()
    monkeypatch.setattr(models, "get_workspace", lambda: ws)
    return ws


def _download_against_401(body: bytes) -> None:
    """`comfy model download` where the host answers 401 with `body`.

    Nothing is stubbed between the response and the print site: the real
    `guess_status_code_reason` builds the reason, the real `DownloadException`
    carries it, and the real `except` clause renders it.
    """
    from comfy_cli.command.models import models

    with patch("httpx.stream", return_value=_FakeErrorResponse(401, body)):
        with pytest.raises(typer.Exit):
            models.download(
                None,
                url="https://hostile.example/m.safetensors",
                relative_path="models/loras",
                filename="m.safetensors",
            )


def test_guess_status_code_reason_401_strips_server_escapes():
    """The boundary itself, so every consumer of the reason benefits — the
    `comfy node install` path renders it too, without markup interpretation."""
    from comfy_cli.file_utils import guess_status_code_reason

    reason = guess_status_code_reason(401, _hostile_401_body(EVIL))

    assert "\x1b" not in reason
    assert "boom" in reason


def test_guess_status_code_reason_leaves_benign_text_alone():
    """The sanitizer is a no-op on a realistic message — no silent mangling."""
    from comfy_cli.file_utils import guess_status_code_reason

    assert "Your API key is invalid" in guess_status_code_reason(401, _hostile_401_body("Your API key is invalid"))


def test_download_error_is_inert(pretty, download_workspace):
    _download_against_401(_hostile_401_body(EVIL))
    assert "boom" in assert_inert(pretty)


def test_download_error_with_unbalanced_markup_does_not_crash(pretty, download_workspace):
    """An unbalanced `[/]` in the server's message used to be the `escape()`
    call's whole job; `sanitize_markup` keeps that guarantee."""
    _download_against_401(_hostile_401_body(UNBALANCED))
    assert "oops" in assert_inert(pretty)


def test_download_status_row_error_is_inert(pretty):
    """The poll verbs re-print the failure recorded in the state file, which a
    detached worker wrote from the same server-chosen reason."""
    from comfy_cli.command.models.models import _render_download_rows

    _render_download_rows(
        [
            {
                "id": "abc123",
                "status": "failed",
                "percent": None,
                "completed_bytes": None,
                "total_bytes": None,
                "elapsed_seconds": 1.0,
                "dest": "/tmp/m.safetensors",
                "error": EVIL,
            }
        ]
    )
    assert "boom" in assert_inert(pretty)


def test_download_status_row_error_with_unbalanced_markup_does_not_crash(pretty):
    from comfy_cli.command.models.models import _render_download_rows

    _render_download_rows(
        [
            {
                "id": "abc123",
                "status": "failed",
                "percent": None,
                "completed_bytes": None,
                "total_bytes": None,
                "elapsed_seconds": 1.0,
                "dest": "/tmp/m.safetensors",
                "error": UNBALANCED,
            }
        ]
    )
    assert "oops" in assert_inert(pretty)


# ---------------------------------------------------------------------------
# jobs.py / system.py / workflow.py — `Table.add_row` cells (BE-6037)
# ---------------------------------------------------------------------------
#
# `sanitize.py` names `Table.add_row` as a markup-interpreting sink, and the
# `models`/`nodes` tables above already honor that. These four command modules
# did not import `sanitize_markup` at all, so a hostile host's `prompt_id`,
# `error_message`, device name or workflow id reached the sink raw — a live
# OSC 8 hyperlink, a repaint of earlier output, or a `MarkupError` crash while
# the CLI is merely rendering a status table.


def test_local_job_status_table_is_inert(pretty):
    """`comfy jobs status` against a hostile `--host/--port`: prompt_id, the
    joined output list and the execution error all come off `/history`."""
    from comfy_cli.command.jobs import _render_status_pretty

    _render_status_pretty(
        {
            "prompt_id": EVIL,
            "status": "completed",
            "outputs": [EVIL, UNBALANCED],
            "error": UNBALANCED,
        },
        host="127.0.0.1",
        port=8188,
    )
    assert "boom" in assert_inert(pretty)


def test_local_job_status_unknown_status_does_not_crash(pretty):
    """An unrecognized `status` falls through to `Text(status)`, which never
    parses markup — the docstring's carve-out, asserted so a later "sanitize
    everything" pass cannot quietly start printing backslashes here."""
    from comfy_cli.command.jobs import _render_status_pretty

    _render_status_pretty(
        {"prompt_id": "p1", "status": UNBALANCED, "outputs": [], "error": None},
        host="127.0.0.1",
        port=8188,
    )
    out = assert_inert(pretty)
    assert "server said [/] oops" in out, f"Text cell was escaped or mangled: {out!r}"


def _cloud_status_against(snap: dict) -> None:
    """Run `_cloud_status` with the cloud snapshot replaced by `snap`."""
    from comfy_cli.command import jobs

    with (
        patch.object(jobs, "cloud_preflight_or_exit"),
        patch.object(jobs, "_cloud_status_snapshot", return_value=snap),
    ):
        jobs._cloud_status("p" * 12)


def _cloud_snap(**overrides) -> dict:
    snap = {
        "prompt_id": "p" * 12,
        "status": "completed",
        "outputs": [],
        "outputs_by_node": {},
        "outputs_by_item": {},
        "assigned_inference": None,
        "error_message": None,
        "created_at": None,
        "updated_at": None,
        "base_url": "https://cloud.example",
    }
    snap.update(overrides)
    return snap


def test_cloud_job_status_table_is_inert(pretty):
    """`comfy jobs status --where cloud`: every cell is a field of the
    `/api/jobs/<id>` response."""
    _cloud_status_against(
        _cloud_snap(
            status=EVIL,
            assigned_inference=EVIL,
            created_at=EVIL,
            updated_at=UNBALANCED,
            error_message=EVIL,
            outputs=[EVIL, UNBALANCED],
        )
    )
    assert "boom" in assert_inert(pretty)


def test_cloud_error_message_with_markup_renders_literally(pretty):
    """The ticket's headline case: a markup-bearing `error_message` must render
    as text and must not raise `MarkupError` mid-table."""
    _cloud_status_against(_cloud_snap(status="error", error_message=UNBALANCED))
    out = assert_inert(pretty)
    assert "server said [/] oops" in out, f"error_message did not render literally: {out!r}"


def test_cloud_output_url_with_markup_renders_literally(pretty):
    """An output URL is the other realistic carrier — one row per URL."""
    _cloud_status_against(_cloud_snap(outputs=["https://cdn.example/a.png[/]"]))
    out = assert_inert(pretty)
    assert "https://cdn.example/a.png[/]" in out, f"output URL did not render literally: {out!r}"


def test_jobs_ls_table_is_inert(pretty):
    """`comfy jobs ls` — prompt_id comes off `/queue` + `/history`, and the
    workflow column is a filename, which can carry `[...]` just as easily.

    The status cell goes through `status_glyph`, which *is* markup by design:
    an unrecognized status is echoed into it, so only the echoed half is
    escaped. Driving `EVIL` through it here is what proves that split holds.
    """
    from comfy_cli.command.jobs import JobRow, _render_jobs_pretty

    _render_jobs_pretty(
        [
            JobRow(
                prompt_id=EVIL,
                status=EVIL,
                queue_position=None,
                elapsed_seconds=None,
                workflow_size=3,
                outputs=1,
                workflow_path=f"/tmp/{UNBALANCED}.json",
            )
        ],
        host="127.0.0.1",
        port=8188,
    )
    assert_inert(pretty)


def test_status_glyph_keeps_its_own_tags_live(pretty):
    """The escaping must not neuter the tags `status_glyph` itself authors —
    a known status still renders styled, with no stray backslashes."""
    from rich.console import Console

    from comfy_cli.output.glyphs import status_glyph

    console = Console(file=pretty, force_terminal=True, width=80)
    console.print(status_glyph("success"))
    out = pretty.getvalue()

    assert "✓ completed" in out, f"styled label was mangled: {out!r}"
    assert "\\" not in out, f"escaping leaked backslashes into a benign status: {out!r}"
    assert _RICH_SGR_RE.search(out), "the bold-green style we author was dropped"


def test_system_stats_table_is_inert(pretty):
    """`comfy system-stats` — device name/type/index and the ComfyUI version
    line are all `/system_stats` fields."""
    from comfy_cli.command.system import _render_stats_pretty

    _render_stats_pretty(
        get_renderer(),
        {
            "devices": [{"name": EVIL, "type": UNBALANCED, "index": EVIL, "vram_free": 1, "vram_total": 2}],
            "system": {"ram_free": 1, "ram_total": 2, "comfyui_version": UNBALANCED},
        },
    )
    assert "boom" in assert_inert(pretty)


def test_local_workflow_list_table_is_inert(pretty):
    """`comfy workflow list --where local` — the rows are the `/userdata`
    listing verbatim, including a `size` the server chose."""
    from comfy_cli.command import workflow

    listing = json.dumps([{"path": EVIL, "size": UNBALANCED, "modified": 0, "created": 0}]).encode()
    target = MagicMock()
    target.url.return_value = "http://127.0.0.1:8188/userdata"
    with patch.object(workflow, "_userdata_request", return_value=(200, listing)):
        workflow._local_list(get_renderer(), target, name=None, limit=50, sort="created", order="desc")
    assert "boom" in assert_inert(pretty)


def test_cloud_workflow_list_table_is_inert(pretty):
    """The cloud saved-workflow catalog reaches the same sink."""
    from comfy_cli.command import workflow

    body = {
        "data": [
            {
                "id": EVIL,
                "name": UNBALANCED,
                "latest_version": EVIL,
                "updated_at": UNBALANCED,
            }
        ]
    }
    target = MagicMock()
    target.is_cloud = True
    target.url.return_value = "https://cloud.example/api/workflows"
    with (
        patch.object(workflow, "_resolve_where_target", return_value=target),
        patch.object(workflow, "_http_request", return_value=(200, body)),
    ):
        workflow.list_cmd(name=None, limit=20, sort="create_time", order="desc", where=None)
    assert_inert(pretty)
