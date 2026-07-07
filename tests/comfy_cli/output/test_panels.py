"""Snapshot/fragment tests for the new pretty-mode panels."""

from __future__ import annotations

import io
import re

from rich.console import Console

from comfy_cli.output.panels import (
    auth_empty_panel,
    auth_list_table,
    discover_panel,
    error_panel,
    which_panel,
)

_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")


def _render(renderable) -> str:
    """Render to a string with ANSI escapes stripped, so fragment asserts work."""
    buf = io.StringIO()
    console = Console(file=buf, force_terminal=True, width=100, no_color=True)
    console.print(renderable)
    return _ANSI_RE.sub("", buf.getvalue())


# ---------------------------------------------------------------------------
# error_panel
# ---------------------------------------------------------------------------


def test_error_panel_includes_code_and_message():
    out = _render(
        error_panel(
            code="cql_no_graph",
            message="failed to reach http://127.0.0.1:8188/object_info",
            hint="pass --input <path>",
        )
    )
    assert "error" in out
    assert "cql_no_graph" in out
    assert "failed to reach" in out
    assert "pass --input" in out
    assert "→" in out  # hint arrow


def test_error_panel_renders_without_hint_or_details():
    out = _render(error_panel(code="boom", message="something broke"))
    assert "boom" in out
    assert "something broke" in out
    # No hint line, no details section.
    assert "→" not in out
    assert "details" not in out.lower()


def test_error_panel_renders_details():
    out = _render(
        error_panel(
            code="server_not_running",
            message="not running",
            hint="run: comfy launch",
            details={"host": "127.0.0.1", "port": 8188},
        )
    )
    assert "details" in out
    assert "127.0.0.1" in out
    assert "8188" in out


# ---------------------------------------------------------------------------
# discover_panel
# ---------------------------------------------------------------------------


def test_discover_panel_shows_surface_counts_and_caps():
    doc = {
        "version": "0.0.0",
        "commands": {
            "run": {"name": "run", "help": "Run an API workflow.", "hidden": False},
            "nodes": {
                "name": "nodes",
                "help": "Introspect ComfyUI node classes.",
                "hidden": False,
                "subcommands": {
                    "ls": {"name": "ls", "help": "List node classes.", "hidden": False},
                    "show": {"name": "show", "help": "Show full schema.", "hidden": False},
                },
            },
            "_secret": {"name": "_secret", "hidden": True},
        },
        "schemas": {"a": {}, "b": {}, "c": {}},
        "error_codes": [{}, {}],
        "capabilities": {
            "json_envelope": True,
            "cql": True,
            "where_routing": True,
            "where_targets": ["local", "cloud"],
            "auth_providers": ["comfy-cloud", "civitai"],
        },
    }
    out = _render(discover_panel(doc, command_count=42))
    assert "comfy CLI" in out or "comfy-cli" in out
    assert "v0.0.0" in out
    assert "Commands" in out
    assert "run" in out  # visible command listed
    assert "nodes (2)" in out  # subcommand count shown
    assert "_secret" not in out  # hidden command excluded
    assert "Run an API workflow." in out  # description shown
    assert "Capabilities" in out
    assert "✓ cql" in out
    assert "✓ json_envelope" in out
    assert "Routing" in out
    assert "local · cloud" in out
    assert "comfy-cloud · civitai" in out
    assert "Summary" in out
    assert "42 total" in out
    assert "discover" in out
    assert "comfy CLI v0.0.0" in out


def test_discover_panel_handles_empty_capabilities():
    doc = {"version": "0.0.0", "commands": {}, "schemas": {}, "error_codes": [], "capabilities": {}}
    out = _render(discover_panel(doc, command_count=0))
    assert "Commands" in out
    assert "(none advertised)" in out


def test_discover_panel_uses_canonical_branded_subtitle():
    """The chrome (title + subtitle) goes through ``branded_panel`` so
    ``discover`` looks identical to ``env``, ``jobs ls``, etc."""
    doc = {"version": "0.0.0", "commands": {}, "schemas": {}, "error_codes": [], "capabilities": {}}
    panel = discover_panel(doc, command_count=42)
    assert "discover" in str(panel.title)
    assert "comfy CLI v0.0.0" in str(panel.subtitle)
    # The old "agent-aware CLI" subtitle is the welcome banner's job now.
    assert "agent-aware CLI" not in str(panel.subtitle)


# ---------------------------------------------------------------------------
# which_panel
# ---------------------------------------------------------------------------


def test_which_panel_running_server():
    out = _render(
        which_panel(
            workspace_path="/Users/k/comfyui",
            workspace_type="recent",
            python_executable="/usr/bin/python3",
            python_version="3.12.8",
            server_running=True,
            server_url="http://127.0.0.1:8188",
        )
    )
    assert "/Users/k/comfyui" in out
    assert "recent" in out
    assert "/usr/bin/python3" in out
    assert "3.12.8" in out
    assert "running" in out
    assert "127.0.0.1:8188" in out


def test_which_panel_stopped_server():
    out = _render(
        which_panel(
            workspace_path="/x",
            workspace_type="here",
            python_executable=None,
            python_version=None,
            server_running=False,
            server_url=None,
        )
    )
    assert "/x" in out
    assert "here" in out
    assert "stopped" in out


def test_which_panel_subtitle_carries_brand():
    """When ``version`` is passed, ``which_panel`` emits the canonical
    title/subtitle via ``branded_panel`` — no separate ``cli_footer``
    needed below."""
    panel = which_panel(
        workspace_path="/x",
        workspace_type="recent",
        python_executable="/usr/bin/python",
        python_version="3.12",
        server_running=False,
        server_url="http://127.0.0.1:8188",
        version="0.0.0",
    )
    assert "workspace" in str(panel.title)
    assert "comfy CLI v0.0.0" in str(panel.subtitle)


# ---------------------------------------------------------------------------
# auth panels
# ---------------------------------------------------------------------------


def test_auth_list_table_renders_rows():
    records = [
        {"provider": "comfy-cloud", "key": "sk-t…7890", "updated_at": "2026-05-15T09:48:24+00:00"},
        {"provider": "huggingface", "key": "hf-f…-baz", "updated_at": "2026-05-15T09:48:24+00:00"},
    ]
    out = _render(
        auth_list_table(
            records,
            supported=["comfy-cloud", "civitai", "huggingface"],
            path="/tmp/secrets.json",
        )
    )
    assert "Auth providers" in out
    assert "2 configured" in out
    assert "comfy-cloud" in out
    assert "huggingface" in out
    assert "sk-t…7890" in out
    assert "supported:" in out
    assert "/tmp/secrets.json" in out


def test_auth_empty_panel_shows_supported_and_hint():
    out = _render(auth_empty_panel(supported=["comfy-cloud", "civitai"], path="/tmp/x.json"))
    assert "No providers configured" in out
    assert "comfy auth set" in out
    assert "Supported:" in out
    assert "comfy-cloud · civitai" in out
    assert "/tmp/x.json" in out
