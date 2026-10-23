"""CLI-surface guard: command-inventory snapshot.

Mirrors the MCP server's ``EXPECTED_TOOLS`` snapshot pattern for the CLI:
:func:`test_command_inventory_snapshot` snapshots the full command tree so any
surface addition/removal/hide shows up as an explicit diff in review.

Regenerate the snapshot after an intentional surface change with::

    UPDATE_CLI_SNAPSHOT=1 pytest tests/comfy_cli/test_cli_surface.py

Help/hint strings that name nonexistent commands are linted separately by
``tests/comfy_cli/test_command_mentions.py``.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from comfy_cli.cmdline import app
from comfy_cli.help_json import build_help_json

SNAPSHOT_PATH = Path(__file__).resolve().parent / "fixtures" / "cli_command_inventory.txt"


def _command_inventory() -> list[str]:
    """Every command path (groups + leaves), sorted, with a ``[hidden]`` marker."""
    doc = build_help_json(app)
    out: list[str] = []

    def walk(commands: dict, path: list[str]) -> None:
        for name, entry in sorted(commands.items()):
            fqp = " ".join(path + [name])
            out.append(f"{fqp}\t[hidden]" if entry.get("hidden") else fqp)
            subs = entry.get("subcommands")
            if subs:
                walk(subs, path + [name])

    walk(doc["commands"], ["comfy"])
    return out


def test_command_inventory_snapshot():
    """The registered command surface matches the committed snapshot."""
    current = _command_inventory()
    rendered = "\n".join(current) + "\n"

    if os.environ.get("UPDATE_CLI_SNAPSHOT", "").strip().lower() in {"1", "true", "yes"}:
        SNAPSHOT_PATH.write_text(rendered, encoding="utf-8")
        pytest.skip(f"snapshot regenerated: {SNAPSHOT_PATH}")

    assert SNAPSHOT_PATH.exists(), (
        f"missing snapshot {SNAPSHOT_PATH}; regenerate with "
        "`UPDATE_CLI_SNAPSHOT=1 pytest tests/comfy_cli/test_cli_surface.py`"
    )
    expected = SNAPSHOT_PATH.read_text(encoding="utf-8").splitlines()
    if current != expected:
        added = sorted(set(current) - set(expected))
        removed = sorted(set(expected) - set(current))
        detail = ""
        if added:
            detail += "\n  added:\n" + "\n".join(f"    + {p}" for p in added)
        if removed:
            detail += "\n  removed:\n" + "\n".join(f"    - {p}" for p in removed)
        pytest.fail(
            "CLI command inventory changed vs. the committed snapshot."
            f"{detail}\n\n"
            "If intentional, regenerate with "
            "`UPDATE_CLI_SNAPSHOT=1 pytest tests/comfy_cli/test_cli_surface.py` "
            "and review the diff."
        )
