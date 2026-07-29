"""Shared read-only helpers for scanning installed custom node packs.

Extracted from :mod:`comfy_cli.command.outdated` so every read-only pack
report (``comfy outdated``, ``comfy node deps``, …) agrees on *what counts as
an installed pack* and on how a pack ``pyproject.toml`` is parsed. Nothing
here mutates the workspace.
"""

from __future__ import annotations

import contextlib
import sys
from pathlib import Path

from comfy_cli.registry import extract_node_configuration


def iter_pack_dirs(custom_nodes_dir: Path) -> list[Path]:
    """Return the pack directories under *custom_nodes_dir*, sorted by name.

    Dotfiles and ``__pycache__`` are skipped; ``.disabled`` packs are kept
    (they are still installed, just not loaded).
    """
    if not custom_nodes_dir.is_dir():
        return []
    packs = []
    for entry in sorted(custom_nodes_dir.iterdir()):
        if not entry.is_dir():
            continue
        if entry.name.startswith(".") or entry.name == "__pycache__":
            continue
        packs.append(entry)
    return packs


def read_pyproject(path: str):
    """Parse a pack/core ``pyproject.toml`` via the shared registry parser.

    ``extract_node_configuration`` emits its own validation warnings through
    ``typer.echo``/rich to *stdout*; in JSON mode that would corrupt the single
    envelope on stdout. Route those side-messages to stderr where they belong.
    """
    with contextlib.redirect_stdout(sys.stderr):
        return extract_node_configuration(path)
