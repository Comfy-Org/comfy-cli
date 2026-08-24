"""Every `comfy build` command's three registration sites must agree.

Registration is atomic per command: the Typer path, the ``COMMAND_SCHEMAS`` key,
and the shipped schema file (its name, its ``$id`` and its ``title``). The
existing discovery ratchets check the emit literal against the tree and the map,
which leaves two holes a rename falls straight through — a command that emits
nothing at all, and a schema file left on disk after its key moved. Both were
real: the ``release`` rename left five ``build_version_*.json`` orphans behind.
"""

from __future__ import annotations

import json
from pathlib import Path

from build_tree_support import leaf_commands

SCHEMAS_DIR = Path(__file__).resolve().parents[3] / "comfy_cli" / "schemas"

_KEY_PREFIX = "comfy build "
_SCHEMA_PREFIX = "build_"


def _registered_build_schemas() -> dict[str, str]:
    from comfy_cli.discovery import COMMAND_SCHEMAS

    return {key: name for key, name in COMMAND_SCHEMAS.items() if key.startswith(_KEY_PREFIX)}


def test_every_build_command_registers_a_schema_file_that_names_it() -> None:
    # Given
    registered = _registered_build_schemas()
    commands = leaf_commands()
    assert commands, "the build tree walk found no commands — this guard would pass vacuously"

    # When
    unregistered = sorted(c for c in commands if f"{_KEY_PREFIX}{c}" not in registered)

    # Then
    assert not unregistered, f"build commands with no COMMAND_SCHEMAS key: {unregistered}"
    for command in sorted(commands):
        key = f"{_KEY_PREFIX}{command}"
        name = registered[key]
        schema = json.loads((SCHEMAS_DIR / f"{name}.json").read_text(encoding="utf-8"))
        assert schema["$id"] == f"https://comfy.org/schemas/{name}.json"
        assert schema["title"] == f"{key} --json data payload"


def test_no_build_schema_file_is_orphaned() -> None:
    """A schema whose key moved keeps validating envelopes nobody emits."""
    # Given
    shipped = {path.stem for path in SCHEMAS_DIR.glob(f"{_SCHEMA_PREFIX}*.json")}

    # When
    registered = set(_registered_build_schemas().values())

    # Then
    assert shipped == registered, (
        f"orphaned schema files: {sorted(shipped - registered)}; "
        f"registered names with no file: {sorted(registered - shipped)}"
    )
