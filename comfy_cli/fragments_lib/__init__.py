"""Bundled fragment library — ships with the CLI.

Resolution order for a bare fragment name is: explicit path → local lib
(``--lib`` / ``./fragments``) → this bundled library. Local names shadow
bundled ones so users can override.
"""

from importlib import resources
from pathlib import Path


def bundled_fragments_dir() -> Path | None:
    """Filesystem path of the bundled fragment library, or None if unavailable."""
    try:
        root = resources.files("comfy_cli.fragments_lib")
        path = Path(str(root))
        return path if path.is_dir() else None
    except (ModuleNotFoundError, TypeError):
        return None
