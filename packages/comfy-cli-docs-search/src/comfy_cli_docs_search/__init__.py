"""Version-matched, optional LanceDB documentation search assets."""

from __future__ import annotations

import json
from importlib import resources
from typing import Any

PACK_SCHEMA_VERSION = 1
PACK_FILENAME = "docs-search-pack.tar.gz"


def pack_resource():
    """Return the bundled pack resource, whether this wheel is complete yet or not."""
    return resources.files(__name__) / "assets" / PACK_FILENAME


def read_pack_manifest() -> dict[str, Any]:
    """Read the version metadata shipped beside the archive."""
    return json.loads((resources.files(__name__) / "assets" / "manifest.json").read_text(encoding="utf-8"))
