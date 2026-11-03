"""Saved userdata UUID contract shared by save receipts and link publication."""

from __future__ import annotations

import re

# Browser consumer: uuid11.1.1 validate(), plus lowercase canonical wire form.
# Accept RFC9562 versions1..8 and variants8..b, plus its explicit NIL/MAX cases.
_SAVED_FILE_UUID = re.compile(
    r"(?:[0-9a-f]{8}-[0-9a-f]{4}-[1-8][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}"
    r"|00000000-0000-0000-0000-000000000000|ffffffff-ffff-ffff-ffff-ffffffffffff)"
)


def canonical_saved_file_id(value: object) -> str | None:
    """Return a received canonical ID, without deriving or coercing one."""
    return value if isinstance(value, str) and _SAVED_FILE_UUID.fullmatch(value) else None
