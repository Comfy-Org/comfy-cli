r"""Strip ANSI escape sequences and control characters from untrusted text.

Server-supplied strings reach the terminal verbatim in pretty mode: a remote or
shared ComfyUI instance (``comfy run --host/--port``) picks the ``prompt_id``,
and an exception raised against that host carries whatever text the host chose.
Rich does not save us here — ``rich.text.Text`` drops a few C0 characters but
passes ``\x1b`` straight through, so a CSI/OSC sequence reaches the user's
terminal and can clear the screen, rewrite the window title, or repaint earlier
lines to spoof CLI output.

This module is the single sanitizing boundary for **pretty** rendering only.
The JSON/NDJSON envelope paths must NOT use it: ``json.dumps`` already escapes
``\x1b`` as a ``\u`` escape, and stripping there would silently mutate
the data agents parse.

Scope: C0/C1 control characters and ANSI escape sequences (CSI, OSC, DCS, and
the shorter two-character forms). Tab and newline are preserved — they are
legitimate layout in multi-line messages. Carriage return is removed, since it
lets a message overwrite a line that has already been printed. Non-control
Unicode is left alone; homoglyph and bidi-override spoofing are a different
problem and deliberately out of scope here.
"""

from __future__ import annotations

import re
from typing import Any

# One pass over the ANSI escape forms a terminal will act on. Ordered so the
# multi-character sequences are consumed before the lone-ESC fallback.
_ESCAPE_SEQUENCE_RE = re.compile(
    r"""
      \x1b\[ [\x30-\x3f]* [\x20-\x2f]* [\x40-\x7e]?      # CSI, 7-bit (ESC [ ...)
    | \x9b   [\x30-\x3f]* [\x20-\x2f]* [\x40-\x7e]?      # CSI, 8-bit C1 form
    | \x1b[\]PX^_] .*? (?: \x1b\\ | \x9c | \x07 | \Z )   # OSC/DCS/SOS/PM/APC, 7-bit
    | [\x90\x9d\x9e\x9f] .*? (?: \x1b\\ | \x9c | \x07 | \Z )  # same, 8-bit C1 form
    | \x1b [\x20-\x2f]* [\x30-\x7e]                      # two-/three-char escapes (ESC ( B, ESC =, ...)
    """,
    re.VERBOSE | re.DOTALL,
)

# Whatever survives the sweep above: stray C0 (minus tab/newline), DEL, and the
# C1 block. A lone trailing ESC and an orphaned 8-bit introducer land here.
_CONTROL_CHAR_RE = re.compile(r"[\x00-\x08\x0b-\x1f\x7f-\x9f]")


def sanitize(text: str) -> str:
    """Return ``text`` with ANSI escape sequences and control characters removed.

    An unterminated OSC/DCS sequence consumes the remainder of the string —
    that is what the terminal itself would do with it, so dropping the tail is
    the conservative reading, not data loss we could have avoided.
    """
    return _CONTROL_CHAR_RE.sub("", _ESCAPE_SEQUENCE_RE.sub("", text))


def sanitize_optional(text: str | None) -> str | None:
    """``sanitize`` for the many call sites whose value is an optional string."""
    return None if text is None else sanitize(text)


def sanitize_value(value: Any) -> str:
    """Stringify ``value`` and sanitize it — for panel fields rendered via ``str()``."""
    return sanitize(value if isinstance(value, str) else str(value))
