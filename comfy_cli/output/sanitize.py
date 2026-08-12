r"""Strip ANSI escape sequences and control characters from untrusted text.

Server-supplied strings reach the terminal verbatim in pretty mode: a remote or
shared ComfyUI instance (``comfy run --host/--port``) picks the ``prompt_id``,
and an exception raised against that host carries whatever text the host chose.
Rich does not save us here — ``rich.text.Text`` drops a few C0 characters but
passes ``\x1b`` straight through, so a CSI/OSC sequence reaches the user's
terminal and can clear the screen, rewrite the window title, or repaint earlier
lines to spoof CLI output.

This module sanitizes **pretty** rendering only. The JSON/NDJSON envelope paths
must NOT use it: ``json.dumps`` already escapes ``\x1b`` as a ``\u`` escape, and
stripping there would silently mutate the data agents parse.

``Renderer`` is not the only path to the terminal, so applying it there is not
sufficient on its own. Command modules also call ``rich.print`` directly under
their own ``is_pretty()`` gates, and those never route through ``Renderer``:
they must call ``sanitize_markup`` on each server-supplied value themselves,
as the ``comfy run`` and ``comfy models`` paths do. Automatic coverage stops at
``Renderer``; treat a bare ``rich.print`` of remote text as unsanitized until
you have checked it.

Scope: C0/C1 control characters and ANSI escape sequences (CSI, OSC, DCS, and
the shorter two-character forms). Tab and newline are preserved — they are
legitimate layout in multi-line messages. Carriage return is removed, since it
lets a message overwrite a line that has already been printed. Non-control
Unicode is left alone; homoglyph and bidi-override spoofing are a different
problem and deliberately out of scope here.

Two display contracts, one module. ``sanitize`` above is for text the CLI
*authors* — a message we compose around an untrusted value, where colour is
ours to add and nothing in the value needs to survive as styling.
``sanitize_terminal_stream`` is for *replaying a captured stream* back to the
terminal (``comfy logs``): the bytes were produced by another program that owns
its own colouring, so SGR and carriage return are kept and only the sequences a
terminal would *act* on are dropped. Use the narrower one whenever the sink is a
message rather than a replayed stream. Neither belongs on a JSON path.

Removing the escape *bytes* is only half the boundary: Rich manufactures new
ones from markup it finds in a string, so text bound for a markup-interpreting
sink also needs ``sanitize_markup`` (see its docstring).
"""

from __future__ import annotations

import re
from typing import Any

from rich.markup import escape as _escape_markup

# One pass over the ANSI escape forms a terminal will act on. Ordered so the
# multi-character sequences are consumed before the lone-ESC fallback.
_ESCAPE_SEQUENCE_RE = re.compile(
    r"""
      \x1b\[ [\x30-\x3f]* [\x20-\x2f]* [\x40-\x7e]?      # CSI, 7-bit (ESC [ ...)
    | \x9b   [\x30-\x3f]* [\x20-\x2f]* [\x40-\x7e]?      # CSI, 8-bit C1 form
    | \x1b[\]PX^_] .*? (?: \x1b\\ | \x9c | \x07 | \Z )   # OSC/DCS/SOS/PM/APC, 7-bit
    | [\x90\x98\x9d\x9e\x9f] .*? (?: \x1b\\ | \x9c | \x07 | \Z )  # same, 8-bit C1 form (DCS/SOS/OSC/PM/APC)
    | \x1b [\x20-\x2f]* [\x30-\x7e]                      # two-/three-char escapes (ESC ( B, ESC =, ...)
    """,
    re.VERBOSE | re.DOTALL,
)

# Whatever survives the sweep above: stray C0 (minus tab/newline), DEL, and the
# C1 block. A lone trailing ESC and an orphaned 8-bit introducer land here.
_CONTROL_CHAR_RE = re.compile(r"[\x00-\x08\x0b-\x1f\x7f-\x9f]")

# Exact-match test for a kept sequence: 7-bit SGR only, parameter bytes
# restricted to digits/;/: so private-mode or intermediate-byte oddities
# ending in 'm' are still stripped. 8-bit C1 CSI (\x9b...m) is NOT kept —
# UTF-8 terminals don't honor it and keeping it buys nothing.
_SGR_RE = re.compile(r"\x1b\[[0-9;:]*m\Z")

# ``_ESCAPE_SEQUENCE_RE``'s alternation plus the residual control characters,
# in one pattern so a single ``.sub()`` pass sees whole escape sequences before
# the lone-control-char fallback can split them. Keeps tab/newline/CR:
# tab+newline are layout, and \r only rewrites the line it is on — which its
# writer already fully controls — while being how tqdm progress lines are
# genuinely recorded.
# The leading newline is load-bearing under ``re.VERBOSE``: the borrowed pattern
# ends in a ``#`` comment, and appending to the same line would make the whole
# alternative part of that comment — a regex that still compiles and still
# strips escape sequences, but silently stops stripping lone control bytes.
_TERMINAL_STREAM_RE = re.compile(
    _ESCAPE_SEQUENCE_RE.pattern + "\n" + r"| [\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]",
    re.VERBOSE | re.DOTALL,
)


def sanitize(text: str) -> str:
    """Return ``text`` with ANSI escape sequences and control characters removed.

    An unterminated OSC/DCS sequence consumes the remainder of the string —
    that is what the terminal itself would do with it, so dropping the tail is
    the conservative reading, not data loss we could have avoided.
    """
    return _CONTROL_CHAR_RE.sub("", _ESCAPE_SEQUENCE_RE.sub("", text))


def sanitize_terminal_stream(text: str) -> str:
    """``sanitize`` for a captured stream replayed to the terminal, keeping colour.

    Same threat model as ``sanitize`` — CSI cursor/erase sequences, OSC title
    rewrites, DCS payloads and stray control bytes are all removed — with two
    deliberate exceptions, because the text being replayed is another program's
    own output rather than a message we composed:

    * **7-bit SGR is preserved** (``\\x1b[31m``), so a ComfyUI log that coloured
      its own warnings still reads the way it did on the launching terminal.
      SGR only restyles subsequent characters; it cannot move the cursor, erase
      anything, or address the terminal outside the stream.
    * **Carriage return is preserved**, so a ``tqdm`` progress line recorded in
      the log renders as one rewritten line instead of a wall of duplicates.
      ``\\r`` can only repaint the line it is already on, which the replayed
      stream owns in full — unlike in ``sanitize``, where a server string would
      be overwriting text the CLI itself printed.

    This cannot be layered on top of ``sanitize``: that function's second pass
    sweeps residual control characters, which would eat the ``\\x1b`` of every
    SGR the first pass deliberately kept. Hence the single combined pass here.

    The caller must still pick a sink that does not parse Rich markup — this
    returns log text verbatim, brackets included. For a markup-interpreting
    sink use ``sanitize_markup`` instead and accept the loss of colour.
    """
    return _TERMINAL_STREAM_RE.sub(lambda m: m.group(0) if _SGR_RE.fullmatch(m.group(0)) else "", text)


def sanitize_optional(text: str | None) -> str | None:
    """``sanitize`` for the many call sites whose value is an optional string.

    Non-``None`` values go through ``sanitize_value``, so a caller that ignores
    the ``str`` annotation gets the ``str()`` coercion the old f-strings did
    rather than a ``TypeError`` from ``re.sub``.
    """
    return None if text is None else sanitize_value(text)


def sanitize_value(value: Any) -> str:
    """Stringify ``value`` and sanitize it — for panel fields rendered via ``str()``."""
    return sanitize(value if isinstance(value, str) else str(value))


def sanitize_markup(value: Any) -> str:
    """``sanitize_value`` plus Rich-markup escaping, for markup-interpreting sinks.

    Stripping the escape bytes is not enough on its own: Rich re-creates them
    from markup found in the string. A server-supplied ``[link=http://evil]x[/link]``
    renders as a live OSC 8 hyperlink, ``[red]`` restyles the line to spoof
    other output, and an unbalanced ``[/]`` raises ``rich.errors.MarkupError``
    — crashing the CLI while it is merely printing an info line.

    Use this for anything interpolated into a markup string or handed to
    ``Table.add_row``. Do NOT use it for a value passed to ``rich.text.Text``:
    ``Text`` never parses markup, so the escaping backslashes would show up
    verbatim on screen.
    """
    return _escape_markup(sanitize_value(value))
