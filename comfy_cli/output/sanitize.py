r"""Strip ANSI escape sequences and control characters from untrusted text.

Server-supplied strings reach the terminal verbatim in pretty mode: a remote or
shared ComfyUI instance (``comfy run --host/--port``) picks the ``prompt_id``,
and an exception raised against that host carries whatever text the host chose.
Rich does not save us here — ``rich.text.Text`` drops a few C0 characters but
passes ``\x1b`` straight through, so a CSI/OSC sequence reaches the user's
terminal and can clear the screen, rewrite the window title, or repaint earlier
lines to spoof CLI output.

This module sanitizes **pretty** rendering only. The JSON/NDJSON envelope paths
must NOT use it: stripping there would silently mutate the data agents parse,
and the consumer decodes JSON rather than acting on control sequences. Note the
envelope is not escape-*free*, though: the renderer dumps with
``ensure_ascii=False``, which escapes C0 (so ``\x1b`` leaves as a six-character
``\u`` escape) but emits U+0080-U+009F literally — the 8-bit C1 introducers ``\x9b``/``\x90``/
``\x9d`` cross it raw. Anything that later renders envelope text to a human
terminal owes it a trip through ``sanitize_terminal_stream``.

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
``sanitize_terminal_stream`` and ``sanitize_log_markup`` are for *replaying a
captured stream* (``comfy logs``, the launch-failure panel): the bytes were
produced by another program, so they are line-bounded rather than allowed to
swallow the tail (see ``_STREAM_*`` below), and the stream variant additionally
keeps SGR and carriage return because that program owns its own colouring. Use
the narrower ``sanitize`` whenever the sink is a message the CLI composed
rather than a stream it recorded. None of them belong on a JSON path.

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
#
# ``{payload}``/``{end}`` are filled in below to give the OSC/DCS/SOS/PM/APC
# ("string") families their two reach variants — see ``_UNTERMINATED_*``. The
# template is fed through ``str.format``, so any regex brace quantifier added
# here (``{2}``, ``{1,3}``) must be doubled.
_ESCAPE_TEMPLATE = r"""
      \x1b\[ [\x30-\x3f]* [\x20-\x2f]* [\x40-\x7e]?      # CSI, 7-bit (ESC [ ...)
    | \x9b   [\x30-\x3f]* [\x20-\x2f]* [\x40-\x7e]?      # CSI, 8-bit C1 form
    | \x1b[\]PX^_] {payload} (?: \x1b\\ | \x9c | \x07 {end} )   # OSC/DCS/SOS/PM/APC, 7-bit
    | [\x90\x98\x9d\x9e\x9f] {payload} (?: \x1b\\ | \x9c | \x07 {end} )  # same, 8-bit C1 form
    | \x1b [\x20-\x2f]* [\x30-\x7e]                      # two-/three-char escapes (ESC ( B, ESC =, ...)
"""

# Authored messages: an unterminated introducer runs to the end of the string,
# which is what the terminal itself would do with it (see ``sanitize``).
_UNTERMINATED_TO_END = {"payload": r".*?", "end": r"| \Z"}
# Replayed streams: an unterminated introducer stops at the next newline. A
# terminal would swallow forward here too, but the bytes are a *recording* of
# another process — a crash that dumps binary into the log would otherwise
# delete every remaining line of `comfy logs` with no truncation indicator,
# while the JSON path still showed them. The introducer and its payload-so-far
# are still removed, so what survives is inert text either way.
_UNTERMINATED_TO_NEWLINE = {"payload": r"[^\n]*?", "end": r"| (?=\n) | \Z"}

_ESCAPE_SEQUENCE_RE = re.compile(_ESCAPE_TEMPLATE.format(**_UNTERMINATED_TO_END), re.VERBOSE | re.DOTALL)

# Whatever survives the sweep above: stray C0 (minus tab/newline), DEL, and the
# C1 block. A lone trailing ESC and an orphaned 8-bit introducer land here.
_CONTROL_CHAR_RE = re.compile(r"[\x00-\x08\x0b-\x1f\x7f-\x9f]")

# 7-bit SGR only, parameter bytes restricted to digits/;/: so private-mode or
# intermediate-byte oddities ending in 'm' are still stripped. 8-bit C1 CSI
# (\x9b...m) is NOT kept — UTF-8 terminals don't honor it and keeping it buys
# nothing. Used as an exact-match test for a kept sequence, and to find the
# SGRs a replayed line left open (``close_open_sgr``).
_SGR_RE = re.compile(r"\x1b\[[0-9;:]*m")
# ...of which these are the ones that close everything again: no parameters, or
# parameters that are all zero.
_SGR_RESET_RE = re.compile(r"\x1b\[[0;:]*m")

# The escape alternation plus the residual control characters, in ONE pattern
# so a single ``.sub()`` pass sees whole escape sequences before the
# lone-control-char fallback can split them.
#
# The keep-SGR variant also keeps CR: tab+newline are layout, and \r only
# rewrites the line it is on — which the replaying stream fully owns — while
# being how tqdm progress lines are genuinely recorded. The strip-all variant
# drops CR, matching ``sanitize``, because its sink is a Rich Panel whose lines
# the CLI (not the log) lays out.
_TERMINAL_STREAM_RE = re.compile(
    _ESCAPE_TEMPLATE.format(**_UNTERMINATED_TO_NEWLINE) + r"| [\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]",
    re.VERBOSE | re.DOTALL,
)
_LOG_CAPTURE_RE = re.compile(
    _ESCAPE_TEMPLATE.format(**_UNTERMINATED_TO_NEWLINE) + r"| [\x00-\x08\x0b-\x1f\x7f-\x9f]",
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

    An unterminated OSC/DCS introducer is bounded at the next newline rather
    than swallowing the tail the way ``sanitize`` does — the difference between
    dropping the rest of a one-line message and dropping the rest of the log.

    This cannot be layered on top of ``sanitize``: that function's second pass
    sweeps residual control characters, which would eat the ``\\x1b`` of every
    SGR the first pass deliberately kept. Hence the single combined pass here.

    Preserving SGR leaves the *closing* of those styles to the caller — see
    ``close_open_sgr``, which a TTY sink should apply to the result.

    The caller must still pick a sink that does not parse Rich markup — this
    returns log text verbatim, brackets included. For a markup-interpreting
    sink use ``sanitize_log_markup`` instead and accept the loss of colour.
    """
    return _TERMINAL_STREAM_RE.sub(lambda m: m.group(0) if _SGR_RE.fullmatch(m.group(0)) else "", text)


def sanitize_log_markup(value: Any) -> str:
    """``sanitize_markup`` for captured log text, without the tail-swallow.

    The launch-failure Panel parses its content as Rich markup, so captured log
    text needs the markup escape (an unbalanced ``[/red]`` in the log otherwise
    raises ``MarkupError`` from inside the failure handler). But the text is a
    recording, not a message we composed, so it gets the stream sanitizers'
    line-bounded reach: with plain ``sanitize_markup`` one stray ``\\x1b]`` byte
    would truncate the panel there and discard exactly the traceback the panel
    exists to display.

    Colour is dropped rather than preserved as in ``sanitize_terminal_stream``:
    the sink re-lays-out and re-styles what it is given, so raw SGR bytes would
    fight the panel's own rendering. Monochrome is fine in an error panel.
    """
    text = value if isinstance(value, str) else str(value)
    return _escape_markup(_LOG_CAPTURE_RE.sub("", text))


def close_open_sgr(text: str) -> str:
    """Append an SGR reset to every line of ``text`` that ends mid-style.

    ``sanitize_terminal_stream`` keeps the log's own colouring, which means the
    log also decides when it stops. A line that never resets — or that opens
    ``\\x1b[8m`` (conceal) or a foreground-equals-background pair — otherwise
    bleeds into everything printed after it: the rest of the replay, and then
    whatever the user types at the prompt next. Since anyone who can land text
    in the log can land an SGR in it, that is a way to hide later log lines
    from a human while the JSON path still shows them.

    Closing per line rather than once at the end is what contains it. The cost
    is fidelity for a log that opens a style on one line and expects it to hold
    across the next — rare, because loggers that colour (rich, colorama,
    ``tqdm``, uvicorn) reset before each newline already.

    Lines already ending in a reset gain nothing, and text with no SGR at all
    is returned unchanged — so an uncoloured log stays byte-identical.
    """
    if "\x1b[" not in text:
        return text
    # Split on "\n" only: ``str.splitlines`` would also break on "\r", which is
    # a within-line repaint here (tqdm), and on the Unicode separators.
    lines = text.split("\n")
    for i, line in enumerate(lines):
        last = None
        for match in _SGR_RE.finditer(line):
            last = match.group(0)
        if last is not None and not _SGR_RESET_RE.fullmatch(last):
            lines[i] = line + "\x1b[0m"
    return "\n".join(lines)


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
