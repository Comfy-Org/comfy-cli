"""What every long wait in the CLI words the same way.

Two commands now narrate a transfer that takes minutes — `build push` sending a
model up, and `deploy up` watching one come down onto a deployment's disk — and
a third will follow. They differ in where the numbers come from: `build push`
counts bytes it is writing itself, while a deployment's numbers are computed by
the deploy service so that the portal, the list and the CLI cannot disagree.
What they must NOT differ in is how a size, a duration or a surface is chosen,
because a person meets both in the same terminal minutes apart.

Kept in `output/` rather than beside either command: it belongs to neither, and
putting it in one would make the other import a module named for a command it
has nothing to do with.
"""

from __future__ import annotations

from typing import Any, Literal

# Which of the three shapes a reporter draws. `live` is one redrawn line for a
# person at a terminal, `lines` is a plain line per sample for a person whose
# output is captured (no carriage returns to litter a log), `events` is the
# machine stream an agent reads.
Surface = Literal["events", "live", "lines"]


def human_bytes(n: float) -> str:
    """A size a person reads, counting in 1024s.

    1024 rather than 1000 because the portal's own `formatBytes` does, and the
    same deployment is read on both: one number quoted two ways is a support
    question. It costs the larger-looking figure a storage vendor would print.
    """
    size = float(n)
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024:
            return f"{size:.0f} {unit}" if unit == "B" else f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} TB"


def human_seconds(seconds: float) -> str:
    """A duration a person reads: `45s`, `3m 20s`, `1h 05m`.

    Rounded rather than truncated, so a time left never shows `0s` for the last
    whole second of a wait.
    """
    whole = int(seconds + 0.5)
    if whole < 60:
        return f"{whole}s"
    if whole < 3600:
        return f"{whole // 60}m {whole % 60:02d}s"
    hours, rest = divmod(whole, 3600)
    return f"{hours}h {rest // 60:02d}m"


def surface_for(renderer: Any) -> Surface:
    """The shape this renderer's audience can actually read.

    The order is the whole rule: a machine mode beats everything, because an
    agent reading captured output must never be handed cursor moves; a real
    terminal gets the redrawn line; anything else gets plain lines. Both
    reporters ask this, so the two waits cannot end up on different surfaces in
    the same terminal.
    """
    if not renderer.is_pretty():
        return "events"
    return "live" if renderer.console().is_terminal else "lines"
