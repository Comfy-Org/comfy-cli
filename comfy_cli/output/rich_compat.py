"""Drop-in replacements for ``rich.print``.

Migration pattern:
    # before
    from rich import print as rprint
    # after
    from comfy_cli.output import rprint

Behavior:
- Pretty mode: byte-identical to ``rich.print``.
- JSON / NDJSON mode: routed to stderr so stdout is reserved for the envelope
  / NDJSON events. The Rich markup is preserved on stderr, so humans tailing
  ``stderr`` (or developers debugging) still see formatted output.

Existing code that uses ``console.print(table)`` for Rich tables should
migrate to ``renderer.console().print(table)`` so the same redirection
applies. That's not required for pretty-mode correctness — those calls
already work — only for clean stdout in JSON mode.
"""

from __future__ import annotations

from typing import Any


def rprint(*args: Any, **kwargs: Any) -> None:
    """Print via the current renderer.

    The renderer is a process-wide singleton; in pretty mode this matches
    ``rich.print`` exactly. In JSON modes it goes to stderr.
    """
    # Imported lazily so we don't take a hard dependency at import time.
    from comfy_cli.output.renderer import get_renderer

    get_renderer().print(*args, **kwargs)


def console_print(*args: Any, **kwargs: Any) -> None:
    """Like rprint but via a Rich Console object (for tables/panels)."""
    from comfy_cli.output.renderer import get_renderer

    get_renderer().console().print(*args, **kwargs)
