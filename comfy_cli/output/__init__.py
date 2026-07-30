"""Output rendering for comfy-cli.

Two audiences, one CLI: humans see Rich-formatted output; agents see JSON envelopes.
The same code path produces both. The mode is decided once at command entry by
`Renderer.resolve(...)` and stored as a process-wide singleton via `set_renderer`.

Migration policy (Phase 1):
- A ``from rich import print as rprint`` whose output is a *log or hint* becomes
  ``from comfy_cli.output import rprint``. The shim suppresses stdout in JSON
  mode (redirecting to stderr so logs are still visible) and is byte-identical
  to rprint in pretty mode.
- A call site that prints a command's *primary result* must NOT be migrated
  until that command emits a ``renderer.emit(...)`` envelope. ``resolve()``
  selects JSON mode whenever stdout is not a TTY, so migrating a result path
  first would route the result to stderr and leave stdout empty -- e.g.
  ``comfy generate ... > out.txt`` would write an empty file. The known
  un-migrated result paths are ``command/generate/app.py``,
  ``command/generate/output.py`` and ``command/pr_command.py``; each carries an
  in-file comment saying so.
- Commands that produce a structured result emit a final envelope with
  ``renderer.emit(data)``. This is the *only* thing on stdout in JSON mode.
- Streaming commands emit ``renderer.event(type, **fields)`` events; one
  NDJSON object per line.
"""

from comfy_cli.output.renderer import (
    OutputMode,
    Renderer,
    get_renderer,
    set_renderer,
)
from comfy_cli.output.rich_compat import (
    console_print,
    rprint,
)

__all__ = [
    "OutputMode",
    "Renderer",
    "console_print",
    "get_renderer",
    "rprint",
    "set_renderer",
]
