"""Output rendering for comfy-cli.

Two audiences, one CLI: humans see Rich-formatted output; agents see JSON envelopes.
The same code path produces both. The mode is decided once at command entry by
`Renderer.resolve(...)` and stored as a process-wide singleton via `set_renderer`.

Migration policy (Phase 1):
- Every existing ``from rich import print as rprint`` becomes
  ``from comfy_cli.output import rprint``. The shim suppresses stdout in JSON
  mode (redirecting to stderr so logs are still visible) and is byte-identical
  to rprint in pretty mode.
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
