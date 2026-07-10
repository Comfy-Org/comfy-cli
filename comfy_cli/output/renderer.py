"""Renderer: decides output mode and emits JSON envelopes or pretty output.

UX contract:
- Pretty mode produces output byte-identical to the pre-Phase-1 CLI.
- JSON mode produces a single envelope on stdout (intermediate messages → stderr).
- NDJSON mode produces one JSON event per line on stdout; the final envelope is
  the last line.
- Errors carry a stable ``code`` and a ``hint``. The hint is rendered as a
  yellow line under the error in pretty mode and as ``error.hint`` in JSON.
"""

from __future__ import annotations

import json
import os
import sys
import time
from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, TextIO

from rich.console import Console

from comfy_cli.caller import Caller, detect_caller

# Machine-output contract versions, surfaced in every envelope/event line and
# in `comfy discover` (output_contract). Bump rule: additive optional fields =
# no bump; rename/remove/retype a field or changed exit semantics = bump.
ENVELOPE_SCHEMA = "envelope/1"
EVENT_SCHEMA = "event/1"


class OutputMode(str, Enum):
    PRETTY = "pretty"
    JSON = "json"
    NDJSON = "ndjson"


def _truthy(value: str | None) -> bool:
    if value is None:
        return False
    return value.strip().lower() in {"1", "true", "yes", "on"}


@dataclass
class Renderer:
    mode: OutputMode = OutputMode.PRETTY
    caller: Caller = field(default_factory=detect_caller)
    command: str = ""
    version: str = ""
    where: str | None = None
    # Stream overrides. When None we look up sys.stdout / sys.stderr at the
    # *call site* so test fixtures (capsys, CliRunner) that monkey-patch
    # streams still capture our output. Tests can set these explicitly to
    # pin redirection in scope.
    _pretty_stream_override: TextIO | None = field(default=None, repr=False)
    _machine_stream_override: TextIO | None = field(default=None, repr=False)
    # Throttle map for high-frequency events: {token: last_emit_ts}.
    _throttle: dict[str, float] = field(default_factory=dict, repr=False)
    _envelope_emitted: bool = field(default=False, repr=False)
    _exit_code: int = field(default=0, repr=False)

    @property
    def pretty_stream(self) -> TextIO:
        if self._pretty_stream_override is not None:
            return self._pretty_stream_override
        # In pretty mode we use stdout. In JSON modes we redirect to stderr so
        # stdout is reserved for the envelope/events.
        return sys.stdout if self.mode is OutputMode.PRETTY else sys.stderr

    @pretty_stream.setter
    def pretty_stream(self, value: TextIO) -> None:
        self._pretty_stream_override = value

    @property
    def machine_stream(self) -> TextIO:
        if self._machine_stream_override is not None:
            return self._machine_stream_override
        return sys.stdout

    @machine_stream.setter
    def machine_stream(self, value: TextIO) -> None:
        self._machine_stream_override = value

    @classmethod
    def resolve(
        cls,
        *,
        json_flag: bool | None = None,
        json_stream_flag: bool | None = None,
        no_json_flag: bool = False,
        env: Mapping[str, str] | None = None,
        is_stdout_tty: bool | None = None,
        caller: Caller | None = None,
        command: str = "",
        version: str = "",
    ) -> Renderer:
        """Decide the output mode.

        Precedence (highest first):
            1. --json-stream flag                       → NDJSON
            2. --json flag                              → JSON
            3. --no-json flag                           → PRETTY
            4. COMFY_OUTPUT env (json/ndjson/pretty)    → that mode
            5. agentic caller detected                  → JSON
            6. stdout is not a TTY                      → JSON
            7. default                                  → PRETTY
        """
        env_map = env if env is not None else os.environ
        caller = caller if caller is not None else detect_caller(env_map)
        if is_stdout_tty is None:
            is_stdout_tty = sys.stdout.isatty()

        mode: OutputMode
        if json_stream_flag:
            mode = OutputMode.NDJSON
        elif json_flag:
            mode = OutputMode.JSON
        elif no_json_flag:
            mode = OutputMode.PRETTY
        else:
            env_choice = (env_map.get("COMFY_OUTPUT") or "").strip().lower()
            if env_choice in {"json", "ndjson", "pretty"}:
                mode = OutputMode(env_choice)
            elif caller.agentic:
                mode = OutputMode.JSON
            elif not is_stdout_tty:
                mode = OutputMode.JSON
            else:
                mode = OutputMode.PRETTY

        # Streams are resolved lazily from sys.stdout/sys.stderr at each call
        # site (see pretty_stream / machine_stream properties), so test
        # fixtures that monkey-patch streams keep working.
        return cls(
            mode=mode,
            caller=caller,
            command=command,
            version=version,
        )

    # ----- pretty (Rich) helpers -----

    def pretty_console(self) -> Console:
        # Recreated each call so it respects whatever sys.stdout/stderr looks
        # like *now*. Console is cheap.
        return Console(file=self.pretty_stream, force_terminal=None, soft_wrap=False)

    def stderr_console(self) -> Console:
        return Console(file=sys.stderr)

    def is_pretty(self) -> bool:
        return self.mode is OutputMode.PRETTY

    def is_json(self) -> bool:
        return self.mode in {OutputMode.JSON, OutputMode.NDJSON}

    def is_stream(self) -> bool:
        return self.mode is OutputMode.NDJSON

    def force_stream(self) -> None:
        """Upgrade this renderer to NDJSON streaming mode.

        Used by command-local streaming flags (e.g. ``comfy run --json``)
        after the global callback has already resolved the mode: the
        command knows it produces an event stream, so a plain ``--json``
        on it means "stream NDJSON events + final envelope" rather than
        the single-envelope JSON mode.
        """
        self.mode = OutputMode.NDJSON

    # ----- printing -----

    def print(self, *args: Any, **kwargs: Any) -> None:
        """Print human-readable text. In JSON modes this goes to stderr.

        Use this as a drop-in for ``rich.print``: the Rich markup, panels,
        tables, and exact whitespace are preserved in pretty mode.
        """
        # Rich's print accepts file= kwarg. We always direct human output to
        # `pretty_stream` (stdout in pretty mode, stderr in JSON mode).
        from rich import print as _rprint

        kwargs.setdefault("file", self.pretty_stream)
        _rprint(*args, **kwargs)

    def console(self) -> Console:
        """Return the Console used for Rich tables / panels in pretty mode.

        In JSON modes, returns a Console attached to stderr so any
        ``console.print(table)`` calls don't pollute stdout.
        """
        return self.pretty_console() if self.is_pretty() else self.stderr_console()

    # ----- semantic helpers -----

    def info(self, message: str, *, hint: str | None = None) -> None:
        self.print(message)
        if hint:
            self.print(f"[yellow]Hint:[/yellow] {hint}")

    def warn(self, message: str, *, hint: str | None = None) -> None:
        self.print(f"[yellow]{message}[/yellow]")
        if hint:
            self.print(f"[yellow]Hint:[/yellow] {hint}")

    def success(self, message: str) -> None:
        self.print(f"[bold green]{message}[/bold green]")

    # ----- structured output -----

    def emit(
        self,
        data: Any = None,
        *,
        command: str | None = None,
        where: str | None = None,
        changed: bool | None = None,
        ok: bool = True,
    ) -> None:
        """Emit the final envelope. In pretty mode this is a no-op (data was
        already shown by ``print``/``success``/etc).

        ``ok`` defaults to True for the common success path. Commands that
        carry a structured result *and* a negative verdict (e.g. ``validate``
        on an invalid workflow, which still emits its error/warning payload as
        data) pass ``ok=False`` so the envelope's ``ok`` agrees with the
        process exit code.
        """
        if self.is_pretty():
            return
        if self._envelope_emitted:
            # Defensive: the harness should only emit once. Don't double-write.
            return
        envelope = self._envelope(
            ok=ok,
            command=command or self.command,
            data=data,
            where=where or self.where,
            changed=changed,
            error=None,
        )
        self._write_json_line(envelope)
        self._envelope_emitted = True

    def error(
        self,
        code: str,
        message: str,
        *,
        hint: str | None = None,
        details: Mapping[str, Any] | None = None,
        exit_code: int = 1,
        command: str | None = None,
    ) -> None:
        """Emit a structured error. In pretty mode, prints red message + yellow
        hint. In JSON mode, emits an envelope with ``ok=false`` and the error
        block; in NDJSON mode also emits the envelope as the final line.

        An error message is a navigation signal toward correctness. When the
        call site doesn't supply a usable ``hint`` — ``None`` OR an empty/blank
        string (call sites often pass ``e.hint or ""``) — fall back to the
        code's REGISTERED hint so every error tells the agent what to do next.
        No dead ends.
        """
        self._exit_code = exit_code
        if not (hint and hint.strip()):
            from comfy_cli import error_codes

            registered = error_codes.get(code)
            if registered is not None and registered.hint:
                hint = registered.hint
        if self.is_pretty():
            # Lazy import to keep panel deps optional.
            from comfy_cli.output.panels import error_panel

            self.console().print(error_panel(code=code, message=message, hint=hint, details=details))
            return
        if self._envelope_emitted:
            return
        envelope = self._envelope(
            ok=False,
            command=command or self.command,
            data=None,
            where=self.where,
            changed=None,
            error={
                "code": code,
                "message": message,
                "hint": hint,
                "details": dict(details) if details else None,
            },
        )
        self._write_json_line(envelope)
        self._envelope_emitted = True

    def event(self, type: str, **fields: Any) -> None:
        """Emit one NDJSON event line. Only meaningful in NDJSON mode."""
        if not self.is_stream():
            return
        payload = {"schema": EVENT_SCHEMA, "type": type, **fields}
        self._write_json_line(payload)

    def throttled_event(self, token: str, type: str, *, max_hz: float = 10.0, **fields: Any) -> bool:
        """Emit an NDJSON event only if at least ``1/max_hz`` seconds have
        passed since the last event with the same ``token``. Returns True if
        the event was emitted.
        """
        if not self.is_stream():
            return False
        now = time.monotonic()
        last = self._throttle.get(token, 0.0)
        if now - last < (1.0 / max_hz):
            return False
        self._throttle[token] = now
        self.event(type, **fields)
        return True

    # ----- internals -----

    def _envelope(
        self,
        *,
        ok: bool,
        command: str,
        data: Any,
        where: str | None,
        changed: bool | None,
        error: Mapping[str, Any] | None,
    ) -> dict[str, Any]:
        env: dict[str, Any] = {
            # Contract discriminator + version first: NDJSON consumers pick
            # out the final line by `type` and negotiate shape on `schema`.
            "schema": ENVELOPE_SCHEMA,
            "type": "envelope",
            "ok": ok,
            "command": command,
            "version": self.version,
            "where": where,
            "data": data,
            "error": dict(error) if error else None,
        }
        if changed is not None:
            env["changed"] = changed
        return env

    def _write_json_line(self, payload: Mapping[str, Any]) -> None:
        line = json.dumps(payload, default=_json_default, ensure_ascii=False)
        self.machine_stream.write(line + "\n")
        self.machine_stream.flush()

    @property
    def exit_code(self) -> int:
        return self._exit_code


def _json_default(obj: Any) -> Any:
    # Best-effort JSON coercion for common non-serializable types.
    from pathlib import Path

    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, Enum):
        return obj.value
    if hasattr(obj, "isoformat"):
        try:
            return obj.isoformat()
        except Exception:  # noqa: BLE001
            pass
    return str(obj)


# ----- process-wide singleton ----------------------------------------------

_RENDERER: Renderer | None = None


def set_renderer(renderer: Renderer) -> None:
    """Install the process-wide renderer. Called once from cmdline.entry()."""
    global _RENDERER
    _RENDERER = renderer


def get_renderer() -> Renderer:
    """Return the installed renderer, creating a default pretty one if none.

    The default is pretty so existing tests and ad-hoc imports don't blow up.
    """
    global _RENDERER
    if _RENDERER is None:
        _RENDERER = Renderer()
    return _RENDERER


def reset_renderer_for_testing() -> None:
    global _RENDERER
    _RENDERER = None
