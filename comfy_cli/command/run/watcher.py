"""Background-watcher subprocess + transient pretty-mode status tail.

``_spawn_watcher`` detaches a Python subprocess that polls the server and
updates the job's state file. ``_tail_state_file`` is the foreground
companion — a short live-status display in pretty mode that exits cleanly
once the watcher takes over.
"""

from __future__ import annotations

import os
import subprocess
import sys

from comfy_cli.output import get_renderer
from comfy_cli.output import rprint as pprint
from comfy_cli.output.sanitize import sanitize_markup


def _no_watch_requested() -> bool:
    """``COMFY_NO_WATCH=1`` (or any other truthy value) suppresses the
    detached watcher subprocess entirely.

    This is the env kill switch for agentic callers: the cloud agent has its
    own native job-wait (Redis pub/sub + a reconcile GET) and has no use for
    a second, credential-holding, ``start_new_session=True`` process that
    outlives the parent and polls the jobs API for up to 6h. Checked as an
    env var (rather than threaded through every call site) so a host can set
    it once in the subprocess's environment without touching argv; ``comfy
    run --no-watch`` sets the same variable for interactive use.
    """
    value = os.environ.get("COMFY_NO_WATCH")
    if value is None:
        return False
    return value.strip().lower() not in {"", "0", "false", "no", "off"}


def _tail_state_file(prompt_id: str, *, seconds: float = 8.0) -> None:
    """Pretty-mode only: poll the state file for up to ``seconds`` showing
    live status transitions, then return. The background watcher keeps
    writing after we return — we just give the human a few seconds of
    "alive" feedback before the foreground exits.

    Always safe to call; no-ops if the renderer isn't pretty or the user
    Ctrl-Cs out.
    """
    import time as _t

    from rich.live import Live
    from rich.text import Text

    renderer = get_renderer()
    if not renderer.is_pretty():
        return

    from comfy_cli import jobs_state
    from comfy_cli.output.glyphs import status_glyph

    def render(status: str, elapsed: float) -> Text:
        return Text.from_markup(f"  {status_glyph(status)}  [dim]· {elapsed:.1f}s · {prompt_id[:8]}…[/dim]")

    deadline = _t.time() + seconds
    last_status = "queued"
    start = _t.time()
    try:
        with Live(render(last_status, 0.0), console=renderer.console(), refresh_per_second=4, transient=True) as live:
            while _t.time() < deadline:
                state = jobs_state.read(prompt_id)
                if state is not None:
                    last_status = state.status
                    live.update(render(last_status, _t.time() - start))
                    if state.is_terminal:
                        break
                _t.sleep(0.25)
    except KeyboardInterrupt:
        pass

    final_state = jobs_state.read(prompt_id)
    if final_state is None:
        return
    elapsed = _t.time() - start
    glyph = status_glyph(final_state.status)
    if final_state.is_terminal:
        pprint(f"  {glyph} [dim]· finished in {elapsed:.1f}s[/dim]")
        for u in (final_state.outputs or [])[:3]:
            pprint(f"  [dim]→[/dim] [cyan]{sanitize_markup(u)}[/cyan]")
    else:
        pprint(f"  {glyph} [dim]· still in flight — track:[/dim] [cyan]comfy jobs ls --watch[/cyan]")


def _spawn_watcher(
    prompt_id: str,
    *,
    where: str,
    host: str | None = None,
    port: int | None = None,
    notify: bool = False,
) -> bool:
    """Detach a watcher subprocess that polls + updates the state file.

    Fully decoupled from the parent: stdio redirected to /dev/null, and a
    detached process group so a controlling terminal closing doesn't kill it.
    POSIX gets its own session; Windows gets
    DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP for the equivalent, because
    ``start_new_session`` is POSIX-only and CPython ignores it there. We don't
    track the PID — the watcher writes its own PID into the state file so
    callers can find it there if needed.

    Returns ``True`` on success, ``False`` if the subprocess could not be
    spawned — or if suppressed entirely via ``COMFY_NO_WATCH=1``
    (see ``_no_watch_requested``).
    """
    if _no_watch_requested():
        return False
    argv = [sys.executable, "-m", "comfy_cli", "_watch", "_watch-job", prompt_id, "--where", where]
    if host:
        argv += ["--host", host]
    if port:
        argv += ["--port", str(port)]
    argv += ["--notify"] if notify else ["--no-notify"]

    kwargs: dict = {}
    if sys.platform == "win32":
        # Not module-level attributes on POSIX, hence the getattr lookups.
        detached = getattr(subprocess, "DETACHED_PROCESS", 0)
        new_group = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
        kwargs["creationflags"] = detached | new_group
    else:
        kwargs["start_new_session"] = True

    try:
        subprocess.Popen(
            argv,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            stdin=subprocess.DEVNULL,
            close_fds=True,
            **kwargs,
        )
        return True
    except (OSError, ValueError):
        # Watcher spawn failed — the job still ran; the user just won't get
        # a state-file update without manual polling. Don't bail the submit:
        # we're past the point of no return, the workflow is already queued.
        # ValueError covers Popen's argument-level rejections (an embedded NUL
        # in host/prompt_id, creationflags on a non-Windows platform), which
        # aren't OSError.
        return False
