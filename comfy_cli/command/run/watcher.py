"""Background-watcher subprocess + transient pretty-mode status tail.

``_spawn_watcher`` detaches a Python subprocess that polls the server and
updates the job's state file. ``_tail_state_file`` is the foreground
companion — a short live-status display in pretty mode that exits cleanly
once the watcher takes over.
"""

from __future__ import annotations

import subprocess
import sys

from comfy_cli.output import get_renderer
from comfy_cli.output import rprint as pprint


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
            pprint(f"  [dim]→[/dim] [cyan]{u}[/cyan]")
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

    Fully decoupled from the parent: stdio redirected to /dev/null, a new
    session so a controlling terminal closing doesn't kill it. We don't
    track the PID — the watcher writes its own PID into the state file so
    callers can find it there if needed.

    Returns ``True`` on success, ``False`` if the subprocess could not be
    spawned.
    """
    argv = [sys.executable, "-m", "comfy_cli", "_watch", "_watch-job", prompt_id, "--where", where]
    if host:
        argv += ["--host", host]
    if port:
        argv += ["--port", str(port)]
    argv += ["--notify"] if notify else ["--no-notify"]
    try:
        subprocess.Popen(
            argv,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            stdin=subprocess.DEVNULL,
            start_new_session=True,
            close_fds=True,
        )
        return True
    except OSError:
        # Watcher spawn failed — the job still ran; the user just won't get
        # a state-file update without manual polling. Don't bail the submit.
        return False
