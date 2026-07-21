"""Detached background watcher for an in-flight prompt.

Invoked as ``comfy _watch-job <prompt_id> --where local|cloud …`` by
``comfy run`` when the user submits a workflow without ``--wait``. Polls
the server for status, mirrors the result into the on-disk state file
(see :mod:`comfy_cli.jobs_state`), and fires a system notification when
the prompt reaches a terminal state.

Hidden from the public surface — agents address jobs via
``comfy jobs status <id>`` or by reading the state file directly. This
command is purely the worker that the foreground ``run`` detaches.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import time
from typing import Annotated, Any

import typer

from comfy_cli import execution_errors, jobs_state

app = typer.Typer(hidden=True)


# How long to wait between polls. Cheap enough not to hammer the server,
# fast enough that a 30s job completes within one poll of finishing.
_POLL_INTERVAL_S = 2.0
# Absolute ceiling. A job that hasn't moved in this long → give up and let
# the file's last status stand. Equivalent to a stuck-process timeout.
_MAX_RUNTIME_S = 60 * 60 * 6  # 6 hours
# Statuses we recognize as legitimately in-flight: local snapshot values plus
# raw cloud statuses that pass through _CLOUD_STATUS_MAP unmapped. Anything
# else that is non-terminal is "unknown" — see the stall guard in watch_job.
_KNOWN_INFLIGHT_STATUSES = {"queued", "pending", "running", "executing", "allocated", "uploading"}
# An unknown status unchanged for this long → terminal error instead of
# letting the watcher idle for the full 6h ceiling on a status we can't map.
_UNKNOWN_STALL_S = 300.0


@app.command("_watch-job")
def watch_job(
    prompt_id: Annotated[str, typer.Argument()],
    where: Annotated[str, typer.Option("--where")],
    host: Annotated[str | None, typer.Option("--host")] = None,
    port: Annotated[int | None, typer.Option("--port")] = None,
    notify: Annotated[bool, typer.Option("--notify/--no-notify")] = True,
):
    state = jobs_state.read(prompt_id)
    if state is None:
        # No state file → nothing to watch. Submit must always write the
        # state file before spawning us, so this shouldn't happen in
        # practice; exit quietly.
        return

    state.watcher_pid = os.getpid()
    jobs_state.write(state)

    cloud_client = None
    if where == "cloud":
        try:
            from comfy_cli.comfy_client import Client
            from comfy_cli.target import resolve_target

            target = resolve_target(where="cloud")
            # Watcher context: read-mostly background poller. A reactive refresh
            # may freshen the access token, but a *fatal* refresh failure must
            # never clear the shared session — the foreground command owns the
            # session lifecycle, and a transient mid-run blip should not log the
            # user off.
            cloud_client = Client(target, timeout=30.0, clear_session_on_auth_failure=False)
        except Exception:  # noqa: BLE001
            pass

    start = time.time()
    # Unknown-status stall guard bookkeeping: the unrecognized status we are
    # currently watching, and when we first saw it.
    unknown_status: str | None = None
    unknown_since = 0.0
    while True:
        if time.time() - start > _MAX_RUNTIME_S:
            prior_status = state.status
            state.status = "error"
            state.error = {
                "code": "watcher_timeout",
                "message": f"Watcher gave up after {_MAX_RUNTIME_S}s without a terminal status.",
                "details": {"last_status": prior_status},
            }
            jobs_state.write(state)
            break

        if where == "cloud":
            terminal = _poll_cloud_once(state, client=cloud_client)
        else:
            terminal = _poll_local_once(state, host=host, port=port)

        jobs_state.write(state)
        if terminal:
            break

        # Stall guard: a non-terminal status we do not recognize (a future
        # cloud status missing from _CLOUD_STATUS_MAP) must not hang the
        # watcher for the full 6h ceiling. Unchanged for _UNKNOWN_STALL_S →
        # declare a terminal error naming the raw status.
        if state.status in _KNOWN_INFLIGHT_STATUSES:
            unknown_status = None
        else:
            now = time.time()
            if state.status != unknown_status:
                unknown_status = state.status
                unknown_since = now
            elif now - unknown_since >= _UNKNOWN_STALL_S:
                state.status = "error"
                state.error = {
                    "code": "unknown_status_stall",
                    "message": (
                        f"cloud reported unrecognized status {unknown_status!r} and it "
                        f"did not change within {_UNKNOWN_STALL_S:.0f}s; giving up"
                    ),
                    "details": {"raw_status": unknown_status, "stall_window_s": _UNKNOWN_STALL_S},
                }
                jobs_state.write(state)
                break

        time.sleep(_POLL_INTERVAL_S)

    if notify:
        _notify(state)


# ---------------------------------------------------------------------------
# polling backends
# ---------------------------------------------------------------------------


def _poll_local_once(state: jobs_state.JobState, *, host: str | None, port: int | None) -> bool:
    """Update ``state`` in-place from a local ComfyUI server. Return True if terminal."""
    from comfy_cli.command import jobs as jobs_module
    from comfy_cli.local_address import resolve_local_host_port

    # Per-job recorded state (state.host/port, captured when the job was
    # submitted) still wins over the env var, so a watcher keeps polling the
    # server it was launched against: flag > state > COMFY_LOCAL_URL > default.
    h, p = resolve_local_host_port(host or state.host, port or state.port)
    # Bracket IPv6 literals so ``_snapshot`` builds a well-formed URL (it takes
    # an already-bracketed host, like the `jobs` resolver produces).
    if ":" in h and not h.startswith("["):
        h = f"[{h}]"
    try:
        snap = jobs_module._snapshot(h, p, state.prompt_id)
    except Exception as e:  # noqa: BLE001 — never crash the watcher on transient errors
        state.error = {"code": "watcher_poll_error", "message": str(e), "details": {}}
        return False

    if snap is None:
        # No record yet — keep polling.
        return False

    # Clear any transient poll error from a previous cycle.
    state.error = None
    snap_status = str(snap.get("status") or "queued")
    state.status = snap_status
    if snap_status == "completed":
        state.outputs = list(snap.get("outputs") or [])
        return True
    if snap_status == "error":
        state.error = {
            "code": "execution_error",
            "message": "ComfyUI reported an execution error.",
            "details": snap,
        }
        return True
    if snap_status == "cancelled":
        state.error = {
            "code": "cancelled",
            "message": "Job was interrupted/cancelled.",
            "details": {},
        }
        return True
    return False


_CLOUD_STATUS_MAP = {
    "success": "completed",
    "completed": "completed",
    "failed": "error",
    "error": "error",
    "non_retryable_error": "error",
    "lost": "error",
    "cancelled": "cancelled",
    "canceled": "cancelled",
}


def _poll_cloud_once(state: jobs_state.JobState, *, client: Any = None) -> bool:
    """Update ``state`` in-place from Comfy Cloud. Return True if terminal."""
    try:
        if client is None:
            from comfy_cli.comfy_client import Client
            from comfy_cli.target import resolve_target

            target = resolve_target(where="cloud")
            # Watcher context — never clear the shared session on a fatal
            # refresh failure (see the note in ``watch_job``).
            client = Client(target, timeout=30.0, clear_session_on_auth_failure=False)
        record = client.get_job_status(state.prompt_id)
    except Exception as e:  # noqa: BLE001
        state.error = {"code": "watcher_poll_error", "message": str(e), "details": {}}
        return False

    if record is None:
        return False

    # Clear any transient poll error from a previous cycle.
    state.error = None
    raw = str(record.get("status") or "queued").lower()
    state.status = _CLOUD_STATUS_MAP.get(raw, raw)

    if state.status == "completed":
        # Cloud's /api/jobs/<id> detail response sometimes includes outputs directly;
        # if not, fetch from history. Match the snapshot logic in jobs.py.
        outputs = record.get("outputs")
        if isinstance(outputs, list) and outputs:
            state.outputs = list(outputs)
        else:
            try:
                history = client.get_history(state.prompt_id)
                if history:
                    # Stash the full node-keyed record so downstream consumers
                    # (grouped outputs, item-named downloads) need no extra call.
                    state.record = history
                    state.outputs = client.extract_output_urls(history)
            except Exception:  # noqa: BLE001 — best effort, state already terminal
                pass
        return True
    if state.status == "error":
        verdict = execution_errors.classify(record.get("error_message"))
        state.error = {
            "code": verdict["code"],
            "message": verdict["message"],
            "hint": verdict["hint"],
            "details": {
                **verdict["details"],
                **{k: record.get(k) for k in ("assigned_inference", "created_at", "updated_at")},
            },
        }
        return True
    if state.status == "cancelled":
        state.error = {
            "code": "cancelled",
            "message": record.get("error_message") or "Cloud job was cancelled.",
            "details": {k: record.get(k) for k in ("assigned_inference", "created_at", "updated_at")},
        }
        return True
    return False


# ---------------------------------------------------------------------------
# notification
# ---------------------------------------------------------------------------


def _notify(state: jobs_state.JobState) -> None:
    """Best-effort system notification. Silently no-ops if unavailable.

    macOS: ``osascript -e 'display notification …'``
    Linux: ``notify-send``
    Fallback: write to stderr (and ring the terminal bell).
    """
    title = "comfy"
    short_id = state.prompt_id[:8]
    if state.status == "completed":
        body = f"✓ {short_id} completed ({len(state.outputs)} output(s))"
    elif state.status == "error":
        body = f"✗ {short_id} failed: {(state.error or {}).get('message', 'unknown')[:120]}"
    elif state.status == "cancelled":
        body = f"⊘ {short_id} cancelled"
    else:
        body = f"{short_id} → {state.status}"

    if sys.platform == "darwin" and shutil.which("osascript"):
        _try_run(["osascript", "-e", f"display notification {_apple_quote(body)} with title {_apple_quote(title)}"])
        return
    if shutil.which("notify-send"):
        _try_run(["notify-send", title, body])
        return
    # Fallback — write to stderr so a still-attached terminal sees it.
    try:
        sys.stderr.write(f"\a[{title}] {body}\n")
        sys.stderr.flush()
    except OSError:
        pass


def _try_run(argv: list[str]) -> None:
    try:
        subprocess.run(argv, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False, timeout=5)
    except (OSError, subprocess.SubprocessError):
        pass


def _apple_quote(s: str) -> str:
    """Quote a string for AppleScript: wrap in double quotes, escape \\ and \"."""
    escaped = s.replace("\\", "\\\\").replace('"', '\\"')
    return f'"{escaped}"'
