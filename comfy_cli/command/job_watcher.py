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
# Consecutive *unreachable* liveness probes before we declare a local server
# dead: ~3 polls x 2s ≈ 6s of a refused port, which tolerates a transient blip
# / a quick server restart while still catching a real crash fast. Only a
# refused connection counts (see _probe_local_server), so a server that is
# merely slow can never accumulate this streak.
_SERVER_DOWN_CONSECUTIVE_LIMIT = 3
# How long to wait on a liveness probe. Bounds a TCP-reachable but wedged
# server (stuck in a CUDA kernel) without hanging the poll cycle.
_PROBE_TIMEOUT_S = 5.0
# Liveness verdicts. Only _PROBE_UNREACHABLE — nothing listening on the port —
# counts toward a death. A timeout or a non-200 means something *is* answering
# and is merely busy (loading a large model into VRAM pins the server for tens
# of seconds), which must never be mistaken for a crash.
_PROBE_ALIVE = "alive"
_PROBE_UNREACHABLE = "unreachable"
_PROBE_UNRESPONSIVE = "unresponsive"
# After a confirmed outage the server may come back as a *fresh* process whose
# queue and history no longer hold this prompt — the job died with the old one.
# A record that stays missing this long after an outage is a death too, rather
# than a zombie poll until the 6h ceiling.
_LOST_AFTER_RESTART_S = 60.0


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

    # Pid + create_time recorded together so the reaper can tell *this* watcher
    # from whatever inherits its pid later (see `_is_watcher_alive` in jobs.py).
    jobs_state.stamp_watcher_identity(state)
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
    # Local-server death detection: how many consecutive probes found the port
    # refused. Reset by any probe that reaches the server at all (see
    # _SERVER_DOWN_CONSECUTIVE_LIMIT). ``saw_outage`` latches once we have seen
    # a real outage, which is what makes a subsequently-missing record mean
    # "the server restarted without this job" rather than "not queued yet";
    # ``missing_since`` times that gap.
    consecutive_down = 0
    saw_outage = False
    missing_since: float | None = None
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
            # A local server that is OOM-killed mid-job takes the queue and the
            # history record with it, so `_snapshot` just reports "no record
            # yet" forever and the watcher would poll a corpse until the 6h
            # ceiling. Probe liveness explicitly so the death gets recorded.
            h, p = _resolve_watch_target(state, host, port)
            if _probe_local_server(h, p) == _PROBE_UNREACHABLE:
                consecutive_down += 1
                saw_outage = True
                if consecutive_down < _SERVER_DOWN_CONSECUTIVE_LIMIT:
                    # A blip or a restart, not a death yet. Skip the poll (it
                    # can only fail against a port that just refused us) and
                    # re-probe on the next cycle.
                    terminal = False
                else:
                    # One last poll before the verdict: the probe hits
                    # `/history`, the poll hits `/queue` + `/history/<id>`, so
                    # a job that finished right as the probe started failing
                    # can still be recovered if anything is still answering.
                    terminal, _ = _poll_local_once(state, host=host, port=port)
                    if not terminal:
                        prior = state.status
                        state = _finalize_server_died(
                            state,
                            message=(
                                f"ComfyUI server {h}:{p} became unreachable while job "
                                f"{state.prompt_id} was '{prior}'. The server likely crashed or was "
                                "killed while executing it (e.g. an out-of-memory allocation)."
                            ),
                            details={
                                "host": h,
                                "port": p,
                                "last_status": prior,
                                "consecutive_failed_probes": consecutive_down,
                            },
                        )
                    else:
                        jobs_state.write(state)
                    # Server confirmed gone — there is nothing left to watch.
                    break
            else:
                # Something answered the port. Even a timeout or an error
                # status means the server exists and is merely busy, so the
                # death streak resets and we poll as normal.
                consecutive_down = 0
                terminal, record_seen = _poll_local_once(state, host=host, port=port)
                if record_seen or not saw_outage:
                    missing_since = None
                elif missing_since is None:
                    # Back up after an outage, but this prompt is gone from the
                    # server: either it is still settling or the process that
                    # held the job was replaced. Give it a grace window.
                    missing_since = time.time()
                elif (missing_for := time.time() - missing_since) >= _LOST_AFTER_RESTART_S:
                    prior = state.status
                    state = _finalize_server_died(
                        state,
                        message=(
                            f"ComfyUI server {h}:{p} restarted while job {state.prompt_id} was "
                            f"'{prior}' and the new process has no record of it. The job died with "
                            "the server (it likely crashed or was killed, e.g. by the OOM killer)."
                        ),
                        details={
                            "host": h,
                            "port": p,
                            "last_status": prior,
                            "restarted": True,
                            "missing_for_s": round(missing_for, 1),
                        },
                    )
                    break

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


def _resolve_watch_target(state: jobs_state.JobState, host: str | None, port: int | None) -> tuple[str, int]:
    """The local ``(host, port)`` this watcher is bound to.

    Shared by the liveness probe and the poll so the two can never disagree
    about which server they are talking about.
    """
    from comfy_cli.env_checker import _bracket_host
    from comfy_cli.local_address import resolve_local_host_port

    # Per-job recorded state (state.host/port, captured when the job was
    # submitted) still wins over the env var, so a watcher keeps polling the
    # server it was launched against: flag > state > COMFY_LOCAL_URL > default.
    h, p = resolve_local_host_port(host or state.host, port or state.port)
    # Bracket IPv6 literals so ``_snapshot`` builds a well-formed URL (it takes
    # an already-bracketed host, like the `jobs` resolver produces). Delegates
    # to the shared ``_bracket_host`` choke point.
    return _bracket_host(h), p


def _probe_local_server(host: str, port: int) -> str:
    """Classify the local ComfyUI server as alive / unreachable / unresponsive.

    ``comfy jobs`` asks ``check_comfy_server_running`` a yes/no question, which
    collapses "the port is refused" and "the server took too long to answer"
    into the same ``False``. The watcher cannot: only the first is a crash, and
    treating a slow server as dead would file a terminal ``server_died`` error
    against a job that is still running. So it runs its own probe and keeps the
    distinction, with the same liveness definition (HTTP 200 → alive).

    ``max_items=1`` keeps the response small — the bare ``/history`` body grows
    without bound on a long-lived server, and a multi-megabyte read every
    ``_POLL_INTERVAL_S`` is both wasteful and a way to time the probe out.
    Servers too old to know the parameter ignore it and still answer 200.

    Never raises: an unexpected failure is reported as *unresponsive* (the
    non-committal verdict), because a probe must not crash the watcher and must
    not be able to invent a death on its own.
    """
    import requests

    try:
        # ``host`` arrives already bracketed from _resolve_watch_target.
        resp = requests.get(f"http://{host}:{port}/history?max_items=1", timeout=_PROBE_TIMEOUT_S)
    except requests.exceptions.Timeout:
        # Listed before ConnectionError: ConnectTimeout subclasses both, and a
        # timeout is never evidence that nothing is listening.
        return _PROBE_UNRESPONSIVE
    except requests.exceptions.ConnectionError:
        return _PROBE_UNREACHABLE
    except Exception:  # noqa: BLE001
        return _PROBE_UNRESPONSIVE
    return _PROBE_ALIVE if resp.status_code == 200 else _PROBE_UNRESPONSIVE


def _finalize_server_died(state: jobs_state.JobState, *, message: str, details: dict[str, Any]) -> jobs_state.JobState:
    """Persist the terminal ``server_died`` verdict; return the state to notify on.

    Re-reads the file rather than trusting the watcher's in-memory copy, and
    does the read and the write as one locked transaction: a concurrent
    ``comfy jobs cancel`` may have recorded a verdict of its own while we were
    failing probes, and a dead server must never invalidate a verdict the job
    already reached. ``jobs_state.write`` takes the same per-file lock, which
    is reentrant within a thread, so the nested acquire is a no-op.

    Any terminal state — on disk or in memory — wins and is returned unwritten.
    """
    from comfy_cli import locking

    lock_path = jobs_state.state_path(state.prompt_id).with_suffix(".lock")
    with locking.file_lock(lock_path):
        on_disk = jobs_state.read(state.prompt_id)
        if on_disk is not None and on_disk.is_terminal:
            return on_disk
        if state.is_terminal:
            return state
        state.status = "error"
        state.error = {"code": "server_died", "message": message, "details": details}
        jobs_state.write(state)
    return state


def _poll_local_once(state: jobs_state.JobState, *, host: str | None, port: int | None) -> tuple[bool, bool]:
    """Update ``state`` in-place from a local ComfyUI server.

    Returns ``(terminal, record_seen)``. ``record_seen`` distinguishes "the
    server has no record of this prompt" from "the record says it is still
    running", which the restart guard in ``watch_job`` needs and a bare
    terminal flag cannot express.
    """
    from comfy_cli.command import jobs as jobs_module

    h, p = _resolve_watch_target(state, host, port)
    try:
        snap = jobs_module._snapshot(h, p, state.prompt_id)
    except Exception as e:  # noqa: BLE001 — never crash the watcher on transient errors
        state.error = {"code": "watcher_poll_error", "message": str(e), "details": {}}
        return False, False

    if snap is None:
        # No record yet — keep polling.
        return False, False

    # Clear any transient poll error from a previous cycle.
    state.error = None
    snap_status = str(snap.get("status") or "queued")
    state.status = snap_status
    if snap_status == "completed":
        state.outputs = list(snap.get("outputs") or [])
        return True, True
    if snap_status == "error":
        state.error = {
            "code": "execution_error",
            "message": "ComfyUI reported an execution error.",
            "details": snap,
        }
        return True, True
    if snap_status == "cancelled":
        state.error = {
            "code": "cancelled",
            "message": "Job was interrupted/cancelled.",
            "details": {},
        }
        return True, True
    return False, True


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


def _cloud_record_meta(record: dict) -> dict[str, Any]:
    """The metadata fields a cloud terminal verdict attaches to ``details``.

    ``/api/jobs/<id>`` (``JobDetailResponse``) serves the timestamps as Unix
    millisecond ints; the deprecated ``/api/job/<id>/status`` served ready-made
    ``created_at``/``updated_at`` strings, and is the only dialect that ever
    served ``assigned_inference``. Read both, old names first, so the state
    file keeps the string shape it has always carried.
    """
    from comfy_cli.command.jobs import _ms_to_iso

    return {
        "assigned_inference": record.get("assigned_inference"),
        "created_at": record.get("created_at") or _ms_to_iso(record.get("create_time")),
        "updated_at": record.get("updated_at") or _ms_to_iso(record.get("update_time")),
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
        # Same endpoint move as `jobs._cloud_status_snapshot`: `/api/jobs/<id>`
        # serves the cause as a structured `execution_error` object, while the
        # deprecated `/api/job/<id>/status` served a JSON-encoded
        # `error_message` string. `classify` parses either shape, so hand it
        # whichever the deployment actually sent — without this the watcher
        # classifies `None` and writes the generic "ComfyUI reported an
        # execution error." into every failed cloud job's state file. The
        # structured object wins when both are present: a deployment that also
        # fills `error_message` with a short generic string would otherwise
        # discard `node_id`/`exception_type`/`traceback_tail`, and the state
        # file keeps no other copy of them. (`classify`'s details are built
        # field-by-field, so the secret-bearing `current_inputs` never reaches
        # the state file — see `execution_errors.redact_record`.)
        structured = record.get("execution_error")
        raw_cause = structured if isinstance(structured, dict) else (record.get("error_message") or structured)
        verdict = execution_errors.classify(raw_cause)
        state.error = {
            "code": verdict["code"],
            "message": verdict["message"],
            "hint": verdict["hint"],
            "details": {**verdict["details"], **_cloud_record_meta(record)},
        }
        return True
    if state.status == "cancelled":
        state.error = {
            "code": "cancelled",
            "message": record.get("error_message") or "Cloud job was cancelled.",
            "details": _cloud_record_meta(record),
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
