"""``comfy jobs`` — list, status, and live-watch ComfyUI prompts.

The ComfyUI server already speaks WebSocket: every node-execution event is
pushed to every connected ``/ws?clientId=…`` client, tagged with the
``prompt_id`` it belongs to. We use that channel as the live "push" feed —
no daemon, no polling.

Three subcommands:

- ``comfy jobs ls``        — combine ``/queue`` (running + pending) and
                              ``/history`` (recent completions) into one
                              ordered list.
- ``comfy jobs status``    — one-shot snapshot of a single prompt_id from
                              ``/history``.
- ``comfy jobs watch``     — live-tail the WS feed, filter on prompt_id,
                              emit events as they arrive (pretty progress
                              bar or NDJSON depending on mode).
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from dataclasses import dataclass, field, replace
from datetime import datetime
from typing import TYPE_CHECKING, Annotated, Any

import typer
from websocket import WebSocket, WebSocketException, WebSocketTimeoutException

from comfy_cli import cancellation, execution_errors, tracking
from comfy_cli.env_checker import check_comfy_server_running
from comfy_cli.host_port import resolve_host_port as _resolve_host_port
from comfy_cli.http import ResponseTooLarge, authed_urlopen, plain_urlopen, read_capped
from comfy_cli.output import get_renderer
from comfy_cli.output.sanitize import sanitize, sanitize_markup
from comfy_cli.where import cloud_preflight_or_exit

if TYPE_CHECKING:
    from comfy_cli.jobs_state import JobState

app = typer.Typer(no_args_is_help=True, help="List, inspect, and live-watch ComfyUI prompts.")


def _is_pid_alive(pid: int) -> bool:
    """Check if a process with the given PID is still running.

    Uses ``psutil.pid_exists`` — never ``os.kill(pid, 0)``, which on Windows
    routes through ``GenerateConsoleCtrlEvent`` (0 == CTRL_C_EVENT) and, on
    Python <= 3.13.1, can fall through to ``TerminateProcess`` and kill the
    probed process (python/cpython gh-58689).
    """
    if pid <= 0:
        return False
    import psutil

    try:
        return psutil.pid_exists(pid)
    except (OverflowError, ValueError, OSError):
        # `watcher_pid` comes off a deliberately tolerant JSON load with no
        # range check, so a corrupt or hand-edited state file can carry a pid
        # psutil can't even look up (out-of-range -> OverflowError; Windows
        # OpenProcess -> OSError). One bad record must read as dead, not abort
        # the scan and take `comfy jobs ls` down with it.
        return False


# Same discriminator (and tolerance) the download-worker liveness check uses.
_PID_CREATE_TIME_TOLERANCE_S = 1.0


def _is_watcher_alive(state: JobState) -> bool:
    """True while ``state``'s watcher pid is live *and still that watcher*.

    Liveness alone is not proof of identity: pids get recycled (aggressively on
    Windows) and job state files outlive the runs that wrote them by days, so a
    bare existence check eventually pins a dead job at ``running`` behind a
    stranger's process — never reaped, never visible under ``--orphaned``. The
    watcher records its own start time next to its pid, so the pair either
    matches a live process or it doesn't; this mirrors
    ``download_state.is_worker_process``.

    Records written before that field existed carry ``None`` and fall back to
    liveness alone — exactly what they got before.
    """
    pid = state.watcher_pid
    if pid is None or pid <= 0:
        return False
    if not _is_pid_alive(pid):
        return False
    if state.watcher_pid_create_time is None:
        return True
    try:
        import psutil

        started = psutil.Process(pid).create_time()
    except Exception:  # noqa: BLE001 — gone, or not ours to inspect
        return False
    return abs(started - state.watcher_pid_create_time) <= _PID_CREATE_TIME_TOLERANCE_S


# Host/port resolution (`resolve_host_port`) is shared with `comfy run` via
# `comfy_cli.host_port`; imported above as `_resolve_host_port` to preserve the
# call sites in this module unchanged.


def _server_or_error(host: str, port: int, *, raise_on_missing: bool = True) -> bool:
    """Return True if the server is up. If ``raise_on_missing`` is True (the
    default) we emit the error envelope and exit; if False, we return False so
    the caller can fall back to a different source (e.g. on-disk state files).
    """
    if check_comfy_server_running(port, host):
        return True
    if not raise_on_missing:
        return False
    renderer = get_renderer()
    renderer.error(
        code="server_not_running",
        message=f"ComfyUI not running on {host}:{port}",
        hint="run: comfy launch",
        details={"host": host, "port": port},
    )
    raise typer.Exit(code=1)


def _http_get_json(url: str, *, timeout: float = 10.0) -> Any:
    """GET a JSON body from the local ComfyUI server.

    The read is capped (``read_capped``) so a server streaming an endless
    ``/history`` can't OOM the CLI. Every failure — unreachable, oversize,
    non-JSON — leaves as a ``RuntimeError``, which is the single family every
    call site below already catches.
    """
    req = urllib.request.Request(url)
    try:
        with plain_urlopen(req, timeout=timeout) as resp:
            return json.loads(read_capped(resp, url))
    except urllib.error.URLError as e:
        raise RuntimeError(f"failed to GET {url}: {e}") from e
    except ResponseTooLarge as e:
        raise RuntimeError(str(e)) from e
    except ValueError as e:
        # json.JSONDecodeError is a ValueError — a non-JSON 200 (captive
        # portal, proxy error page) must look like any other GET failure to
        # callers, not crash them with an uncaught decode error.
        raise RuntimeError(f"failed to parse JSON from {url}: {e}") from e


# ---------------------------------------------------------------------------
# Terminal job verdict — shared helper for `jobs watch` (local + cloud)
# ---------------------------------------------------------------------------

# Terminal job status -> (ok, error_code, exit_code). A failed or cancelled job
# must surface ok:false + non-zero exit so `comfy --json jobs watch $ID && next`
# stops. Mirrors `run --wait` (command/run/__init__.py).
_TERMINAL_VERDICT: dict[str, tuple[bool, str | None, int]] = {
    "completed": (True, None, 0),
    "error": (False, "execution_error", 1),
    "cancelled": (False, "cancelled", 130),
}


def _emit_terminal(renderer, payload: dict, *, command: str, where: str | None = None) -> None:
    """Emit a job's terminal envelope with ok/exit derived from its status.

    completed -> ok:true exit 0; error -> ok:false exit 1; cancelled -> exit 130.
    Unknown statuses default to ok:true exit 0 (non-terminal/best-effort).
    """
    status = str(payload.get("status") or "unknown")
    ok, code, exit_code = _TERMINAL_VERDICT.get(status, (True, None, 0))
    if ok:
        renderer.emit(payload, command=command, where=where)
        return
    err = payload.get("error")
    raw = err.get("message") if isinstance(err, dict) and err.get("message") else None
    if not raw:
        # Cloud snapshots (_cloud_status_snapshot) carry the failure text at
        # top-level `error_message`; the local WS path carries the decoded
        # execution_error event dict under `details`.
        raw = payload.get("error_message") or payload.get("details")
    message = raw if isinstance(raw, str) else None
    hint = None
    if status == "error":
        verdict = execution_errors.classify(raw)
        code, message, hint = verdict["code"], verdict["message"], verdict["hint"]
        # The raw server text repeats the full traceback; keep the envelope to
        # the one-line cause + structured tail and leave the full record to
        # `jobs status`.
        if payload.get("error_message"):
            payload["error_message"] = verdict["message"]
        if isinstance(payload.get("details"), dict) and "traceback" in payload["details"]:
            payload["details"] = verdict["details"]
        if isinstance(err, dict):
            payload["error"] = {**err, "code": code, "message": verdict["message"], "details": verdict["details"]}
    renderer.error(
        code=code or "execution_error",
        message=message or f"job {payload.get('prompt_id')} ended in status {status!r}",
        hint=hint,
        details=payload,
        exit_code=exit_code,
        command=command,
    )
    raise typer.Exit(code=exit_code)


# ---------------------------------------------------------------------------
# `jobs ls` — combine /queue + /history
# ---------------------------------------------------------------------------


# Row statuses that mean "this job failed" — the ones an `error_code` is
# meaningful alongside. `completed` is terminal but deliberately absent.
_ERROR_STATUSES = frozenset({"error", "cancelled"})

# Cloud's raw job statuses → the row vocabulary above. Mirrors
# ``job_watcher._CLOUD_STATUS_MAP`` (kept as a local copy rather than imported:
# ``job_watcher`` imports this module, so the dependency only runs one way).
# The failure spellings matter here — an unmapped `non_retryable_error` /
# `lost` / `canceled` misses `_ERROR_STATUSES` and silently drops the state
# file's `error_code` for exactly the jobs that failed.
_CLOUD_ROW_STATUS_MAP = {
    "success": "completed",
    "completed": "completed",
    "failed": "error",
    "error": "error",
    "non_retryable_error": "error",
    "lost": "error",
    "cancelled": "cancelled",
    "canceled": "cancelled",
}


@dataclass(frozen=True)
class JobRow:
    prompt_id: str
    status: str  # "running" | "pending" | "completed" | "error" | "cancelled" | "allocated" | "executing"
    queue_position: int | None
    elapsed_seconds: float | None
    workflow_size: int | None  # number of nodes
    outputs: int
    where: str = "local"  # "local" | "cloud"
    workflow_path: str | None = None  # set when sourced from a state file
    updated_at: str | None = None  # ISO timestamp, set for state-file rows
    # `error.code` off the state file (e.g. "server_died", "execution_error",
    # "watcher_crashed"). Without it a listing can say a job is in `error` and
    # name its workflow, but never say *why* — which is the whole point of the
    # state file surviving a server death. Only ever set alongside a status in
    # `_ERROR_STATUSES`: the watcher parks a transient `watcher_poll_error` on
    # the state file of a job that is still healthily `running`, and that is a
    # poll blip, not this row's failure cause.
    error_code: str | None = None


def _state_error_code(err: Any, status: Any) -> str | None:
    """Pull ``error.code`` out of a state file's ``error`` blob, or None.

    Returns None unless ``status`` is one of ``_ERROR_STATUSES``. A non-failed
    job can carry an ``error`` blob: ``job_watcher._poll_local_once`` /
    ``_poll_cloud_once`` record ``watcher_poll_error`` when a single poll
    raises, leave the status at ``queued``/``running``, and only clear it on a
    later poll that actually returns a snapshot — so an in-flight job can hold
    that code for many cycles. Surfacing it as ``error_code`` would advertise a
    failure cause for a job that has not failed.

    ``jobs_state.read`` keeps known keys' values untouched, so a hand-edited or
    truncated file can carry a non-dict ``error`` (or a non-string ``code``) —
    neither may reach the emitted envelope, where ``error_code`` is typed
    ``string | null``.
    """
    if status not in _ERROR_STATUSES:
        return None
    if not isinstance(err, dict):
        return None
    code = err.get("code")
    return code if isinstance(code, str) and code else None


def _state_str(value: Any) -> str | None:
    """Narrow a state-file value to ``str | None`` for the emitted envelope.

    Same defensiveness as ``_state_error_code``, for the fields published as
    ``string | null``: ``jobs_state.read`` type-checks nothing it keeps, and a
    numeric ``updated_at`` from a hand-edited or legacy file would otherwise
    reach ``_merge_jobs``'s ``sort_key`` and raise ``TypeError`` comparing int
    against str — aborting the whole listing, not just one row.
    """
    return value if isinstance(value, str) and value else None


def _state_where(value: Any) -> str:
    """Narrow a state file's ``where`` to the published ``local``/``cloud`` enum.

    Missing, empty, or unrecognized (a legacy ``"remote"``) reads as ``local``,
    matching the filter in ``_gather_local_state_files`` — so a row can never be
    scoped as local yet report a ``where`` the schema rejects.
    """
    return value if value in ("local", "cloud") else "local"


def _gather_local_state_files(*, limit: int, orphaned_only: bool = False, where: str | None = None) -> list[JobRow]:
    """Read every state file in the jobs state dir → JobRow.

    This is the canonical "what did *I* submit via this CLI" view —
    independent of whether the server is reachable. Surfaces async submits
    that the user otherwise wouldn't see in `jobs ls`.

    When ``orphaned_only`` is True, return only rows whose state file
    has ``error.code == "watcher_crashed"`` — jobs where the
    background watcher died and was reaped. Useful for cleanup.

    ``where`` scopes the rows to one routing target (``"local"`` /
    ``"cloud"``) so a ``--where local`` listing can't surface cloud jobs
    submitted in an earlier run. A state file whose ``where`` is missing or
    empty counts as ``"local"``. ``None`` (the default) keeps the unfiltered
    union view — used by ``jobs ls --all`` and ``--orphaned``.
    """
    import re as _re

    from comfy_cli import jobs_state

    # Reasonable prompt_ids are alphanumeric + dashes + underscores (UUIDs,
    # short hex IDs). Anything wilder (e.g. legacy MagicMock leak from tests)
    # is filtered so ``jobs ls`` stays clean.
    _SANE_ID = _re.compile(r"^[A-Za-z0-9_-]{1,128}$")
    rows: list[JobRow] = []
    state_dir = jobs_state.state_dir()
    for path in sorted(state_dir.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True):
        if not _SANE_ID.match(path.stem):
            continue
        state = jobs_state.read(path.stem)
        if state is None:
            continue
        # Reap stale watchers: if the job is non-terminal and the watcher
        # PID is recorded but dead, mark the job as errored so it doesn't
        # sit as "running" forever.
        if (
            not state.is_terminal
            and state.watcher_pid is not None
            and state.watcher_pid > 0
            and not _is_watcher_alive(state)
        ):
            state.status = "error"
            state.error = {
                "code": "watcher_crashed",
                "message": f"Background watcher (pid {state.watcher_pid}) is no longer running.",
                "hint": "re-submit the workflow, or check `comfy jobs status <id>` against the server",
            }
            state.watcher_pid = None
            state.watcher_pid_create_time = None
            jobs_state.write(state)
        # Scope to the resolved --where target. Done *after* the stale-watcher
        # reap above so cleanup stays where-agnostic no matter which view the
        # caller asked for.
        if where is not None and _state_where(state.where) != where:
            continue
        if orphaned_only:
            err = state.error or {}
            if not (isinstance(err, dict) and err.get("code") == "watcher_crashed"):
                continue
        rows.append(
            JobRow(
                prompt_id=state.prompt_id,
                status=state.status,
                queue_position=None,
                elapsed_seconds=None,
                workflow_size=None,
                outputs=len(state.outputs or []),
                # Same narrowing the filter above uses, so a row can never be
                # scoped as local yet report a `where` outside the published
                # enum (JobRow.where is typed `str` and defaults to "local").
                where=_state_where(state.where),
                workflow_path=_state_str(state.workflow),
                updated_at=_state_str(state.updated_at),
                error_code=_state_error_code(state.error, state.status),
            )
        )
        if len(rows) >= limit:
            break
    return rows


def _parse_epoch(ts: str | None) -> float:
    """Parse an ISO ``updated_at`` to epoch seconds; 0.0 if missing/unparseable."""
    if not ts:
        return 0.0
    try:
        return datetime.fromisoformat(ts).timestamp()
    except (ValueError, TypeError):
        return 0.0


def _merge_jobs(state_rows: list[JobRow], server_rows: list[JobRow]) -> list[JobRow]:
    """Server's view wins for prompts it knows about (fresher status); state
    files fill in everything else (jobs the server doesn't see, e.g. cloud
    jobs viewed from a local-only `jobs ls`).

    "Wins" is per-row, not per-field: several fields exist only on the state
    file, and a server row that supersedes one would otherwise blank them.

    - ``error_code``: neither ``/queue``/``/history`` nor the cloud job list
      carries one, so the cause the state file recorded would be lost. Carried
      across only when *both* views call the job failed — the state file's,
      because a code recorded next to a healthy status is a watcher poll blip
      rather than this job's cause; the server's, so a state file left holding
      a stale ``server_died`` can't contradict a server that now reports the
      prompt as completed.
    - ``workflow_path`` / ``updated_at``: also state-file-only. Blanking
      ``workflow_path`` drops the one field that says *which workflow* a
      prompt_id was, and blanking ``updated_at`` sorts a terminal row to epoch
      0 below every dated one, where the caller's ``[:limit]`` slice can drop a
      fresh completion. Carried whenever the server row has nothing to say.

    ``where`` is deliberately *not* carried: server rows now set it themselves
    (``_cloud_job_to_row`` marks cloud, ``_gather_jobs`` is local by
    construction), so the server row is authoritative.

    The prior row is looked up from a snapshot of ``state_rows`` rather than
    from the accumulating map: ``/queue`` and ``/history`` are fetched
    separately, so one gather can yield two rows for a transitioning prompt
    (``running``, then ``error``), and reading the map would let the first one
    clobber the state row before the second one gets to inherit from it.
    """
    state_by_id: dict[str, JobRow] = {r.prompt_id: r for r in state_rows}
    by_id: dict[str, JobRow] = dict(state_by_id)
    for r in server_rows:
        prior = state_by_id.get(r.prompt_id)
        if prior is not None:
            carried: dict[str, Any] = {}
            if (
                r.error_code is None
                and prior.error_code is not None
                and prior.status in _ERROR_STATUSES
                and r.status in _ERROR_STATUSES
            ):
                carried["error_code"] = prior.error_code
            if r.workflow_path is None and prior.workflow_path is not None:
                carried["workflow_path"] = prior.workflow_path
            if r.updated_at is None and prior.updated_at is not None:
                carried["updated_at"] = prior.updated_at
            if carried:
                r = replace(r, **carried)
        by_id[r.prompt_id] = r

    # Sort: non-terminal first (running/pending/allocated/executing), then
    # terminal ones by updated_at desc (freshest completions first, so the
    # caller's slice keeps the newest results).
    def sort_key(r: JobRow) -> tuple[int, float | str]:
        terminal = r.status in {"completed", "error", "cancelled"}
        if terminal:
            return (1, -_parse_epoch(r.updated_at))
        return (0, "" if not r.updated_at else r.updated_at)

    return sorted(by_id.values(), key=sort_key, reverse=False)


def _gather_jobs(host: str, port: int, *, limit: int) -> list[JobRow]:
    """Pull running + pending + recent history; merge into a single ordered list."""
    rows: list[JobRow] = []
    try:
        queue = _http_get_json(f"http://{host}:{port}/queue")
    except RuntimeError:
        queue = {"queue_running": [], "queue_pending": []}

    for i, entry in enumerate(queue.get("queue_running") or []):
        prompt_id, wf = _safe_queue_entry(entry)
        rows.append(
            JobRow(
                prompt_id=prompt_id,
                status="running",
                queue_position=None,
                elapsed_seconds=None,
                workflow_size=len(wf) if isinstance(wf, dict) else None,
                outputs=0,
            )
        )
    for i, entry in enumerate(queue.get("queue_pending") or []):
        prompt_id, wf = _safe_queue_entry(entry)
        rows.append(
            JobRow(
                prompt_id=prompt_id,
                status="pending",
                queue_position=i + 1,
                elapsed_seconds=None,
                workflow_size=len(wf) if isinstance(wf, dict) else None,
                outputs=0,
            )
        )

    try:
        history = _http_get_json(f"http://{host}:{port}/history")
    except RuntimeError:
        history = {}
    if not isinstance(history, dict):
        history = {}

    history_items = list(history.items())
    # /history is keyed by prompt_id; values include prompt + outputs. Order
    # isn't documented but recent entries are typically last — pull the tail.
    history_items = history_items[-limit:]
    for prompt_id, body in reversed(history_items):
        if not isinstance(body, dict):
            continue
        status_obj = body.get("status") or {}
        completed = status_obj.get("completed")
        # error: status_str == "error" OR any message of type execution_error
        status_str = "completed" if completed else "error"
        messages = status_obj.get("messages") or []
        for msg in messages:
            if isinstance(msg, list) and msg and msg[0] == "execution_error":
                status_str = "error"
                break
        outputs = body.get("outputs") or {}
        output_count = sum(
            len(items)
            for v in outputs.values()
            if isinstance(v, dict)
            for key in ("images", "gifs", "videos", "audio", "files")
            for items in [v.get(key) or []]
            if isinstance(items, list)
        )
        wf = body.get("prompt") or [None, None, None, None]
        wf_dict = wf[2] if isinstance(wf, list) and len(wf) > 2 else None
        rows.append(
            JobRow(
                prompt_id=str(prompt_id),
                status=status_str,
                queue_position=None,
                elapsed_seconds=None,
                workflow_size=len(wf_dict) if isinstance(wf_dict, dict) else None,
                outputs=output_count,
            )
        )
    return rows


def _safe_queue_entry(entry: Any) -> tuple[str, Any]:
    """ComfyUI /queue rows are [<num>, <prompt_id>, <prompt_dict>, ...]."""
    if isinstance(entry, list) and len(entry) >= 3:
        return str(entry[1]), entry[2]
    return ("?", None)


@app.command(
    "ls",
    help=(
        "List jobs: locally-tracked async submits + server queue/history, "
        "scoped to the resolved --where target (use --all for every target)."
    ),
)
@tracking.track_command("jobs")
def ls_cmd(
    host: Annotated[str | None, typer.Option(help="Server host (defaults to background or 127.0.0.1).")] = None,
    port: Annotated[int | None, typer.Option(help="Server port (defaults to background or 8188).")] = None,
    limit: Annotated[int, typer.Option(help="How many history entries to include.")] = 10,
    where: Annotated[
        str | None,
        typer.Option("--where", help="'local' (default) or 'cloud'. Cloud requires `comfy cloud login`."),
    ] = None,
    local_only: Annotated[
        bool,
        typer.Option(
            "--local-only",
            show_default=False,
            help="Only read the on-disk state files; skip the server query. Useful when offline.",
        ),
    ] = False,
    orphaned: Annotated[
        bool,
        typer.Option(
            "--orphaned",
            show_default=False,
            help=(
                "Show only jobs whose background watcher died (error.code == "
                "watcher_crashed). Implies --local-only because the server "
                "doesn't track watcher liveness."
            ),
        ),
    ] = False,
    all_wheres: Annotated[
        bool,
        typer.Option(
            "--all",
            show_default=False,
            help="Show state-file rows for every target, not just the resolved --where.",
        ),
    ] = False,
    watch: Annotated[
        bool,
        typer.Option(
            "--watch",
            show_default=False,
            help="Live-refresh the table every 2s (pretty mode only). Ctrl-C to exit.",
        ),
    ] = False,
):
    renderer = get_renderer()

    # Resolve the routing target once: per-command --where flag > COMFY_WHERE
    # env (how the top-level `comfy --where` arrives) > config default. Both
    # the server query and the state-file scope key off this single decision.
    target_where = "cloud" if _is_cloud(where) else "local"

    # --orphaned only makes sense for state files (the server doesn't know
    # whether a watcher crashed), so skip the server query in that mode.
    if orphaned:
        local_only = True

    # State-file rows are scoped to the resolved target, so a `--where local`
    # listing can't surface cloud jobs from an earlier run; `--all` restores
    # the union view. `--orphaned` stays unfiltered — watcher cleanup is
    # where-agnostic. `server_rows` are never filtered: they are already
    # scoped by which backend we queried.
    #
    # Both decisions are made *before* the --watch branch so the live table
    # applies exactly the same filters as the one-shot listing.
    state_where = None if (all_wheres or orphaned) else target_where

    if watch:
        if not renderer.is_pretty():
            renderer.error(
                code="json_incompatible",
                message="--watch requires pretty mode (TTY). For JSON, poll with a shell loop.",
                hint="drop --json, or run `while true; do comfy --json jobs ls; sleep 2; done`",
            )
            raise typer.Exit(code=1)
        _watch_ls(
            host=host,
            port=port,
            limit=limit,
            where=where,
            local_only=local_only,
            state_where=state_where,
            orphaned_only=orphaned,
        )
        return

    state_rows = _gather_local_state_files(limit=limit, orphaned_only=orphaned, where=state_where)

    server_rows: list[JobRow] = []
    h, p = _resolve_host_port(host, port)
    if not local_only:
        if target_where == "cloud":
            try:
                cloud_preflight_or_exit()
                client = _cloud_client()
                server_rows = [_cloud_job_to_row(j) for j in client.list_jobs(limit=limit)]
            except typer.Exit:
                # Preflight surfaced an error envelope already. Fall through
                # to state-only view; the local files are still useful.
                pass
        else:
            try:
                _server_or_error(h, p, raise_on_missing=False)
                server_rows = _gather_jobs(h, p, limit=limit)
            except RuntimeError:
                # Server unreachable — that's fine, state files cover us.
                pass

    rows = _merge_jobs(state_rows, server_rows)[:limit]

    if renderer.is_pretty():
        _render_jobs_pretty(rows, host=h if target_where != "cloud" else "cloud.comfy.org", port=p)
    renderer.emit(
        {
            "host": h,
            "port": p,
            "where": target_where,
            # Which state-file view the caller got: the resolved target, or
            # "all" when the union view was requested (--all/--orphaned).
            "scope": state_where or "all",
            "count": len(rows),
            "jobs": [_row_to_dict(r) for r in rows],
        },
        command="jobs ls",
    )


def _watch_ls(*, host, port, limit, where, local_only, state_where=None, orphaned_only=False):
    """Rich Live refresh of the jobs table every 2s until Ctrl-C.

    ``state_where`` scopes the state-file rows exactly as the one-shot path
    does (``None`` = the unfiltered union view, i.e. ``--all``), and
    ``orphaned_only`` mirrors ``--orphaned``, so the live table shows the same
    jobs as ``jobs ls``.
    """
    import time

    from rich.live import Live
    from rich.table import Table

    from comfy_cli.output.glyphs import status_glyph

    renderer = get_renderer()
    console = renderer.console()
    h, p = _resolve_host_port(host, port)

    def build_table() -> Table:
        state_rows = _gather_local_state_files(limit=limit, where=state_where, orphaned_only=orphaned_only)
        server_rows: list[JobRow] = []
        if not local_only:
            try:
                if _is_cloud(where):
                    client = _cloud_client()
                    server_rows = [_cloud_job_to_row(j) for j in client.list_jobs(limit=limit)]
                else:
                    server_rows = _gather_jobs(h, p, limit=limit)
            except (RuntimeError, Exception):  # noqa: BLE001 — best effort, keep refreshing
                pass
        rows = _merge_jobs(state_rows, server_rows)[:limit]

        title_loc = "cloud.comfy.org" if _is_cloud(where) else f"{h}:{p}"
        tbl = Table(
            title=f"Jobs ({title_loc}) — refreshing every 2s · Ctrl-C to exit",
            show_header=True,
            header_style="bold magenta",
            border_style="cyan",
            pad_edge=False,
        )
        tbl.add_column("prompt_id", style="bold white", no_wrap=True, overflow="fold")
        tbl.add_column("status", no_wrap=True)
        tbl.add_column("where", style="dim", no_wrap=True)
        tbl.add_column("outputs", no_wrap=True, justify="right")
        tbl.add_column("workflow", style="dim", overflow="fold")
        for r in rows:
            wf_display = ""
            if r.workflow_path:
                from pathlib import Path

                wf_display = Path(r.workflow_path).name
            tbl.add_row(
                sanitize_markup(r.prompt_id[:8] + "…" if len(r.prompt_id) > 8 else r.prompt_id),
                status_glyph(r.status),
                sanitize_markup(r.where),
                str(r.outputs) if r.outputs else "—",
                sanitize_markup(wf_display),
            )
        if not rows:
            tbl.add_row("[dim]no jobs[/dim]", "", "", "", "")
        return tbl

    try:
        with Live(build_table(), console=console, refresh_per_second=2) as live:
            while True:
                time.sleep(2)
                live.update(build_table())
    except KeyboardInterrupt:
        # Clean exit — Rich Live tears down automatically.
        return


def _row_to_dict(r: JobRow) -> dict:
    return {
        "prompt_id": r.prompt_id,
        "status": r.status,
        "queue_position": r.queue_position,
        "workflow_size": r.workflow_size,
        "outputs": r.outputs,
        "where": r.where,
        "workflow_path": r.workflow_path,
        "updated_at": r.updated_at,
        # Why an `error` row failed, when the state file knows. `jobs status`
        # points callers at `comfy jobs ls` as the escape hatch after a server
        # death, so the hatch has to be able to name the cause.
        "error_code": r.error_code,
    }


def _render_jobs_pretty(rows: list[JobRow], *, host: str, port: int) -> None:
    from rich.table import Table

    from comfy_cli.config_manager import ConfigManager
    from comfy_cli.output.branding import branded_panel
    from comfy_cli.output.glyphs import status_glyph

    renderer = get_renderer()
    is_cloud = str(host).startswith("http") or host == "cloud.comfy.org"
    where_label = "cloud" if is_cloud else "local"
    host_label = host if is_cloud else f"{host}:{port}"

    if not rows:
        empty = "[dim]No jobs.[/dim]\n[dim]→ comfy run --workflow X.json[/dim]"
        renderer.console().print(
            branded_panel(
                empty,
                title="jobs",
                version=ConfigManager().get_cli_version(),
                where=where_label,
                host=host_label,
            )
        )
        return

    tbl = Table(
        show_header=True,
        header_style="bold magenta",
        border_style="dim",
        pad_edge=False,
        expand=True,
    )
    tbl.add_column("prompt_id", style="bold white", no_wrap=True, overflow="fold")
    tbl.add_column("status", no_wrap=True)
    tbl.add_column("queue", no_wrap=True, justify="right")
    tbl.add_column("nodes", no_wrap=True, justify="right")
    tbl.add_column("outputs", no_wrap=True, justify="right")
    for r in rows:
        tbl.add_row(
            sanitize_markup(r.prompt_id[:8] + "…" if len(r.prompt_id) > 8 else r.prompt_id),
            status_glyph(r.status),
            str(r.queue_position) if r.queue_position is not None else "—",
            str(r.workflow_size) if r.workflow_size is not None else "—",
            str(r.outputs) if r.outputs else "—",
        )

    renderer.console().print(
        branded_panel(
            tbl,
            title="jobs",
            version=ConfigManager().get_cli_version(),
            where=where_label,
            host=host_label,
        )
    )


# ---------------------------------------------------------------------------
# `jobs status` — single prompt_id from /history (or /queue if still in flight)
# ---------------------------------------------------------------------------


def _state_file_snapshot(st: JobState, *, prompt_id: str, host: str, port: int, server_running: bool) -> dict:
    """Shape a `jobs status` payload out of the on-disk state file.

    Used by both fallback paths — the server being down, and a live server
    that has no record of the prompt — so the two agree on field names.
    ``server_running`` is the one thing that differs between them, and it is
    what tells the caller which fallback it is looking at.
    """
    return {
        "prompt_id": prompt_id,
        "status": st.status,
        # `jobs_state.read` drops unknown keys but does not type-check the ones
        # it keeps, so a hand-edited or truncated file can carry a non-list
        # `outputs`. `list()` would shred a str into characters and raise on a
        # scalar — only trust an actual list.
        "outputs": list(st.outputs) if isinstance(st.outputs, list) else [],
        # Every live `_snapshot()` result carries these three, so a consumer
        # that indexes them on a `jobs status` success payload would hit a
        # `KeyError` on this source alone. The state file records output URLs
        # flat, with no node or item association, so the grouped views cannot
        # be reconstructed from it — they are present but empty, which is the
        # same thing the live queue-hit payload emits.
        "outputs_by_node": {},
        "outputs_by_item": {},
        "workflow_size": None,
        "error": st.error,
        "host": host,
        "port": port,
        "server_running": server_running,
        "source": "state_file",
        "submitted_at": st.submitted_at,
        "updated_at": st.updated_at,
        "workflow": st.workflow,
    }


# Spellings that all name "the local machine". Folded together before a
# state file's recorded host is compared with the queried one, because the two
# are written by different code paths that do NOT agree on spelling:
#
#   * `comfy run`'s `execute()` substitutes the wildcard bind `0.0.0.0` with
#     `127.0.0.1` before it writes the state file, while `resolve_host_port`
#     canonicalizes a wildcard only when it came from `config.background` — an
#     explicit `--host 0.0.0.0`, or a `COMFY_LOCAL_URL` naming it, reaches
#     `jobs status` verbatim. Same env var, same server, two spellings.
#   * `localhost` vs `127.0.0.1` is just which flag the caller happened to type;
#     `resolve_host_port` passes both through unchanged.
#
# A missed match here is a silent FALSE NEGATIVE — the record is discarded and
# the command falls back to the bare `prompt_not_found` envelope, throwing away
# the very `server_died` attribution this fallback exists to preserve.
_LOOPBACK_HOST_SPELLINGS = frozenset({"127.0.0.1", "localhost", "::1", "0.0.0.0", "::"})


def _canonical_local_host(host: str) -> str:
    """Fold one host spelling into a form comparable for same-server checks.

    Unbrackets IPv6 literals (``[::1]`` and ``::1`` are the same address —
    brackets are a URL encoding, and `Target.host` stores the raw literal while
    `resolve_host_port` returns the bracketed one), lowercases (hostnames are
    case-insensitive), and collapses every loopback/wildcard spelling onto one.
    Anything else is returned as-is, so a genuinely different host still fails
    the comparison.
    """
    from comfy_cli.env_checker import _unbracket_host

    h = _unbracket_host(str(host).strip()).lower()
    return "127.0.0.1" if h in _LOOPBACK_HOST_SPELLINGS else h


def _state_file_for_local_target(prompt_id: str, *, host: str, port: int) -> JobState | None:
    """Read `prompt_id`'s state file, but only if it answers for THIS local target.

    The file is keyed by prompt_id alone, so an unscoped read will happily
    return a cloud run, or a job from a second local instance on another port,
    and `_state_file_snapshot` would then stamp the *queried* host/port onto
    output URLs belonging to a different server. Everything below is a reason
    the record is not an answer about ``host:port``.
    """
    from comfy_cli import jobs_state

    try:
        st = jobs_state.read(prompt_id)
    except ValueError:  # unsafe prompt_id — no state file to read
        return None
    except OSError:
        # `read` goes through `state_path` -> `state_dir`, which mkdirs the
        # config root and can fail (read-only or permission-denied home). A
        # traceback here would replace a clean envelope, so treat it as absent
        # — the same guard `_gather_waitable_ids` puts on this call.
        return None
    if st is None:
        return None
    # `state_path` maps "/" and "\" to "_" *before* validating, so read("a/b")
    # resolves to the file for the distinct prompt "a_b". Require the record to
    # name the id we actually asked about.
    if st.prompt_id != prompt_id:
        return None
    if st.where != "local":
        return None
    # host/port are None on files written before they were recorded, so only a
    # positive mismatch disqualifies a record. Compare canonicalized spellings:
    # a literal `!=` rejects `localhost` against `127.0.0.1` (see
    # `_canonical_local_host`), which would silently break this whole fallback.
    if st.host is not None and _canonical_local_host(st.host) != _canonical_local_host(host):
        return None
    if st.port is not None and str(st.port) != str(port):
        return None
    return st


def _hint_for_missing_local(prompt_id: str, default: str) -> str:
    """Redirect to `--where cloud` when that is why the local lookup came up empty.

    ``_state_file_for_local_target`` rejects a cloud record silently, which
    leaves the commonest mistake — a cloud job asked about without
    ``--where cloud`` — indistinguishable from a job that never existed. The
    default hints both point at ``comfy jobs ls``, whose scope follows the same
    resolved target, so it would not list that job either. Name the query that
    does work instead.
    """
    from comfy_cli import jobs_state

    try:
        st = jobs_state.read(prompt_id)
    except (ValueError, OSError):
        return default
    if st is None or st.prompt_id != prompt_id or st.where != "cloud":
        return default
    return f"this prompt_id is tracked as a cloud job — try: comfy jobs status {prompt_id} --where cloud"


def _server_confirms_no_record(host: str, port: int, prompt_id: str) -> bool:
    """True only if both `/queue` and `/history` answered and neither knows `prompt_id`.

    `_snapshot` returns None for two very different things: the server has no
    record, or the fetch failed (every `_http_get_json` failure is a
    `RuntimeError`, and `_snapshot` swallows it). Only the first licenses the
    "this job died with an earlier process" inference — reading a stale verdict
    out of a busy or briefly unreachable server would manufacture a
    `server_died` report for a job that is still running fine. The watcher is
    equally careful here (it keeps a grace window before drawing the same
    conclusion), so this demands a positive confirmation rather than trusting
    an absence of evidence.
    """
    try:
        q = _http_get_json(f"http://{host}:{port}/queue")
        hist = _http_get_json(f"http://{host}:{port}/history/{prompt_id}")
    except RuntimeError:
        return False
    if not isinstance(q, dict) or not isinstance(hist, dict):
        return False
    if prompt_id in hist:
        return False
    for key in ("queue_running", "queue_pending"):
        for entry in q.get(key) or []:
            pid, _wf = _safe_queue_entry(entry)
            if pid == prompt_id:
                return False
    return True


@app.command("status", help="Show the status of a single prompt_id (local or --where cloud).")
@tracking.track_command("jobs")
def status_cmd(
    prompt_id: Annotated[str, typer.Argument(help="The prompt_id returned by `comfy run`.")],
    host: Annotated[str | None, typer.Option()] = None,
    port: Annotated[int | None, typer.Option()] = None,
    where: Annotated[
        str | None,
        typer.Option("--where", help="'local' (default) or 'cloud'."),
    ] = None,
):
    renderer = get_renderer()
    if _is_cloud(where):
        return _cloud_status(prompt_id)

    h, p = _resolve_host_port(host, port)
    if not _server_or_error(h, p, raise_on_missing=False):
        # Server is down. The on-disk state file (written by `comfy run` and
        # maintained by the async watcher) still knows what this prompt was
        # doing when the server was last seen — a bare `server_not_running`
        # throws that attribution away.
        st = _state_file_for_local_target(prompt_id, host=h, port=p)

        if st is None:
            # Untracked prompt: same envelope as before, byte for byte.
            renderer.error(
                code="server_not_running",
                message=f"ComfyUI not running on {h}:{p}",
                hint=_hint_for_missing_local(prompt_id, "run: comfy launch"),
                details={"host": h, "port": p},
            )
            raise typer.Exit(code=1)

        if st.is_terminal:
            # The job finished before the server stopped — the state file is
            # the authoritative record, so this is a normal result, not an
            # error. Callers branch on `status`/`error`.
            snapshot = _state_file_snapshot(st, prompt_id=prompt_id, host=h, port=p, server_running=False)
            if renderer.is_pretty():
                _render_status_pretty(snapshot, host=h, port=p)
            renderer.emit(snapshot, command="jobs status")
            return

        # Non-terminal: the job was queued/running when the server was last
        # seen, so the server most likely died underneath it. Keep the
        # `server_not_running` code (callers key on it) and attribute.
        renderer.error(
            code="server_not_running",
            message=(
                f"ComfyUI not running on {h}:{p} — job {prompt_id} was {st.status!r} when the server "
                f"was last seen (submitted {st.submitted_at}, last update {st.updated_at}). The server "
                f"may have died while executing it (e.g. killed by the OS on an out-of-memory allocation)."
            ),
            hint="run: comfy launch — then check `comfy jobs ls` for the job's last recorded state",
            details={
                "host": h,
                "port": p,
                "prompt_id": prompt_id,
                "last_known_status": st.status,
                "submitted_at": st.submitted_at,
                "updated_at": st.updated_at,
                "workflow": st.workflow,
            },
        )
        raise typer.Exit(code=1)

    snapshot = _snapshot(h, p, prompt_id)
    if snapshot is None:
        # The server answered, but neither /queue nor /history knows this
        # prompt. That is *not* only "pruned from /history": the documented
        # recovery from a server death is `comfy launch` and then check, and a
        # relaunched ComfyUI is a FRESH process — its empty /queue and /history
        # are precisely what a job that died with the old process looks like.
        # So the state file, which still holds the verdict the watcher wrote
        # (e.g. error.code == "server_died"), is the better answer here.
        #
        # This is the same inference the async watcher already makes: see
        # `_LOST_AFTER_RESTART_S` in `comfy_cli/command/job_watcher.py`, where a
        # prompt missing from a server that came back is finalized as
        # `server_died`. `jobs status` reading the file it wrote keeps the two
        # in agreement rather than having them contradict each other.
        st = _state_file_for_local_target(prompt_id, host=h, port=p)

        # `_snapshot` returning None is not by itself proof the server has no
        # record — it swallows fetch failures too. Confirm before inferring.
        confirmed_absent = _server_confirms_no_record(h, p, prompt_id) if st is not None else False

        if st is not None and st.is_terminal and confirmed_absent:
            # The state file holds a final verdict and the server has positively
            # disowned the prompt — emit the verdict as a normal result, exactly
            # as the server-down path does. The one difference is
            # `server_running: True`, so a caller can tell it is looking at a
            # live server with no record rather than a dead one.
            snapshot = _state_file_snapshot(st, prompt_id=prompt_id, host=h, port=p, server_running=True)
            if renderer.is_pretty():
                _render_status_pretty(snapshot, host=h, port=p)
            renderer.emit(snapshot, command="jobs status")
            return

        if st is not None:
            # Either the record is non-terminal (the watcher never got to write
            # a verdict — it may still be inside its grace window, or it died
            # too), or the server would not confirm the absence. Keep the
            # `prompt_not_found` code — callers key on it — and attach what the
            # file does know, mirroring the server-down non-terminal branch.
            if confirmed_absent:
                tail = "The server may have been restarted since, in which case the job died with the previous process."
            else:
                # Don't assert a death the code has not established.
                tail = (
                    "The server did not answer /queue and /history reliably, so whether it still has a "
                    "record of this job is unknown — retry before treating this as the job's outcome."
                )
            renderer.error(
                code="prompt_not_found",
                message=(
                    f"No prompt with id {prompt_id!r} on {h}:{p} — the local state file last recorded it as "
                    f"{st.status!r} (submitted {st.submitted_at}, last update {st.updated_at}). {tail}"
                ),
                hint="check `comfy jobs ls`; very old prompts may have been pruned from /history",
                details={
                    "prompt_id": prompt_id,
                    "host": h,
                    "port": p,
                    "last_known_status": st.status,
                    "submitted_at": st.submitted_at,
                    "updated_at": st.updated_at,
                    "workflow": st.workflow,
                    "server_confirmed_no_record": confirmed_absent,
                },
            )
            raise typer.Exit(code=1)

        # Untracked prompt: same envelope as before, byte for byte.
        renderer.error(
            code="prompt_not_found",
            message=f"No prompt with id {prompt_id!r} on {h}:{p}.",
            hint=_hint_for_missing_local(
                prompt_id, "check `comfy jobs ls`; very old prompts may have been pruned from /history"
            ),
            details={"prompt_id": prompt_id, "host": h, "port": p},
        )
        raise typer.Exit(code=1)

    if renderer.is_pretty():
        _render_status_pretty(snapshot, host=h, port=p)
    renderer.emit(snapshot, command="jobs status")


def _snapshot(host: str, port: int, prompt_id: str) -> dict | None:
    # First: is it in the queue?
    try:
        q = _http_get_json(f"http://{host}:{port}/queue")
    except RuntimeError:
        q = {}
    for state, key in (("running", "queue_running"), ("pending", "queue_pending")):
        for entry in q.get(key) or []:
            pid, wf = _safe_queue_entry(entry)
            if pid == prompt_id:
                return {
                    "prompt_id": prompt_id,
                    "status": state,
                    "workflow_size": len(wf) if isinstance(wf, dict) else None,
                    "outputs": [],
                    "outputs_by_node": {},
                    "outputs_by_item": {},
                    "text_outputs": {},
                    "host": host,
                    "port": port,
                }

    # Then: history.
    try:
        h = _http_get_json(f"http://{host}:{port}/history/{prompt_id}")
    except RuntimeError:
        return None
    if not isinstance(h, dict) or prompt_id not in h:
        return None
    body = h[prompt_id]
    if not isinstance(body, dict):
        return None
    status_obj = body.get("status") or {}
    completed = bool(status_obj.get("completed"))
    error_detail = None
    interrupted = False
    for msg in status_obj.get("messages") or []:
        if isinstance(msg, list) and msg:
            if msg[0] == "execution_error":
                error_detail = msg[1] if len(msg) > 1 else None
            elif msg[0] == "execution_interrupted":
                interrupted = True
    # Flatten the node-keyed /history outputs into URL entries that keep
    # their producing-node association — same flatten the cloud snapshot
    # uses, so the grouped keys match the cloud envelope shape exactly.
    from comfy_cli import jobs_state
    from comfy_cli.comfy_client import _group_outputs, extract_output_entries, extract_text_outputs

    node_outputs: list[dict] = []
    for entry in extract_output_entries(body):
        q = urllib.parse.urlencode({k: entry[k] for k in ("filename", "subfolder", "type")})
        node_outputs.append({**entry, "url": f"http://{host}:{port}/view?{q}"})
    output_urls = [o["url"] for o in node_outputs]
    # The compose item_map (foreach item -> node ids) lives on the job state
    # file, written at submit time by `comfy run`.
    try:
        job = jobs_state.read(prompt_id)
    except ValueError:  # unsafe prompt_id — no state file to join against
        job = None
    item_map = job.item_map if job is not None else None
    outputs_by_node, outputs_by_item = _group_outputs(node_outputs, item_map)

    return {
        "prompt_id": prompt_id,
        "status": ("error" if error_detail else "completed" if completed else "cancelled" if interrupted else "queued"),
        "workflow_size": None,
        "outputs": output_urls,
        "outputs_by_node": outputs_by_node,
        "outputs_by_item": outputs_by_item,
        # Text/STRING node outputs (image descriptions, ShowText, …) live under
        # outputs[node]["text"] as bare strings, which the URL flatten drops.
        # Additive key: full untruncated strings for `--json`; the pretty
        # renderer previews them. Empty {} when the run emitted no text.
        "text_outputs": extract_text_outputs(body),
        "error": error_detail,
        "host": host,
        "port": port,
    }


_TEXT_PREVIEW_LIMIT = 20


def _render_status_pretty(snap: dict, *, host: str, port: int) -> None:
    from rich.console import Group
    from rich.panel import Panel
    from rich.table import Table
    from rich.text import Text

    renderer = get_renderer()
    status = snap["status"]
    # An unrecognized status falls through to the server's own string. `Text`
    # declines to *parse* markup, but it still forwards `\x1b` — rich's
    # `strip_control_codes` covers only BEL/BS/VT/FF/CR — so the escape bytes
    # need the plain `sanitize`. Not `sanitize_markup`: a `Text` would print its
    # backslashes verbatim, which is why the cell is control-stripped, not
    # markup-escaped.
    badge = {
        "running": Text.assemble(("● ", "bold green"), ("running", "bold green")),
        "pending": Text.assemble(("◌ ", "bold yellow"), ("pending", "bold yellow")),
        "completed": Text.assemble(("✓ ", "bold green"), ("completed", "bold green")),
        "queued": Text.assemble(("◌ ", "dim"), ("queued", "dim")),
        "error": Text.assemble(("✗ ", "bold red"), ("error", "bold red")),
    }.get(status, Text(sanitize(str(status))))

    tbl = Table.grid(padding=(0, 2), expand=False)
    tbl.add_column(justify="right", style="dim", no_wrap=True)
    tbl.add_column(overflow="fold")
    # `Table.add_row` parses markup in a `str` cell; every value below is
    # chosen by the host answering `/queue` and `/history`. `badge` is the one
    # exception — a `Text`, already control-stripped above.
    tbl.add_row("prompt_id", sanitize_markup(snap["prompt_id"]))
    tbl.add_row("status", badge)
    if snap.get("outputs"):
        tbl.add_row("outputs", "\n".join(sanitize_markup(o) for o in snap["outputs"]))
    if snap.get("text_outputs"):
        # Bounded preview: first non-blank line, ~120 chars per entry, capped at
        # _TEXT_PREVIEW_LIMIT entries total. The full untruncated text ships on
        # the `--json` path (renderer.emit). Built as Text (not markup strings)
        # since node ids / node text are server-supplied and may contain `[...]`
        # that Rich would otherwise interpret as style markup.
        preview_lines: list[Text] = []
        total = sum(len(texts) for texts in snap["text_outputs"].values())
        for node_id, texts in snap["text_outputs"].items():
            for text in texts:
                if len(preview_lines) >= _TEXT_PREVIEW_LIMIT:
                    break
                stripped = str(text).strip()
                first = stripped.splitlines()[0] if stripped else ""
                if len(first) > 120:
                    first = first[:117] + "…"
                preview_lines.append(Text(f"[{node_id}] {first}"))
            if len(preview_lines) >= _TEXT_PREVIEW_LIMIT:
                break
        if total > len(preview_lines):
            preview_lines.append(Text(f"… ({total - len(preview_lines)} more)", style="dim"))
        if preview_lines:
            tbl.add_row("text", Text("\n").join(preview_lines))
    if snap.get("error"):
        # Truncate first, escape second: the 600-char budget stays a budget on
        # the server's text rather than on the backslashes we add to it, and
        # escaping last is what guarantees no half-written tag survives the cut.
        tbl.add_row("error", sanitize_markup(str(snap["error"])[:600]))

    renderer.console().print(
        Panel(
            Group(tbl),
            title=Text(f"job on {host}:{port}", style="bold cyan"),
            title_align="left",
            border_style="cyan",
            padding=(0, 1),
        )
    )


# ---------------------------------------------------------------------------
# `jobs wait` — block until N prompt_ids are all terminal (multi-job wait)
# ---------------------------------------------------------------------------


def _wait_fetch_snapshot(
    prompt_id: str, *, cloud: bool, host: str | None, port: int | None, server_up: bool
) -> dict | None:
    """Best-effort single-job status snapshot for the wait loop.

    cloud -> /api/jobs/<id>; local -> live /history when the server is up,
    else fall back to the on-disk state file the async watcher maintains.
    """
    if cloud:
        return _cloud_status_snapshot(prompt_id)
    if server_up and host is not None and port is not None:
        snap = _snapshot(host, port, prompt_id)
        if snap is not None:
            return snap
    from comfy_cli import jobs_state

    try:
        st = jobs_state.read(prompt_id)
    except ValueError:
        st = None
    if st is None:
        return None
    err_msg = st.error.get("message") if isinstance(st.error, dict) else None
    return {
        "prompt_id": prompt_id,
        "status": st.status,
        "outputs": list(st.outputs or []),
        "error_message": err_msg,
    }


def _wait_loop(prompt_ids, fetch, *, poll_interval: float, deadline: float, renderer):
    """Poll ``fetch(pid)`` for each id until all are terminal or the deadline
    passes. Emits a ``settled`` NDJSON event as each job finishes. Returns
    ``(snapshots, still_pending)``.
    """
    from comfy_cli import cancellation, jobs_state

    pending = list(prompt_ids)
    snapshots: dict[str, dict] = {}
    total = len(pending)
    cancel_token = cancellation.get_token()
    while pending:
        still: list[str] = []
        for pid in pending:
            snap = fetch(pid)
            status = (snap or {}).get("status")
            if status in jobs_state.TERMINAL_STATUSES:
                snapshots[pid] = snap if isinstance(snap, dict) else {"prompt_id": pid, "status": status}
                renderer.event("settled", prompt_id=pid, status=status, settled=len(snapshots), total=total)
            else:
                still.append(pid)
        pending = still
        if not pending:
            break
        if time.time() >= deadline or (cancel_token is not None and cancel_token.is_set()):
            break
        time.sleep(max(0.0, min(poll_interval, deadline - time.time())))
    return snapshots, pending


def _gather_waitable_ids(cloud: bool) -> list[str]:
    """Every non-terminal locally-tracked prompt_id whose ``where`` matches routing."""
    from comfy_cli import jobs_state

    want = "cloud" if cloud else "local"
    out: list[str] = []
    try:
        paths = sorted(jobs_state.state_dir().glob("*.json"))
    except OSError:
        return out
    for path in paths:
        st = jobs_state.read(path.stem)
        if st is None or st.where != want or st.is_terminal:
            continue
        out.append(st.prompt_id)
    return out


def _render_wait_pretty(summary: dict) -> None:
    from rich.table import Table
    from rich.text import Text

    badge = {
        "completed": ("✓", "bold green"),
        "error": ("✗", "bold red"),
        "cancelled": ("⊘", "bold yellow"),
        "timed_out": ("⏱", "bold yellow"),
    }
    tbl = Table(
        title=f"waited on {summary['total']} job(s) — {summary['elapsed_seconds']}s",
        border_style="cyan",
        show_header=True,
    )
    tbl.add_column("prompt_id", style="dim", no_wrap=True)
    tbl.add_column("status")
    for r in summary["jobs"]:
        glyph, style = badge.get(r["status"], ("•", "white"))
        # The status cell is a `Text`, which never parses markup — so it takes
        # the plain `sanitize` (escape bytes still pass through `Text`) rather
        # than `sanitize_markup`, whose backslashes it would print verbatim.
        tbl.add_row(
            sanitize_markup(r["prompt_id"]),
            Text(f"{glyph} {sanitize(str(r['status']))}", style=style),
        )
    get_renderer().console().print(tbl)


@app.command("wait", help="Block until ALL given prompt_ids reach a terminal state; emit a summary.")
@tracking.track_command("jobs")
def wait_cmd(
    prompt_ids: Annotated[
        list[str] | None,
        typer.Argument(help="prompt_ids to wait on (omit and use --all to wait on every tracked job)."),
    ] = None,
    host: Annotated[str | None, typer.Option()] = None,
    port: Annotated[int | None, typer.Option()] = None,
    where: Annotated[str | None, typer.Option("--where", help="'local' (default) or 'cloud'.")] = None,
    poll_interval: Annotated[
        float,
        typer.Option("--poll-interval", help="Seconds between status polls (these are long jobs; don't hammer)."),
    ] = 5.0,
    timeout: Annotated[float, typer.Option("--timeout", help="Give up after this many seconds total.")] = 1800.0,
    wait_all: Annotated[bool, typer.Option("--all", help="Wait on all locally-tracked non-terminal jobs.")] = False,
):
    renderer = get_renderer()
    cloud = _is_cloud(where)

    ids = list(prompt_ids or [])
    if wait_all:
        ids.extend(_gather_waitable_ids(cloud))
    ids = list(dict.fromkeys(ids))  # de-dup, preserve order
    if not ids:
        renderer.error(
            code="no_prompt_ids",
            message="no prompt_ids to wait on",
            hint="pass one or more prompt_ids, or --all to wait on every tracked job",
        )
        raise typer.Exit(code=2)

    h: str | None = None
    p: int | None = None
    server_up = False
    if cloud:
        cloud_preflight_or_exit()
    else:
        h, p = _resolve_host_port(host, port)
        server_up = _server_or_error(h, p, raise_on_missing=False)

    start = time.time()
    deadline = start + timeout

    def fetch(pid: str) -> dict | None:
        return _wait_fetch_snapshot(pid, cloud=cloud, host=h, port=p, server_up=server_up)

    if renderer.is_pretty():
        renderer.console().print(
            f"[bold]Waiting on {len(ids)} job(s)[/bold] [dim](poll {poll_interval}s, timeout {timeout:.0f}s)[/dim]"
        )

    snapshots, pending = _wait_loop(ids, fetch, poll_interval=poll_interval, deadline=deadline, renderer=renderer)

    jobs_list: list[dict] = []
    for pid in ids:
        if pid in pending:
            jobs_list.append({"prompt_id": pid, "status": "timed_out", "ok": False})
            continue
        snap = snapshots.get(pid) or {"prompt_id": pid, "status": "unknown"}
        status = str(snap.get("status") or "unknown")
        row: dict = {"prompt_id": pid, "status": status, "ok": status == "completed"}
        if snap.get("outputs"):
            row["outputs"] = snap["outputs"]
        if snap.get("error_message"):
            row["error_message"] = snap["error_message"]
        jobs_list.append(row)

    completed = sum(1 for r in jobs_list if r["status"] == "completed")
    failed = sum(1 for r in jobs_list if r["status"] == "error")
    cancelled = sum(1 for r in jobs_list if r["status"] == "cancelled")
    timed_out = sum(1 for r in jobs_list if r["status"] == "timed_out")
    summary = {
        "total": len(ids),
        "completed": completed,
        "failed": failed,
        "cancelled": cancelled,
        "timed_out": timed_out,
        "elapsed_seconds": round(time.time() - start, 2),
        "jobs": jobs_list,
    }
    where_label = "cloud" if cloud else "local"

    if renderer.is_pretty():
        _render_wait_pretty(summary)

    if failed == 0 and cancelled == 0 and timed_out == 0:
        renderer.emit(summary, command="jobs wait", where=where_label)
        return

    # Literal codes (not a variable) so the error-code registry ratchet can
    # AST-scan them. execution_error/cancelled are already registered; wait_timeout
    # is registered alongside no_prompt_ids in comfy_cli/error_codes.py.
    msg = f"{completed}/{len(ids)} completed — {failed} failed, {cancelled} cancelled, {timed_out} timed out"
    if failed:
        renderer.error(code="execution_error", message=msg, details=summary, exit_code=1, command="jobs wait")
        raise typer.Exit(code=1)
    if cancelled:
        renderer.error(code="cancelled", message=msg, details=summary, exit_code=130, command="jobs wait")
        raise typer.Exit(code=130)
    renderer.error(
        code="wait_timeout",
        message=msg,
        hint="the jobs may still be running — raise `--timeout`, or check `comfy jobs status <id>`",
        details=summary,
        exit_code=1,
        command="jobs wait",
    )
    raise typer.Exit(code=1)


# ---------------------------------------------------------------------------
# `jobs cancel` — stop a running or pending prompt, locally or on cloud
# ---------------------------------------------------------------------------


@app.command(
    "cancel",
    help="Cancel a job. Idempotent for known jobs; unknown ids error with prompt_not_found.",
)
@tracking.track_command("jobs")
def cancel_cmd(
    prompt_id: Annotated[str, typer.Argument(help="The prompt_id to cancel.")],
    host: Annotated[str | None, typer.Option()] = None,
    port: Annotated[int | None, typer.Option()] = None,
    where: Annotated[
        str | None,
        typer.Option("--where", help="'local' (default) or 'cloud'."),
    ] = None,
):
    if _is_cloud(where):
        return _cloud_cancel(prompt_id)
    h, p = _resolve_host_port(host, port)
    _server_or_error(h, p)
    return _local_cancel(prompt_id, h, p)


def _local_cancel(prompt_id: str, host: str, port: int) -> None:
    """Cancel a local prompt by removing it from the pending queue AND
    interrupting any in-flight execution. ComfyUI splits these into two
    endpoints; we hit both so the call works regardless of phase.

    Idempotent for prompts we can prove exist — running, pending, in the
    server's history, or in the local state store — including already-terminal
    ones, which return ok. An id that is nowhere is a `prompt_not_found` error
    (exit 1), matching what the cloud path does with a 404: ``POST /queue
    {"delete": [id]}`` 200s for unknown ids, so without the probe below a
    typo'd id is indistinguishable from a real cancel.
    """
    renderer = get_renderer()
    base = f"http://{host}:{port}"
    from comfy_cli import jobs_state

    # 0. An empty/whitespace id can never name a real prompt, and it would turn
    #    the /history/<id> probe below into `GET /history/` — the list-ALL
    #    endpoint, whose non-empty body would read as "found". Reject it up
    #    front, before any probe or mutation.
    if not prompt_id.strip():
        renderer.error(
            code="prompt_not_found",
            message="prompt id must be a non-empty string",
            hint="check `comfy jobs ls`",
            details={"prompt_id": prompt_id, "host": host, "port": port},
        )
        raise typer.Exit(code=1)

    # 1. Existence probe, BEFORE mutating anything — the queue delete in step 2
    #    would erase the only evidence that a pending prompt ever existed.
    try:
        queue = _http_get_json(f"{base}/queue")
        queue_reachable_pre = True
    except RuntimeError:
        queue = {}
        queue_reachable_pre = False
    if not isinstance(queue, dict):
        queue = {}
    running_ids = {str(_safe_queue_entry(entry)[0]) for entry in (queue.get("queue_running") or [])}
    pending_ids = {str(_safe_queue_entry(entry)[0]) for entry in (queue.get("queue_pending") or [])}

    # A state file means WE submitted it, so it existed even if the server has
    # since forgotten it (restart, history trimmed). Read for existence only —
    # the status write at the end re-reads, so a concurrent `jobs watch` update
    # in the meantime isn't clobbered.
    found = prompt_id in running_ids or prompt_id in pending_ids or jobs_state.read(prompt_id) is not None
    probes_reachable = queue_reachable_pre
    if not found and queue_reachable_pre:
        # /history/<id> is `{}` for an unknown id, and a dict keyed by the id
        # for a known one (running, completed, errored, or cancelled). Quote the
        # id into the path so a hostile value can't escape the segment — same
        # defense in depth as the cloud path.
        try:
            history = _http_get_json(f"{base}/history/{urllib.parse.quote(prompt_id, safe='')}")
            found = isinstance(history, dict) and bool(history)
        except RuntimeError:
            # Unreachable is not "absent" — absence of evidence isn't evidence
            # of absence, so fall through to the idempotent path instead.
            probes_reachable = False

    if not found and probes_reachable:
        renderer.error(
            code="prompt_not_found",
            message=f"no local job with id {prompt_id!r}",
            hint="check `comfy jobs ls`",
            details={"prompt_id": prompt_id, "host": host, "port": port},
        )
        raise typer.Exit(code=1)

    # 2. Remove from the pending queue (no-op if not pending).
    queue_body = json.dumps({"delete": [prompt_id]}).encode("utf-8")
    queue_req = urllib.request.Request(
        f"{base}/queue", data=queue_body, method="POST", headers={"Content-Type": "application/json"}
    )
    queue_ok = True
    try:
        with plain_urlopen(queue_req, timeout=10) as resp:
            _ = resp.read()
    except (urllib.error.HTTPError, urllib.error.URLError, OSError):
        # Server refused the delete; common when the prompt isn't in queue.
        # Don't fail the whole command — try the interrupt next.
        queue_ok = False

    # 3. Interrupt only if THIS prompt is the one currently executing.
    #    /interrupt takes NO prompt_id — it kills whatever is running — so
    #    blindly posting it after a pending-job delete would also abort an
    #    unrelated running job ("cancel B" silently cancelling A). Gate on a
    #    FRESH read of /queue's queue_running: the step-1 snapshot predates the
    #    delete round-trip, and in that window our prompt can go pending→running
    #    (missing it = silent cancel failure) or a different prompt can take
    #    over the running slot (interrupting it = cancelling A instead of B).
    try:
        queue_now = _http_get_json(f"{base}/queue")
        queue_reachable = True
        if not isinstance(queue_now, dict):
            queue_now = {}
        is_running = prompt_id in {str(_safe_queue_entry(entry)[0]) for entry in (queue_now.get("queue_running") or [])}
    except RuntimeError:
        # Can't confirm; fall back to the step-1 snapshot. The server is very
        # likely down, in which case /interrupt fails harmlessly too — better
        # a best-effort interrupt than a silently skipped cancel.
        queue_reachable = False
        is_running = prompt_id in running_ids

    interrupt_ok = True
    if is_running:
        interrupt_req = urllib.request.Request(f"{base}/interrupt", method="POST")
        try:
            with plain_urlopen(interrupt_req, timeout=10) as resp:
                _ = resp.read()
        except (urllib.error.HTTPError, urllib.error.URLError, OSError):
            interrupt_ok = False

    if not queue_ok and not queue_reachable:
        renderer.error(
            code="cancel_failed",
            message=f"both /queue delete and /queue status failed on {host}:{port}",
            hint="check the server is still reachable",
            details={"host": host, "port": port, "prompt_id": prompt_id},
        )
        raise typer.Exit(code=1)

    payload = {
        "prompt_id": prompt_id,
        "where": "local",
        "host": host,
        "port": port,
        "found": found,
        "queue_delete_ok": queue_ok,
        "interrupt_ok": interrupt_ok,
    }

    # Re-read right before the write: the step-1 read predates three network
    # round-trips, and a concurrent `jobs watch` may have recorded newer
    # status/outputs in the meantime (the per-file lock stops torn writes, not
    # stale overwrites). Already-terminal jobs keep their recorded outcome —
    # cancelling a finished job is an idempotent ok, not a re-labelling of a
    # 'completed' run as 'cancelled'.
    existing = jobs_state.read(prompt_id)
    if existing is not None and not existing.is_terminal:
        existing.status = "cancelled"
        jobs_state.write(existing)

    if renderer.is_pretty():
        from rich.text import Text

        msg = Text.from_markup(f"  [bold green]✓[/bold green]  cancel sent for [cyan]{prompt_id[:8]}…[/cyan]")
        renderer.console().print(msg)
    renderer.emit(payload, command="jobs cancel")


def _cloud_cancel(prompt_id: str) -> None:
    """Cancel a cloud job via ``POST /api/jobs/<id>/cancel`` — idempotent."""
    cloud_preflight_or_exit()
    renderer = get_renderer()

    from comfy_cli.target import resolve_target

    target = resolve_target(where="cloud")
    # Quote prompt_id into the path segment so a hostile/malformed value can't
    # escape (e.g. ``../foo`` → ``%2E%2E%2Ffoo``). Cloud rejects bad UUIDs
    # upstream too; encoding here is defense in depth.
    url = target.url("jobs", urllib.parse.quote(prompt_id, safe=""), "cancel")

    try:
        with authed_urlopen(url, target, method="POST", data=b"", timeout=15) as resp:
            body = resp.read()
    except (urllib.error.HTTPError, urllib.error.URLError, OSError) as e:
        from comfy_cli.command._cloud_errors import handle_cloud_http_error

        raise handle_cloud_http_error(
            renderer,
            e,
            operation="cancel",
            not_found_code="prompt_not_found",
            not_found_message=f"no cloud job with id {prompt_id!r}",
            not_found_hint="check `comfy jobs ls --where cloud`",
            id_label="prompt_id",
            resource_id=prompt_id,
        ) from e

    parsed: dict | None
    try:
        parsed = json.loads(body) if body else None
    except json.JSONDecodeError:
        parsed = None
    payload = {
        "prompt_id": prompt_id,
        "where": "cloud",
        "base_url": target.base_url,
        "response": parsed if isinstance(parsed, dict) else None,
    }
    if renderer.is_pretty():
        from rich.text import Text

        renderer.console().print(
            Text.from_markup(f"  [bold green]✓[/bold green]  cancel sent for [cyan]{prompt_id[:8]}…[/cyan]")
        )
    renderer.emit(payload, command="jobs cancel", where="cloud")


# ---------------------------------------------------------------------------
# `jobs watch` — tail WS events live, filtered on prompt_id
# ---------------------------------------------------------------------------


@dataclass
class _WatchState:
    """Loop-local state shared across the `jobs watch` WS recv loop.

    Holds both the immutable per-watch context (renderer, prompt_id, host,
    port) and the mutable accumulators the per-type handlers and the
    connect/timeout/cancel state machine both write to (completed_nodes,
    outputs, end_reason, end_details). ``terminal`` is the handlers' way of
    signalling the recv loop to break.
    """

    renderer: Any
    prompt_id: str
    host: str
    port: int
    completed_nodes: set[str] = field(default_factory=set)
    outputs: list[str] = field(default_factory=list)
    end_reason: str | None = None
    end_details: Any = None
    terminal: bool = False


def _watch_executing(state: _WatchState, data: dict[str, Any]) -> None:
    node = data.get("node")
    if node is None:
        # A null node marks the end of execution for the prompt.
        state.end_reason = "completed"
        state.terminal = True
        return
    renderer = state.renderer
    if renderer.is_pretty():
        # ``node`` is server-controlled; escape so it can't inject Rich markup.
        from rich.markup import escape

        renderer.console().print(f"[dim]→[/dim] executing node [bold]{escape(str(node))}[/bold]")
    renderer.event("executing", node=str(node), prompt_id=state.prompt_id)


def _watch_execution_cached(state: _WatchState, data: dict[str, Any]) -> None:
    nodes = data.get("nodes") or []
    for n in nodes:
        state.completed_nodes.add(str(n))
    renderer = state.renderer
    if renderer.is_pretty():
        renderer.console().print(f"[dim]✓[/dim] cached: {len(nodes)} node(s)")
    renderer.event(
        "execution_cached",
        nodes=[str(n) for n in nodes],
        prompt_id=state.prompt_id,
    )


def _watch_progress(state: _WatchState, data: dict[str, Any]) -> None:
    state.renderer.throttled_event(
        f"progress:{data.get('node')}",
        "progress",
        max_hz=10,
        node=str(data.get("node")),
        completed=data.get("value"),
        total=data.get("max"),
        prompt_id=state.prompt_id,
    )


def _watch_executed(state: _WatchState, data: dict[str, Any]) -> None:
    renderer = state.renderer
    node = str(data.get("node"))
    state.completed_nodes.add(node)
    output = data.get("output") or {}
    for key in ("images", "gifs", "videos", "audio", "files"):
        for item in output.get(key) or []:
            if isinstance(item, dict) and "filename" in item:
                q = urllib.parse.urlencode({k: item[k] for k in ("filename", "subfolder", "type") if k in item})
                url = f"http://{state.host}:{state.port}/view?{q}"
                state.outputs.append(url)
                if renderer.is_pretty():
                    renderer.console().print(f"[bold green]✓[/bold green] output: [cyan]{url}[/cyan]")
                renderer.event("output", url=url, prompt_id=state.prompt_id)
    renderer.event("executed", node=node, prompt_id=state.prompt_id)


def _watch_execution_error(state: _WatchState, data: dict[str, Any]) -> None:
    state.end_reason = "error"
    state.end_details = data
    state.terminal = True


# type → pure per-message handler. Each mutates ``state`` (and sets
# ``state.terminal`` for the two terminal events); the recv loop owns the break.
_WATCH_HANDLERS = {
    "executing": _watch_executing,
    "execution_cached": _watch_execution_cached,
    "progress": _watch_progress,
    "executed": _watch_executed,
    "execution_error": _watch_execution_error,
}


@app.command("watch", help="Tail live execution events for a prompt_id (WS local / polling cloud).")
@tracking.track_command("jobs")
def watch_cmd(
    prompt_id: Annotated[str, typer.Argument(help="The prompt_id returned by `comfy run`.")],
    host: Annotated[str | None, typer.Option()] = None,
    port: Annotated[int | None, typer.Option()] = None,
    timeout: Annotated[int, typer.Option(help="Per-recv (or per-poll) timeout in seconds.")] = 30,
    where: Annotated[
        str | None,
        typer.Option("--where", help="'local' (WebSocket) or 'cloud' (HTTP polling)."),
    ] = None,
    poll_interval: Annotated[
        float,
        typer.Option("--poll-interval", help="Cloud-only: seconds between status polls."),
    ] = 1.5,
    max_wait: Annotated[
        float,
        typer.Option("--max-wait", help="Cloud-only: give up after this many seconds total."),
    ] = 600.0,
):
    renderer = get_renderer()
    if _is_cloud(where):
        return _cloud_watch(prompt_id, poll_interval=poll_interval, max_wait=max_wait)

    h, p = _resolve_host_port(host, port)
    _server_or_error(h, p)

    # If the job already finished, just print status and return — there will
    # be no more WS events.
    snap = _snapshot(h, p, prompt_id)
    if snap and snap["status"] in {"completed", "error", "cancelled"}:
        if renderer.is_pretty():
            renderer.console().print(f"[dim]Prompt {prompt_id} already {snap['status']}; nothing more to watch.[/dim]")
            _render_status_pretty(snap, host=h, port=p)
        _emit_terminal(renderer, snap, command="jobs watch")
        return

    ws = WebSocket()
    client_id = str(uuid.uuid4())
    try:
        ws.connect(f"ws://{h}:{p}/ws?clientId={client_id}")
    except (WebSocketException, ConnectionError, OSError) as e:
        renderer.error(
            code="ws_disconnected",
            message=f"Could not open WebSocket: {e}",
            hint="check the server is reachable; try `comfy jobs status` instead",
        )
        raise typer.Exit(code=1)

    token = cancellation.get_token()
    token.on_cancel(lambda: _safe_close_ws(ws))

    ws.settimeout(timeout)

    state = _WatchState(renderer=renderer, prompt_id=prompt_id, host=h, port=p)
    saw_any_event = False
    missing_deadline: float | None = None
    start = time.time()

    if renderer.is_pretty():
        renderer.console().print(f"[bold]Watching prompt[/bold] {prompt_id} on {h}:{p}   [dim](Ctrl-C to stop)[/dim]")

    try:
        while True:
            try:
                raw = ws.recv()
            except WebSocketTimeoutException:
                # If the job moved to completed between recvs, exit cleanly.
                snap = _snapshot(h, p, prompt_id)
                if snap and snap["status"] in {"completed", "error", "cancelled"}:
                    state.end_reason = snap["status"]
                    state.end_details = snap
                    state.outputs.extend(snap.get("outputs") or [])
                    break
                # Bounded wait for an unknown prompt: if the server has never
                # heard of this prompt_id (no snapshot) and no events have
                # arrived, don't loop forever on a typoed/already-pruned id —
                # mirror the cloud path's deadline and surface prompt_not_found.
                if snap is None and not saw_any_event:
                    if missing_deadline is None:
                        missing_deadline = time.time() + max(timeout, 1)
                    elif time.time() >= missing_deadline:
                        # The enclosing `finally` closes the socket.
                        renderer.error(
                            code="prompt_not_found",
                            message=f"prompt {prompt_id} not found on {h}:{p}",
                            hint="check the prompt_id; it may be a typo or already pruned",
                            details={"prompt_id": prompt_id, "host": h, "port": p},
                        )
                        raise typer.Exit(code=1)
                else:
                    missing_deadline = None
                continue
            except (WebSocketException, ConnectionError, OSError) as e:
                # Cancellation closes the socket out from under recv(). Check
                # the token before classifying as "server disconnected".
                if token.is_set():
                    state.end_reason = "cancelled"
                    break
                renderer.error(
                    code="ws_disconnected",
                    message=f"Lost connection while watching {prompt_id}: {e}",
                    hint="re-run `comfy jobs status` to check final state",
                )
                raise typer.Exit(code=1) from e
            if not isinstance(raw, str):
                continue
            try:
                msg = json.loads(raw)
            except json.JSONDecodeError:
                continue
            data = msg.get("data") or {}
            if data.get("prompt_id") != prompt_id:
                continue
            saw_any_event = True
            # ``type`` is server-controlled: a JSON array/object is unhashable
            # and would make dict.get() raise TypeError, so only dispatch on a
            # str key (unknown types fall through to be ignored, as before).
            mtype = msg.get("type")
            handler = _WATCH_HANDLERS.get(mtype) if isinstance(mtype, str) else None
            if handler is not None:
                handler(state, data)
                if state.terminal:
                    break
    finally:
        _safe_close_ws(ws)

    elapsed = time.time() - start
    final_status = state.end_reason or ("completed" if state.completed_nodes else "unknown")
    if renderer.is_pretty():
        from rich.text import Text

        if final_status == "completed":
            renderer.console().print(
                Text.assemble(("\n✓ ", "bold green"), ("completed", "bold green"), (f"  in {elapsed:.1f}s", "dim"))
            )
        elif final_status == "error":
            renderer.console().print(Text.assemble(("\n✗ ", "bold red"), ("error", "bold red")))
        elif final_status == "cancelled":
            renderer.console().print(Text.assemble(("\n⊘ ", "yellow"), ("cancelled", "yellow")))

    payload = {
        "prompt_id": prompt_id,
        "status": final_status,
        "outputs": state.outputs,
        "completed_nodes": sorted(state.completed_nodes),
        "elapsed_seconds": elapsed,
        "host": h,
        "port": p,
    }
    if state.end_details is not None:
        payload["details"] = state.end_details if isinstance(state.end_details, dict) else {"raw": state.end_details}
    if not saw_any_event and final_status == "unknown":
        payload["hint"] = "watch returned without events; the prompt may already have completed"
    _emit_terminal(renderer, payload, command="jobs watch")


# ---------------------------------------------------------------------------
# Cloud handlers — /api/jobs, /api/jobs/<id>, /api/history_v2/<id>
# ---------------------------------------------------------------------------


def _is_cloud(where: str | None) -> bool:
    """Resolve the routing target using the same precedence as the rest of
    the CLI: per-command ``--where`` flag > ``COMFY_WHERE`` env var >
    persisted ``where_default`` config > default ``local``.

    Honoring the env var matters because the top-level ``comfy --where
    cloud`` flag is sugar for ``COMFY_WHERE=cloud``: ``cmdline.py`` sets
    the env so every subcommand inherits the routing decision without
    repeating the flag. A previous implementation looked only at the
    per-command parameter, which silently dropped the top-level flag
    for ``jobs ls/status/watch``.
    """
    from comfy_cli import where as where_module

    try:
        decision = where_module.resolve_default(flag=where)
    except ValueError:
        # Invalid value — fall back to local; the validating command
        # (cmdline.py top-level option) will surface ``where_invalid``.
        return False
    return decision.target is where_module.WhereTarget.CLOUD


def _cloud_job_to_row(j: dict) -> JobRow:
    """Map a /api/jobs entry to our JobRow shape.

    Statuses go through the same map the watcher uses, so cloud's other failure
    spellings (``non_retryable_error``, ``lost``, ``canceled``) normalize to the
    row vocabulary instead of passing through raw. Beyond keeping the rendered
    status consistent with `jobs status`/`jobs watch`, it is what lets
    ``_merge_jobs`` recognize those rows as failures and keep the state file's
    ``error_code`` — the failures that most need a named cause.
    """
    raw_status = (j.get("status") or "").lower()
    status = _CLOUD_ROW_STATUS_MAP.get(raw_status, raw_status or "pending")
    outputs = int(j.get("outputs_count") or 0)
    return JobRow(
        prompt_id=str(j.get("id") or ""),
        status=status,
        queue_position=None,
        elapsed_seconds=None,
        workflow_size=None,
        outputs=outputs,
        # These rows come from the cloud job list; without this they'd take
        # JobRow's "local" default and supersede the state row that correctly
        # said "cloud", so `jobs ls --where cloud` would emit an envelope
        # saying cloud with every row inside claiming local.
        where="cloud",
    )


def _cloud_client():
    """Construct a unified Client targeting cloud. Raises if not signed in.

    Observer commands (status/ls/watch snapshots) must never clear the shared
    OAuth session on a fatal refresh error: batch workloads run dozens of these
    concurrently, and one spurious invalid_grant wiping the login mid-run turns
    a transient hiccup into a hard logout. Session lifecycle belongs to
    login/logout and the foreground submit path.
    """
    from comfy_cli.comfy_client import Client, Unauthenticated
    from comfy_cli.target import resolve_target

    target = resolve_target(where="cloud")
    try:
        return Client(target, clear_session_on_auth_failure=False)
    except Unauthenticated as e:
        renderer = get_renderer()
        renderer.error(code="cloud_unauthorized", message=str(e), hint="run: comfy cloud login")
        raise typer.Exit(code=1) from e


def _cloud_status_snapshot(prompt_id: str) -> dict | None:
    """Compose a cloud snapshot from /api/jobs/<id> + /api/history_v2/<id>."""
    from comfy_cli import jobs_state
    from comfy_cli.comfy_client import _group_outputs

    client = _cloud_client()
    status = client.get_job_status(prompt_id)
    if status is None:
        return None
    raw = (status.get("status") or "").lower()
    # Deliberately NOT _CLOUD_ROW_STATUS_MAP: this map lacks cloud's two cancel
    # spellings, so a cancelled cloud job snapshots as the raw `canceled` — not
    # in the published `status` enum, not in `_cloud_watch`'s terminal set (so
    # `jobs watch --where cloud` spins to `cloud_timeout`), and not in
    # `_TERMINAL_VERDICT` (so `jobs status` reports it ok:true/exit 0 instead of
    # the documented 130). Real bugs, but adding the aliases here changes an
    # exit code on a path this PR does not otherwise touch — see BE-6612.
    state = {
        "success": "completed",
        "completed": "completed",
        "failed": "error",
        "error": "error",
        "non_retryable_error": "error",
        "lost": "error",
    }.get(raw, raw or "pending")

    outputs: list[str] = []
    outputs_by_node: dict[str, list[str]] = {}
    outputs_by_item: dict[str, list[str]] = {}
    if state == "completed":
        record = client.get_history(prompt_id)
        if record:
            node_outputs = client.extract_outputs(record)
            outputs = [o["url"] for o in node_outputs]
            # The compose item_map (foreach item -> node ids) lives on the
            # job state file, written at submit time by `comfy run`.
            job = jobs_state.read(prompt_id)
            item_map = job.item_map if job is not None else None
            outputs_by_node, outputs_by_item = _group_outputs(node_outputs, item_map)

    return {
        "prompt_id": prompt_id,
        "status": state,
        "outputs": outputs,
        "outputs_by_node": outputs_by_node,
        "outputs_by_item": outputs_by_item,
        "assigned_inference": status.get("assigned_inference"),
        "error_message": status.get("error_message"),
        "created_at": status.get("created_at"),
        "updated_at": status.get("updated_at"),
        "base_url": client.target.base_url,
    }


def _cloud_status(prompt_id: str) -> None:
    cloud_preflight_or_exit()
    renderer = get_renderer()
    snap = _cloud_status_snapshot(prompt_id)
    if snap is None:
        renderer.error(
            code="prompt_not_found",
            message=f"No cloud prompt with id {prompt_id!r}.",
            hint="check `comfy jobs ls --where cloud`",
            details={"prompt_id": prompt_id},
        )
        raise typer.Exit(code=1)

    if renderer.is_pretty():
        from rich.table import Table

        # Rich parses a `str` table title as markup too, so the id belongs in
        # the same escaping regime as the cells — an unbalanced `[/]` there
        # raises `MarkupError` before a single row is added.
        tbl = Table(title=f"Cloud prompt {sanitize_markup(prompt_id[:8])}…", border_style="cyan", show_header=False)
        tbl.add_column(style="bold cyan")
        tbl.add_column()
        # Every cell below comes straight off `/api/jobs/<id>` — including
        # `status`, which falls through to the server's own vocabulary when it
        # is not one of the aliases `_cloud_status_snapshot` knows.
        tbl.add_row("status", sanitize_markup(snap["status"]))
        if snap.get("assigned_inference"):
            tbl.add_row("inference", sanitize_markup(snap["assigned_inference"]))
        if snap.get("created_at"):
            tbl.add_row("created", sanitize_markup(snap["created_at"]))
        if snap.get("updated_at"):
            tbl.add_row("updated", sanitize_markup(snap["updated_at"]))
        if snap.get("error_message"):
            # Same 600-char budget the local `/history` error cell uses, and for
            # the same reason: an unbounded API string becomes an unbounded Rich
            # cell. Truncate first, escape second (see `_render_status_pretty`).
            tbl.add_row("error", sanitize_markup(str(snap["error_message"])[:600]))
        for u in snap.get("outputs") or []:
            tbl.add_row("output", sanitize_markup(u))
        renderer.console().print(tbl)
    renderer.emit(snap, command="jobs status", where="cloud")


def _cloud_watch(prompt_id: str, *, poll_interval: float, max_wait: float) -> None:
    """Poll cloud's job status, emit NDJSON events on each transition."""
    cloud_preflight_or_exit()
    renderer = get_renderer()
    base_url = _cloud_client().target.base_url

    cancel_token = cancellation.get_token()
    deadline = time.time() + max_wait
    last_state: str | None = None
    start = time.time()

    if renderer.is_pretty():
        renderer.console().print(
            f"[bold]Watching cloud prompt[/bold] {sanitize_markup(prompt_id)}   "
            f"[dim]({sanitize_markup(base_url)}, Ctrl-C to stop)[/dim]"
        )

    final_snap: dict | None = None
    while not cancel_token.is_set():
        snap = _cloud_status_snapshot(prompt_id)
        if snap is None:
            # Not yet known to the cloud — wait briefly.
            if time.time() >= deadline:
                renderer.error(
                    code="prompt_not_found",
                    message=f"prompt {prompt_id} not found on cloud after {max_wait}s",
                    details={"prompt_id": prompt_id, "base_url": base_url},
                )
                raise typer.Exit(code=1)
            time.sleep(min(poll_interval, deadline - time.time()))
            continue

        if snap["status"] != last_state:
            last_state = snap["status"]
            if renderer.is_pretty():
                # `_cloud_status` hardens the one-shot view of this same
                # snapshot; the streaming sibling prints the same server-chosen
                # `status` into markup on every transition, and a `[/]` in it
                # would raise `MarkupError` mid-poll. The `renderer.event` line
                # below stays raw on purpose — JSON escapes it already.
                renderer.console().print(f"[dim]→[/dim] state [bold]{sanitize_markup(last_state)}[/bold]")
            renderer.event("state", prompt_id=prompt_id, status=last_state)

        if snap["status"] in {"completed", "error", "cancelled"}:
            for u in snap.get("outputs") or []:
                renderer.event("output", url=u, prompt_id=prompt_id)
                if renderer.is_pretty():
                    renderer.console().print(f"[bold green]✓[/bold green] output: [cyan]{sanitize_markup(u)}[/cyan]")
            final_snap = snap
            break

        if time.time() >= deadline:
            renderer.error(
                code="cloud_timeout",
                message=f"prompt {prompt_id} still {snap['status']} after {max_wait}s",
                hint="raise --max-wait or re-run with --where cloud",
                details=snap,
            )
            raise typer.Exit(code=1)

        time.sleep(min(poll_interval, deadline - time.time()))

    payload = final_snap or {"prompt_id": prompt_id, "status": "cancelled"}
    payload["elapsed_seconds"] = time.time() - start
    _emit_terminal(renderer, payload, command="jobs watch", where="cloud")


def _safe_close_ws(ws) -> None:
    try:
        ws.close()
    except Exception:  # noqa: BLE001
        pass
