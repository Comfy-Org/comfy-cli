"""Watch a v2 data-plane job without treating its live event stream as a log."""

from __future__ import annotations

import http.client
import json
import time
import urllib.error
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from typing import Final, Literal

from typing_extensions import assert_never

from comfy_cli.command.build_spec import JsonObject
from comfy_cli.deploy_api_errors import DeployAPIError, assert_safe_deploy_url
from comfy_cli.http import (
    ResponseTooLarge,
    build_authed_request,
    no_redirect_urlopen,
    read_capped,
    request_json,
)
from comfy_cli.target import Target

# Preview frames contain base64 JPEGs, so the frame cap is deliberately larger
# than ordinary JSON while still preventing an unbounded allocation.
MAX_SSE_FRAME_BYTES: Final = 2 * 1024 * 1024
# urllib applies this socket timeout between stream reads: a silent stream must
# yield to the authoritative GET rather than leave the CLI waiting forever.
SSE_IDLE_TIMEOUT_SECONDS: Final = 30.0
MAX_JOB_JSON_BYTES: Final = 8 * 1024 * 1024
MAX_ERROR_JSON_BYTES: Final = 64 * 1024
MAX_STREAM_CONNECT_RETRIES: Final = 3
MAX_AUTHORITATIVE_GET_ATTEMPTS: Final = 3
GET_RETRY_BACKOFF_INITIAL_SECONDS: Final = 1.0
POLL_BACKOFF_INITIAL_SECONDS: Final = 1.0
MAX_IDLE_INTERVAL_SECONDS: Final = 10.0

JobStatus = Literal["queued", "running", "succeeded", "canceling", "canceled", "failed", "expired"]
JobEventName = Literal["status", "progress", "preview", "output"]
_TERMINAL_STATUSES: Final = frozenset({"succeeded", "canceled", "failed", "expired"})
_RETRIABLE_STREAM_CODES: Final = frozenset({"too_many_streams", "rate_limited"})


def _ignore_event(_name: JobEventName, _payload: JsonObject) -> None:
    return


@dataclass(frozen=True, slots=True)
class JobWatchRequest:
    target: Target
    job_url: str
    events_url: str


@dataclass(frozen=True, slots=True)
class JobEventCallbacks:
    on_event: Callable[[JobEventName, JsonObject], None] = _ignore_event


@dataclass(frozen=True, slots=True)
class JobWatchResult:
    job: JsonObject
    outputs: list[JsonObject]


class DeployJobFailedError(DeployAPIError):
    code = "deploy_job_failed"

    def __init__(self, job: JsonObject) -> None:
        self.job = job
        job_id = job.get("id")
        super().__init__(self.code, f"deploy job {job_id} failed", details={"job": job})


class _WatchState(str, Enum):
    CONNECT_STREAM = "connect_stream"
    GET_AUTHORITY = "get_authority"
    POLL = "poll"
    DONE = "done"


def _retry_after(error: urllib.error.HTTPError, fallback: float) -> float:
    value = error.headers.get("Retry-After") if error.headers is not None else None
    try:
        parsed = float(value) if value is not None else fallback
    except (TypeError, ValueError):
        parsed = fallback
    return min(max(parsed, 0.0), MAX_IDLE_INTERVAL_SECONDS)


def _stream_error_code(error: urllib.error.HTTPError, url: str) -> str | None:
    try:
        raw = read_capped(error, url, max_bytes=MAX_ERROR_JSON_BYTES)
        parsed = json.loads(raw) if raw else None
    except (ResponseTooLarge, json.JSONDecodeError, UnicodeDecodeError, RecursionError):
        return None
    match parsed:
        case {"error": {"code": str(code)}}:
            return code
        case {"error": str(code)}:
            return code
        case _:
            return None


def _status(value: JsonObject) -> JobStatus:
    match value.get("status"):
        case ("queued" | "running" | "succeeded" | "canceling" | "canceled" | "failed" | "expired") as status:
            return status
        case invalid:
            raise DeployAPIError(
                "deploy_server_error",
                "the data-plane job returned an invalid status",
                details={"status": invalid},
            )


def _dispatch_frame(
    event_name: str,
    data_lines: list[str],
    callbacks: JobEventCallbacks,
    rendered_outputs: set[str],
) -> bool:
    if event_name == "log":
        return False
    match event_name:
        case "status" | "progress" | "preview" | "output":
            event_type: JobEventName = event_name
        case _:
            return False
    try:
        payload = json.loads("\n".join(data_lines))
    except (json.JSONDecodeError, RecursionError):
        return True
    if not isinstance(payload, dict):
        return True
    if event_type == "output":
        asset_id = payload.get("id")
        if not isinstance(asset_id, str) or not asset_id:
            return True
        if asset_id in rendered_outputs:
            return False
        rendered_outputs.add(asset_id)
    callbacks.on_event(event_type, payload)
    return event_type == "status" and payload.get("status") in _TERMINAL_STATUSES


def _consume_stream(response, callbacks: JobEventCallbacks, rendered_outputs: set[str]) -> None:
    event_name = ""
    data_lines: list[str] = []
    frame_bytes = 0
    while True:
        try:
            raw_line = response.readline(MAX_SSE_FRAME_BYTES - frame_bytes + 1)
        except (TimeoutError, OSError, http.client.HTTPException):
            return
        if not raw_line:
            return
        frame_bytes += len(raw_line)
        if frame_bytes > MAX_SSE_FRAME_BYTES:
            return
        try:
            line = raw_line.decode("utf-8").rstrip("\r\n")
        except UnicodeDecodeError:
            return
        if not line:
            if event_name or data_lines:
                if (
                    not event_name
                    or not data_lines
                    or _dispatch_frame(event_name, data_lines, callbacks, rendered_outputs)
                ):
                    return
            event_name = ""
            data_lines = []
            frame_bytes = 0
            continue
        if line.startswith(":"):
            continue
        field, separator, value = line.partition(":")
        if not separator:
            return
        value = value.removeprefix(" ")
        if field == "event":
            event_name = value
        elif field == "data":
            data_lines.append(value)


def _open_and_consume(request: JobWatchRequest, callbacks: JobEventCallbacks, rendered_outputs: set[str]) -> None:
    stream_request = build_authed_request(request.events_url, request.target)
    stream_request.add_header("Accept", "text/event-stream")
    with no_redirect_urlopen(stream_request, timeout=SSE_IDLE_TIMEOUT_SECONDS) as response:
        _consume_stream(response, callbacks, rendered_outputs)


def _authoritative_get(request: JobWatchRequest, sleep_fn: Callable[[float], None]) -> JsonObject:
    for attempt in range(MAX_AUTHORITATIVE_GET_ATTEMPTS):
        try:
            _, parsed = request_json(request.job_url, request.target, max_bytes=MAX_JOB_JSON_BYTES)
        except urllib.error.HTTPError as error:
            retryable = error.code == 429 or 500 <= error.code <= 599
            if not retryable or attempt + 1 == MAX_AUTHORITATIVE_GET_ATTEMPTS:
                raise
            sleep_fn(_retry_after(error, GET_RETRY_BACKOFF_INITIAL_SECONDS * (2**attempt)))
            continue
        except (TimeoutError, urllib.error.URLError):
            if attempt + 1 == MAX_AUTHORITATIVE_GET_ATTEMPTS:
                raise
            sleep_fn(min(GET_RETRY_BACKOFF_INITIAL_SECONDS * (2**attempt), MAX_IDLE_INTERVAL_SECONDS))
            continue
        if not isinstance(parsed, dict):
            raise DeployAPIError("deploy_server_error", "the data-plane job response is not an object")
        _status(parsed)
        return parsed
    raise AssertionError("authoritative GET retry loop exhausted without returning or raising")


def _outputs(job: JsonObject) -> list[JsonObject]:
    raw_outputs = job.get("outputs")
    if not isinstance(raw_outputs, list):
        raise DeployAPIError("deploy_server_error", "the data-plane job response has no outputs array")
    reconciled: list[JsonObject] = []
    asset_ids: set[str] = set()
    for output in raw_outputs:
        if not isinstance(output, dict):
            raise DeployAPIError("deploy_server_error", "the data-plane job response has an invalid output")
        asset_id = output.get("id")
        if not isinstance(asset_id, str) or not asset_id:
            raise DeployAPIError("deploy_server_error", "the data-plane job output has no asset id")
        if asset_id not in asset_ids:
            asset_ids.add(asset_id)
            reconciled.append(output)
    return reconciled


def watch_job(
    request: JobWatchRequest,
    callbacks: JobEventCallbacks = JobEventCallbacks(),
    sleep_fn: Callable[[float], None] = time.sleep,
) -> JobWatchResult:
    """Watch live hints, then return only state reconciled from authoritative GETs."""
    assert_safe_deploy_url(request.events_url, source="the job events url")
    assert_safe_deploy_url(request.job_url, source="the job url")
    state = _WatchState.CONNECT_STREAM
    stream_retries = 0
    poll_delay = POLL_BACKOFF_INITIAL_SECONDS
    final_job: JsonObject | None = None
    rendered_outputs: set[str] = set()

    while state is not _WatchState.DONE:
        match state:
            case _WatchState.CONNECT_STREAM:
                try:
                    _open_and_consume(request, callbacks, rendered_outputs)
                except urllib.error.HTTPError as error:
                    code = _stream_error_code(error, request.events_url) if error.code == 429 else None
                    can_retry = code in _RETRIABLE_STREAM_CODES and stream_retries < MAX_STREAM_CONNECT_RETRIES
                    if can_retry:
                        sleep_fn(_retry_after(error, GET_RETRY_BACKOFF_INITIAL_SECONDS * (2**stream_retries)))
                        stream_retries += 1
                        continue
                    state = _WatchState.POLL
                except (TimeoutError, urllib.error.URLError):
                    state = _WatchState.POLL
                else:
                    state = _WatchState.GET_AUTHORITY
            case _WatchState.GET_AUTHORITY | _WatchState.POLL:
                final_job = _authoritative_get(request, sleep_fn)
                if _status(final_job) in _TERMINAL_STATUSES:
                    state = _WatchState.DONE
                else:
                    state = _WatchState.POLL
                    sleep_fn(poll_delay)
                    poll_delay = min(poll_delay * 2, MAX_IDLE_INTERVAL_SECONDS)
            case _WatchState.DONE:
                # Unreachable by the `while state is not DONE` guard above, and kept
                # only so `assert_never` can prove the match exhausts _WatchState.
                raise AssertionError("done state must terminate the watch loop")
            case unreachable:
                assert_never(unreachable)

    if final_job is None:
        raise AssertionError("done state requires an authoritative job")
    status = _status(final_job)
    if status == "failed":
        raise DeployJobFailedError(final_job)
    if status != "succeeded":
        # `canceled` and `expired` are classified by the caller, which raises
        # before it reads these outputs. Demanding an outputs array of a job that
        # never finished would report `deploy_server_error` for a job the user
        # themselves canceled, burying `deploy_job_canceled`.
        return JobWatchResult(job=final_job, outputs=[])
    return JobWatchResult(job=final_job, outputs=_outputs(final_job))
