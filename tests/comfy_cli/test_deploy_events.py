from __future__ import annotations

import copy
import io
import json
import urllib.error
from email.message import Message

import pytest

import comfy_cli.deploy_events as deploy_events
from comfy_cli.deploy_events import (
    DeployJobFailedError,
    JobEventCallbacks,
    JobWatchRequest,
    watch_job,
)
from comfy_cli.target import Target

JOB_URL = "https://api.example.com/api/v2/jobs/job-1"
EVENTS_URL = f"{JOB_URL}/events"
TARGET = Target(kind="cloud", base_url="https://api.example.com", auth_token="token")


def _output(asset_id: str) -> dict:
    return {
        "node_id": "7",
        "name": f"{asset_id}.png",
        "type": "image",
        "content_type": "image/png",
        "size_bytes": 123,
        "id": asset_id,
        "hash": None,
        "url": f"https://assets.example.com/{asset_id}",
        "url_expires_at": "2026-08-23T12:00:00Z",
    }


def _job(status: str, *, outputs: tuple[dict, ...] = ()) -> dict:
    return {
        "id": "job-1",
        "status": status,
        "created_at": "2026-08-23T10:00:00Z",
        "started_at": "2026-08-23T10:00:01Z",
        "completed_at": "2026-08-23T10:00:02Z" if status != "running" else None,
        "expires_at": "2026-08-24T10:00:00Z",
        "queue_position": None,
        "progress": None,
        "outputs": list(outputs),
        "error": {"code": "execution_failed", "message": "node failed"} if status == "failed" else None,
        "metrics": {"duration_ms": 1000},
        "urls": {"events": EVENTS_URL},
    }


def _sse(*frames: tuple[str, dict]) -> bytes:
    return "".join(f"event: {name}\ndata: {json.dumps(payload)}\n\n" for name, payload in frames).encode()


def _http_error(status: int, code: str, *, retry_after: str | None = None) -> urllib.error.HTTPError:
    headers = Message()
    if retry_after is not None:
        headers["Retry-After"] = retry_after
    body = json.dumps({"error": {"code": code}}).encode()
    return urllib.error.HTTPError(EVENTS_URL, status, code, headers, io.BytesIO(body))


def _install_wire(monkeypatch, streams, jobs):
    request_log: list[str] = []
    sleeps: list[float] = []
    stream_results = list(streams)
    job_results = list(jobs)

    def open_stream(request, *, timeout):
        request_log.append("events")
        result = stream_results.pop(0)
        if isinstance(result, BaseException):
            raise result
        return io.BytesIO(result)

    def request_json(url, target, *, method="GET", body=None, headers=None, timeout=30.0, max_bytes):
        request_log.append("job")
        result = job_results.pop(0)
        if isinstance(result, BaseException):
            raise result
        return 200, copy.deepcopy(result)

    monkeypatch.setattr(deploy_events, "no_redirect_urlopen", open_stream)
    monkeypatch.setattr(deploy_events, "request_json", request_json)
    return request_log, sleeps, lambda seconds: sleeps.append(seconds)


def _watch(sleep_fn, *, callbacks: JobEventCallbacks | None = None):
    request = JobWatchRequest(target=TARGET, job_url=JOB_URL, events_url=EVENTS_URL)
    return watch_job(request, callbacks or JobEventCallbacks(), sleep_fn)


def test_501_makes_one_stream_request_then_polls(monkeypatch) -> None:
    # Given
    request_log, _, sleep_fn = _install_wire(monkeypatch, [_http_error(501, "not_implemented")], [_job("succeeded")])

    # When
    result = _watch(sleep_fn)

    # Then
    assert result.job["status"] == "succeeded"
    assert request_log == ["events", "job"]


def test_midstream_eof_gets_authority_then_polls_without_reconnecting(monkeypatch) -> None:
    # Given
    stream = _sse(("status", {"status": "running"}))
    request_log, sleeps, sleep_fn = _install_wire(monkeypatch, [stream], [_job("running"), _job("succeeded")])

    # When
    result = _watch(sleep_fn)

    # Then
    assert result.job["status"] == "succeeded"
    assert request_log == ["events", "job", "job"]
    assert sleeps == [deploy_events.POLL_BACKOFF_INITIAL_SECONDS]


def test_malformed_frame_falls_back_without_reconnecting(monkeypatch) -> None:
    # Given
    stream = _sse(("status", {"status": "running"})) + b"event: output\ndata: {broken}\n\n"
    request_log, _, sleep_fn = _install_wire(monkeypatch, [stream], [_job("succeeded")])

    # When
    result = _watch(sleep_fn)

    # Then
    assert result.job["status"] == "succeeded"
    assert request_log == ["events", "job"]


def test_429_retries_stream_three_times_then_polls(monkeypatch) -> None:
    # Given
    limited = [_http_error(429, "too_many_streams", retry_after="7") for _ in range(4)]
    request_log, sleeps, sleep_fn = _install_wire(monkeypatch, limited, [_job("succeeded")])

    # When
    result = _watch(sleep_fn)

    # Then
    assert result.job["status"] == "succeeded"
    assert request_log == ["events", "events", "events", "events", "job"]
    assert sleeps == [7.0, 7.0, 7.0]


def test_authoritative_failure_overrules_stream_success(monkeypatch) -> None:
    # Given
    stream = _sse(("status", {"status": "succeeded"}))
    request_log, _, sleep_fn = _install_wire(monkeypatch, [stream], [_job("failed")])

    # When
    with pytest.raises(DeployJobFailedError) as exc_info:
        _watch(sleep_fn)

    # Then
    assert exc_info.value.code == "deploy_job_failed"
    assert exc_info.value.job["status"] == "failed"
    assert request_log == ["events", "job"]


def test_final_outputs_include_assets_missing_from_sse(monkeypatch) -> None:
    # Given
    first = _output("asset-1")
    second = _output("asset-2")
    stream = _sse(("output", first), ("status", {"status": "succeeded"}))
    _, _, sleep_fn = _install_wire(monkeypatch, [stream], [_job("succeeded", outputs=(first, second))])

    # When
    result = _watch(sleep_fn)
    would_download = [output["id"] for output in result.outputs]

    # Then
    assert would_download == ["asset-1", "asset-2"]


def test_duplicate_output_hints_render_and_download_once(monkeypatch) -> None:
    # Given
    output = _output("asset-1")
    stream = _sse(("output", output), ("output", output), ("status", {"status": "succeeded"}))
    _, _, sleep_fn = _install_wire(monkeypatch, [stream], [_job("succeeded", outputs=(output, output))])
    rendered: list[str] = []

    def record_output(name, payload) -> None:
        asset_id = payload.get("id")
        if name == "output" and isinstance(asset_id, str):
            rendered.append(asset_id)

    callbacks = JobEventCallbacks(on_event=record_output)

    # When
    result = _watch(sleep_fn, callbacks=callbacks)

    # Then
    assert rendered == ["asset-1"]
    assert [item["id"] for item in result.outputs] == ["asset-1"]


def test_log_frame_is_ignored_and_later_frames_are_processed(monkeypatch) -> None:
    # Given
    stream = _sse(("log", {"message": "reserved"}), ("progress", {"value": 1, "nodes_done": 1, "nodes_total": 1}))
    _, _, sleep_fn = _install_wire(monkeypatch, [stream], [_job("succeeded")])
    rendered: list[str] = []
    callbacks = JobEventCallbacks(on_event=lambda name, _payload: rendered.append(name))

    # When
    result = _watch(sleep_fn, callbacks=callbacks)

    # Then
    assert result.job["status"] == "succeeded"
    assert rendered == ["progress"]


def test_final_authoritative_get_retries_a_transient_failure(monkeypatch) -> None:
    # Given
    stream = _sse(("status", {"status": "running"}))
    transient = _http_error(503, "temporarily_unavailable")
    request_log, sleeps, sleep_fn = _install_wire(monkeypatch, [stream], [transient, _job("succeeded")])

    # When
    result = _watch(sleep_fn)

    # Then
    assert result.job["status"] == "succeeded"
    assert request_log == ["events", "job", "job"]
    assert sleeps == [deploy_events.GET_RETRY_BACKOFF_INITIAL_SECONDS]
