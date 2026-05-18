"""Async-job polling for partner endpoints.

There are two flavors:

1. **BFL** — the server returns ``{id, polling_url}`` on submit and we just GET
   that URL until the ``status`` field is terminal.
2. **Everything else** — a small ``PollSpec`` per partner describes where the
   job id lives in the create response, how to construct the poll URL (some
   partners use a sibling endpoint relative to the create path; others have a
   dedicated ``/tasks/{id}`` endpoint), and which status values mean
   "succeeded" / "failed".

The generic poller walks dot-paths into the JSON to extract the id/status
without having to write a new adapter for each partner.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import httpx

from comfy_cli.command.generate import client


@dataclass
class PollResult:
    """Normalized terminal state of an async job."""

    status: str  # "succeeded" | "failed" | "cancelled"
    raw: dict[str, Any]  # last response body — full upstream payload
    image_urls: list[str]  # any image/video result URLs we could pluck out
    error: str | None = None


# Recognized result extensions when sniffing URLs out of a poll body.
_MEDIA_EXTS = (
    ".png",
    ".jpg",
    ".jpeg",
    ".webp",
    ".gif",
    ".svg",
    ".mp4",
    ".mov",
    ".webm",
    ".m4v",
    ".gltf",
    ".glb",
    ".obj",
    ".fbx",
    ".wav",
    ".mp3",
    ".m4a",
    ".flac",
)


def _now() -> float:
    return time.monotonic()


def _extract_urls(node: Any) -> list[str]:
    """Walk a JSON tree, collecting strings that look like media URLs."""
    found: list[str] = []

    def visit(n: Any) -> None:
        if isinstance(n, str):
            low = n.lower()
            if n.startswith(("http://", "https://")) and (
                low.split("?", 1)[0].endswith(_MEDIA_EXTS) or "image" in low or "video" in low
            ):
                found.append(n)
            return
        if isinstance(n, dict):
            for v in n.values():
                visit(v)
        elif isinstance(n, list):
            for v in n:
                visit(v)

    visit(node)
    seen: set[str] = set()
    return [u for u in found if not (u in seen or seen.add(u))]


def _dotget(body: Any, path: str) -> Any:
    """Look up a dotted path inside a JSON body. Returns None if any segment misses."""
    cur: Any = body
    for part in path.split("."):
        if isinstance(cur, dict):
            cur = cur.get(part)
        else:
            return None
    return cur


def _first(body: Any, paths: tuple[str, ...]) -> Any:
    for p in paths:
        v = _dotget(body, p)
        if v not in (None, "", []):
            return v
    return None


def _sleep(seconds: float) -> None:
    time.sleep(seconds)


def poll_bfl(
    initial: dict[str, Any],
    api_key: str,
    *,
    interval: float = 2.0,
    timeout: float = 300.0,
    on_progress: Callable[[float], None] | None = None,
    create_path: str | None = None,  # ignored, kept for uniform signature
) -> PollResult:
    """BFL polls a server-issued ``polling_url`` until ``status`` flips to Ready."""
    url = initial.get("polling_url")
    if not url:
        raise client.ApiError(0, "", "BFL response missing polling_url")
    deadline = _now() + timeout
    last_body: dict[str, Any] = {}
    while _now() < deadline:
        resp = client.get(url, api_key=api_key)
        if resp.status_code >= 400:
            client.raise_for_status(resp)
        last_body = resp.json()
        status = str(last_body.get("status", "")).strip()
        if on_progress is not None:
            progress = last_body.get("progress")
            if isinstance(progress, int | float):
                on_progress(float(progress))
        if status == "Ready":
            urls = _extract_urls(last_body.get("result"))
            return PollResult(status="succeeded", raw=last_body, image_urls=urls)
        if status in {"Error", "Task not found", "Content Moderated", "Request Moderated"}:
            return PollResult(status="failed", raw=last_body, image_urls=[], error=status)
        _sleep(interval)
    return PollResult(status="failed", raw=last_body, image_urls=[], error=f"timed out after {timeout:.0f}s")


@dataclass(frozen=True)
class PollSpec:
    """Per-partner polling configuration.

    ``poll_url`` is a template supporting ``{id}`` and ``{create_path}``;
    ``post_success_url`` (optional) is a second-stage fetcher invoked once the
    job reaches a success state — for partners like MiniMax where the terminal
    poll response gives you a file id you still need to redeem for a URL."""

    name: str
    id_paths: tuple[str, ...]
    poll_url: str
    status_paths: tuple[str, ...]
    success_values: tuple[str, ...]
    failure_values: tuple[str, ...] = ()
    progress_path: str | None = None
    post_success_url: str | None = None
    post_success_id_paths: tuple[str, ...] = field(default_factory=tuple)


_POLL_SPECS: dict[str, PollSpec] = {
    "kling": PollSpec(
        name="kling",
        id_paths=("data.task_id",),
        poll_url="{create_path}/{id}",
        status_paths=("data.task_status",),
        success_values=("succeed",),
        failure_values=("failed",),
    ),
    "luma": PollSpec(
        name="luma",
        id_paths=("id",),
        poll_url="/proxy/luma/generations/{id}",
        status_paths=("state",),
        success_values=("completed",),
        failure_values=("failed",),
    ),
    "minimax": PollSpec(
        name="minimax",
        id_paths=("task_id",),
        poll_url="/proxy/minimax/query/video_generation?task_id={id}",
        status_paths=("status",),
        success_values=("Success",),
        failure_values=("Fail",),
        post_success_url="/proxy/minimax/files/retrieve?file_id={id}",
        post_success_id_paths=("file_id",),
    ),
    "runway": PollSpec(
        name="runway",
        id_paths=("id",),
        poll_url="/proxy/runway/tasks/{id}",
        status_paths=("status",),
        success_values=("SUCCEEDED",),
        failure_values=("FAILED", "CANCELLED", "THROTTLED"),
        progress_path="progress",
    ),
    "moonvalley": PollSpec(
        name="moonvalley",
        id_paths=("id",),
        poll_url="/proxy/moonvalley/prompts/{id}",
        status_paths=("status",),
        success_values=("completed",),
        failure_values=("failed", "error"),
    ),
    "pika": PollSpec(
        name="pika",
        id_paths=("video_id", "id"),
        poll_url="/proxy/pika/videos/{id}",
        status_paths=("status",),
        success_values=("finished",),
        # Pika's enum has no explicit failure state; treat sustained queued/started
        # as in-progress and rely on `timeout` to surface stalls.
        failure_values=(),
        progress_path="progress",
    ),
    "vidu": PollSpec(
        name="vidu",
        id_paths=("task_id",),
        poll_url="/proxy/vidu/tasks/{id}/creations",
        status_paths=("state",),
        success_values=("success",),
        failure_values=("failed",),
    ),
    "xai_video": PollSpec(
        name="xai_video",
        id_paths=("request_id",),
        poll_url="/proxy/xai/v1/videos/{id}",
        status_paths=("status",),
        success_values=("done",),
        failure_values=(),
    ),
    "seedance": PollSpec(
        name="seedance",
        id_paths=("id",),
        poll_url="/proxy/byteplus/api/v3/contents/generations/tasks/{id}",
        status_paths=("status",),
        success_values=("succeeded",),
        failure_values=("failed", "cancelled"),
    ),
}


def _build_poll_url(spec: PollSpec, job_id: str, create_path: str | None) -> str:
    url = spec.poll_url.replace("{id}", str(job_id))
    if "{create_path}" in url:
        if not create_path:
            raise client.ApiError(0, "", f"{spec.name} poller needs the create path")
        url = url.replace("{create_path}", create_path)
    return url


def poll_generic(
    initial: dict[str, Any],
    api_key: str,
    *,
    spec: PollSpec,
    create_path: str | None = None,
    interval: float = 2.0,
    timeout: float = 300.0,
    on_progress: Callable[[float], None] | None = None,
) -> PollResult:
    """Drive a partner's poll endpoint by reading dot-paths out of the JSON.

    ``initial`` is the create-response body; we pull a job id out of it, build
    the poll URL from ``spec.poll_url``, and GET it until the status field hits
    a terminal value. Handles MiniMax-style two-stage flows via
    ``spec.post_success_url`` (a follow-up GET keyed off something the terminal
    poll body contains, e.g. ``file_id``)."""
    job_id = _first(initial, spec.id_paths)
    if job_id is None:
        raise client.ApiError(0, "", f"{spec.name} response missing id (looked for {spec.id_paths})")
    url = _build_poll_url(spec, str(job_id), create_path)
    deadline = _now() + timeout
    last_body: dict[str, Any] = {}
    while _now() < deadline:
        resp = client.get(url, api_key=api_key)
        if resp.status_code >= 400:
            client.raise_for_status(resp)
        last_body = resp.json()
        if on_progress is not None and spec.progress_path:
            p = _dotget(last_body, spec.progress_path)
            if isinstance(p, int | float):
                # Some partners report 0–100, others 0–1; normalize.
                on_progress(float(p) / 100.0 if p > 1 else float(p))
        status = _first(last_body, spec.status_paths)
        status_str = str(status) if status is not None else ""
        if status_str in spec.success_values:
            merged = dict(last_body)
            if spec.post_success_url:
                redeem_id = _first(last_body, spec.post_success_id_paths)
                if redeem_id is not None:
                    redeem_url = spec.post_success_url.replace("{id}", str(redeem_id))
                    r2 = client.get(redeem_url, api_key=api_key)
                    if r2.status_code < 400:
                        try:
                            merged["_redeemed"] = r2.json()
                        except ValueError:
                            pass
            urls = _extract_urls(merged)
            return PollResult(status="succeeded", raw=merged, image_urls=urls)
        if status_str in spec.failure_values:
            return PollResult(status="failed", raw=last_body, image_urls=[], error=status_str)
        _sleep(interval)
    return PollResult(status="failed", raw=last_body, image_urls=[], error=f"timed out after {timeout:.0f}s")


def extract_job_id(name: str, body: dict[str, Any]) -> str | None:
    """Pull the partner's job id out of a create-response body for display."""
    if name == "bfl":
        return body.get("id") or None
    spec = _POLL_SPECS.get(name)
    if spec is None:
        return None
    v = _first(body, spec.id_paths)
    return str(v) if v is not None else None


def build_synthetic_initial(name: str, job_id: str, base_url: str | None = None) -> dict[str, Any]:
    """Recreate a minimal create-response so ``poll_generic`` can find the id.

    Used by ``comfy generate resume`` — the user supplies just a partner key
    and a job id, and we reverse-engineer the shape the poller expects."""
    if name == "bfl":
        if not base_url:
            raise client.ApiError(0, "", "BFL resume needs a base URL to build the polling_url")
        return {"polling_url": f"{base_url}/proxy/bfl/get_result?id={job_id}"}
    spec = _POLL_SPECS.get(name)
    if not spec:
        raise client.ApiError(0, "", f"No polling adapter for partner {name!r}")
    primary = spec.id_paths[0]
    body: dict[str, Any] = {}
    cur = body
    parts = primary.split(".")
    for p in parts[:-1]:
        cur[p] = {}
        cur = cur[p]
    cur[parts[-1]] = job_id
    return body


def get_poller(name: str) -> Callable[..., PollResult]:
    """Return the poller callable for a partner name.

    All pollers accept the same kwargs (``api_key``, ``timeout``, ``on_progress``,
    ``create_path``) so callers don't need to special-case which one they got."""
    if name == "bfl":
        return poll_bfl
    if name in _POLL_SPECS:
        spec = _POLL_SPECS[name]

        def runner(
            initial: dict[str, Any],
            api_key: str,
            *,
            create_path: str | None = None,
            interval: float = 2.0,
            timeout: float = 300.0,
            on_progress: Callable[[float], None] | None = None,
        ) -> PollResult:
            return poll_generic(
                initial,
                api_key,
                spec=spec,
                create_path=create_path,
                interval=interval,
                timeout=timeout,
                on_progress=on_progress,
            )

        return runner
    raise client.ApiError(0, "", f"No polling adapter for partner {name!r}")


def sync_result_from_response(resp: httpx.Response) -> PollResult:
    """Wrap a sync response in a PollResult so the run path is uniform."""
    ctype = resp.headers.get("content-type", "")
    if ctype.startswith(("image/", "video/", "audio/")):
        return PollResult(status="succeeded", raw={"_binary": True}, image_urls=[])
    try:
        body = resp.json()
    except ValueError:
        return PollResult(status="succeeded", raw={"_text": resp.text}, image_urls=[])
    return PollResult(status="succeeded", raw=body, image_urls=_extract_urls(body))
