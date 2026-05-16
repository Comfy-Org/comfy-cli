"""Async-job polling adapters, one per partner.

Each adapter takes the initial response from the submit POST and yields a
canonical ``PollResult`` once the upstream job reaches a terminal state.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import httpx

from comfy_cli.command.generate import client


@dataclass
class PollResult:
    """Normalized terminal state of an async job."""

    status: str  # "succeeded" | "failed" | "cancelled"
    raw: dict[str, Any]  # last response body — full upstream payload
    image_urls: list[str]  # any image result URLs we could pluck out
    error: str | None = None


def _now() -> float:
    return time.monotonic()


def _extract_urls(node: Any) -> list[str]:
    """Walk a JSON tree, collecting strings that look like image URLs."""
    found: list[str] = []

    def visit(n: Any) -> None:
        if isinstance(n, str):
            low = n.lower()
            if n.startswith(("http://", "https://")) and (
                low.endswith((".png", ".jpg", ".jpeg", ".webp", ".gif", ".svg")) or "image" in low
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
    # De-dupe preserving order.
    seen: set[str] = set()
    return [u for u in found if not (u in seen or seen.add(u))]


def _sleep(seconds: float) -> None:
    time.sleep(seconds)


def poll_bfl(
    initial: dict[str, Any],
    api_key: str,
    *,
    interval: float = 2.0,
    timeout: float = 300.0,
    on_progress: Callable[[float], None] | None = None,
) -> PollResult:
    """BFL: GET ``polling_url`` until ``status`` is terminal."""
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


_POLLERS: dict[str, Callable[..., PollResult]] = {
    "bfl": poll_bfl,
    # "kling": poll_kling,   # follow-up: not in v1 image allowlist
    # "luma": poll_luma,     # follow-up
    # "topaz": poll_topaz,   # follow-up
}


def get_poller(name: str) -> Callable[..., PollResult]:
    try:
        return _POLLERS[name]
    except KeyError as e:
        raise client.ApiError(0, "", f"No polling adapter for partner {name!r}") from e


def sync_result_from_response(resp: httpx.Response) -> PollResult:
    """Wrap a sync response in a PollResult so the run path is uniform."""
    if resp.headers.get("content-type", "").startswith("image/"):
        return PollResult(status="succeeded", raw={"_binary": True}, image_urls=[])
    try:
        body = resp.json()
    except ValueError:
        return PollResult(status="succeeded", raw={"_text": resp.text}, image_urls=[])
    return PollResult(status="succeeded", raw=body, image_urls=_extract_urls(body))
