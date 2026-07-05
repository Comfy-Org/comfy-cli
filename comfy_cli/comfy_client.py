"""Unified HTTP client for ComfyUI — local or cloud.

A single class that takes a :class:`comfy_cli.target.Target` and exposes the
shared REST surface (``submit_prompt``, ``get_history``, ``list_jobs``,
``get_job_status``). Local and cloud differ in three small ways — path
prefix, history endpoint name, and whether a Bearer token is added — all of
which are encoded as Target fields.

WebSocket-based live-watch stays in :mod:`comfy_cli.command.jobs` since it's
local-only.
"""

from __future__ import annotations

import json
import random
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from comfy_cli.http import NoRedirectHandler
from comfy_cli.target import Target

_LOOPBACK_HOSTS = {"localhost", "127.0.0.1", "::1", "[::1]"}

# Transient HTTP failures during polling should back off and retry, not abort.
# 429 (rate limit) is retried for any method — the request was rejected, not
# processed, so even a POST is safe to repeat. Transient 5xx is retried for
# GET only, since a 5xx on a POST may have partially applied (double-submit).
_MAX_TRANSIENT_RETRIES = 4
_RETRYABLE_5XX = {502, 503, 504}

# Poll-level resilience for wait_for_completion: a sustained 429 storm (or a
# plain 500, which the in-request layer never retries) must not abort a wait
# for a job that is still running. Backoff doubles from 2s up to 60s with
# jitter; only _MAX_POLL_FAILURES CONSECUTIVE failed polls give up and
# surface the HTTPError to the caller's existing error path.
_MAX_POLL_FAILURES = 5
_POLL_BACKOFF_BASE = 2.0
_POLL_BACKOFF_CAP = 60.0

# Strip Bearer tokens out of any text we might carry into logs/exceptions.
_BEARER_RE = re.compile(r"Bearer\s+[A-Za-z0-9._\-]+", re.IGNORECASE)
_TOKEN_KEYS_RE = re.compile(
    r'("(?:auth_token_comfy_org|api_key_comfy_org|access_token|refresh_token)"\s*:\s*")[^"]*(")',
    re.IGNORECASE,
)


def _redact(text: str) -> str:
    """Strip Bearer tokens and known token keys from a string."""
    if not text:
        return text
    return _TOKEN_KEYS_RE.sub(r"\1***\2", _BEARER_RE.sub("Bearer ***", text))


def _parse_retry_after(headers: Any) -> float | None:
    """Parse a ``Retry-After`` header (seconds form) to a float, else None."""
    if headers is None:
        return None
    value = headers.get("Retry-After")
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


class HTTPError(Exception):
    """Server returned a non-2xx response.

    ``retry_after`` carries the server's ``Retry-After`` header (seconds)
    when one was present, so retry layers above ``_request`` (e.g. the
    wait_for_completion poll loop) can honor it.
    """

    def __init__(self, status: int, message: str, body: str = "", *, retry_after: float | None = None):
        # Redact before super().__init__ so str(self) is safe to log.
        message = _redact(message)
        body = _redact(body)
        super().__init__(f"HTTP {status}: {message}")
        self.status = status
        self.message = message
        self.body = body
        self.retry_after = retry_after


class Unauthenticated(Exception):
    """Target needs auth but no valid session is present."""


_OPENER = urllib.request.build_opener(NoRedirectHandler())


def _assert_safe_url(url: str) -> None:
    """Reject plaintext HTTP for non-loopback hosts.

    Anything carrying a Bearer token over the wire must be HTTPS unless the
    host is a loopback address (where there's no network to sniff).
    """
    parsed = urllib.parse.urlsplit(url)
    if parsed.scheme == "https":
        return
    host = (parsed.hostname or "").lower()
    if host in _LOOPBACK_HOSTS:
        return
    raise ValueError(
        f"refusing to send request to non-https, non-loopback URL: {url} "
        "(set COMFY_CLOUD_BASE_URL to an https:// endpoint)"
    )


@dataclass
class SubmitResult:
    prompt_id: str
    number: int | None
    node_errors: dict[str, Any]


class Client:
    """Single HTTP client across local + cloud ComfyUI."""

    def __init__(self, target: Target, *, timeout: float = 30.0, clear_session_on_auth_failure: bool = True):
        self.target = target
        self.timeout = timeout
        # Whether a fatal OAuth refresh failure (reuse-detection / invalid_grant)
        # on the reactive 401 path may clear the stored session. Foreground,
        # user-driven commands leave this True (they own the session lifecycle).
        # The detached background watcher sets it False: it is read-mostly and
        # must never log the user off the shared session over a transient blip.
        self.clear_session_on_auth_failure = clear_session_on_auth_failure
        if target.is_cloud and not (target.auth_token or target.api_key):
            raise Unauthenticated(
                "cloud target requires credentials — run `comfy cloud login` or set COMFY_CLOUD_API_KEY"
            )

    # ----- low-level -----

    def _try_refresh_token(self) -> bool:
        """Refresh the OAuth token after a server 401. Returns True if a new
        access token was obtained and installed on the target.

        This is the *reactive* leg and it shares the single, cross-process
        locked + double-checked refresh in ``oauth.ensure_fresh_session`` (via
        ``get_session(force=True)``) — there is no second refresh/persist code
        path here. Coalescing matters: in a parallel fan-out the first caller
        refreshes and the rest pick up the rotated token from the store without
        a second network call, so a consumed refresh token is never replayed.

        Raises ``Unauthenticated`` when the refresh hit reuse-detection /
        ``invalid_grant``: the family is dead, the session has already been
        cleared, and retrying is pointless — the caller surfaces the
        ``cloud_unauthorized`` login guidance exactly once.
        """
        if not self.target.is_cloud or not self.target.auth_token:
            return False
        # Only refresh OAuth tokens, not API keys.
        if self.target.api_key:
            return False

        from comfy_cli.credentials import get_session

        old_token = self.target.auth_token
        # Was there a stored OAuth session backing this token? If not (e.g. a
        # bare token handed to the client directly), there's nothing to refresh
        # and a cleared store doesn't mean "revoked" — just let the 401 stand.
        had_session = get_session(refresh=False) is not None
        # force=True: a server 401 is authoritative even if our local clock
        # still thinks the access token is valid (skew / no recorded expiry).
        # allow_clear: foreground commands may clear on a fatal token failure;
        # the background watcher passes False so a transient/spurious
        # invalid_grant can't wipe the shared session.
        session = get_session(refresh=True, force=True, allow_clear=self.clear_session_on_auth_failure)
        if session is not None and session.access_token and session.access_token != old_token:
            # Frozen dataclass — install the rotated token for the retry + any
            # subsequent requests (and the partner-API extra_data rebuild).
            object.__setattr__(self.target, "auth_token", session.access_token)
            return True
        if had_session and session is None:
            # The refresh hit a fatal token error. When
            # ``clear_session_on_auth_failure`` is True the stored session has
            # already been cleared; when False (watcher) it is deliberately
            # preserved for the foreground command to manage. Either way, don't
            # loop on a dead token — surface the failure once.
            #
            # Report the auth server's *actual* reason when we have it (e.g.
            # "invalid_grant: workspace membership lost") instead of guessing
            # "reuse detected" — these failures look identical to the user but
            # have very different root causes.
            from comfy_cli.cloud import oauth

            reason = oauth.take_last_fatal_refresh_reason()
            detail = reason or "refresh token reuse detected or expired"
            raise Unauthenticated(f"Comfy Cloud session is no longer valid ({detail}) — run `comfy cloud login`")
        # Same token back (a transient/network refresh failure kept the stale
        # session), or no stored session to refresh: let the original 401
        # propagate to the caller's HTTP-error handling.
        return False

    @staticmethod
    def _retry_delay(exc: urllib.error.HTTPError, attempt: int) -> float:
        """Seconds to wait before retrying a transient failure in-request.

        Honors a server ``Retry-After`` (seconds) when present; otherwise uses
        exponential backoff with jitter so concurrent waiters don't synchronize
        into bursts.
        """
        retry_after = _parse_retry_after(getattr(exc, "headers", None))
        if retry_after is not None:
            return min(retry_after, 30.0)
        base = min(2**attempt, 16)
        return base + random.uniform(0, base * 0.5)

    @staticmethod
    def _poll_retry_delay(exc: HTTPError, attempt: int) -> float:
        """Seconds to wait before the next poll after a transient poll failure.

        Same shape as ``_retry_delay`` but tuned for the long-lived
        wait_for_completion loop: ``Retry-After`` is honored (capped), else
        exponential backoff from ``_POLL_BACKOFF_BASE`` up to
        ``_POLL_BACKOFF_CAP`` with jitter.
        """
        if exc.retry_after is not None:
            return min(exc.retry_after, _POLL_BACKOFF_CAP)
        base = min(_POLL_BACKOFF_BASE * (2**attempt), _POLL_BACKOFF_CAP)
        return base + random.uniform(0, base * 0.5)

    def _request(
        self,
        method: str,
        path_parts: tuple[str, ...],
        *,
        body: dict | None = None,
        body_factory: Callable[[], dict] | None = None,
        timeout: float | None = None,
        _retried: bool = False,
        _attempt: int = 0,
    ) -> Any:
        url = self.target.url(*path_parts)
        # Only enforce https-or-loopback when we're carrying a bearer token.
        if self.target.is_cloud and (self.target.auth_token or self.target.api_key):
            _assert_safe_url(url)
        if body_factory is not None:
            body = body_factory()
        data = json.dumps(body).encode("utf-8") if body is not None else None
        req = urllib.request.Request(url, data=data, method=method)
        req.add_header("Accept", "application/json")
        # Usage-source attribution on every ComfyUI/cloud API request so the
        # server can tell CLI-originated traffic apart from the web UI (#468).
        req.add_header("Comfy-Usage-Source", "comfy-cli")
        if data is not None:
            req.add_header("Content-Type", "application/json")
        # Cloud auth: the policy layer (`resolve_target`) is OAuth-first and
        # populates at most one of api_key / auth_token, so this is just the
        # mechanic — send whichever field is set. Only attached on cloud
        # targets so a stray auth_token on a local target can't leak
        # credentials to a plaintext server.
        if self.target.is_cloud:
            if self.target.auth_token:
                req.add_header("Authorization", f"Bearer {self.target.auth_token}")
            elif self.target.api_key:
                req.add_header("X-API-Key", self.target.api_key)
        try:
            with _OPENER.open(req, timeout=timeout or self.timeout) as resp:
                text = resp.read().decode("utf-8", errors="replace")
                if not text:
                    return None
                return json.loads(text)
        except urllib.error.HTTPError as e:
            body_text = ""
            try:
                body_text = e.read().decode("utf-8", errors="replace")
            except Exception:  # noqa: BLE001
                pass
            # Auto-refresh on 401 for OAuth cloud targets, retry once.
            if (
                e.code == 401
                and not _retried
                and self.target.is_cloud
                and self.target.auth_token
                and self._try_refresh_token()
            ):
                return self._request(
                    method,
                    path_parts,
                    body=body,
                    body_factory=body_factory,
                    timeout=timeout,
                    _retried=True,
                )
            # Transient: back off and retry. 429 for any method (rejected, not
            # processed); 5xx for idempotent GETs only.
            retryable = e.code == 429 or (e.code in _RETRYABLE_5XX and method == "GET")
            if retryable and _attempt < _MAX_TRANSIENT_RETRIES:
                time.sleep(self._retry_delay(e, _attempt))
                return self._request(
                    method,
                    path_parts,
                    body=body,
                    body_factory=body_factory,
                    timeout=timeout,
                    _retried=_retried,
                    _attempt=_attempt + 1,
                )
            raise HTTPError(
                e.code,
                e.reason or "HTTP error",
                body_text,
                retry_after=_parse_retry_after(getattr(e, "headers", None)),
            ) from e

    # ----- submit -----

    def submit_prompt(
        self,
        workflow: dict,
        client_id: str,
        *,
        timeout: float | None = None,
        extra_data: dict | None = None,
    ) -> SubmitResult:
        """POST {prefix}/prompt — submit a workflow for execution.

        Caller may pass ``extra_data`` (merged into the request, not overwritten).
        For cloud submissions, the user's OAuth token is injected as
        ``auth_token_comfy_org`` so partner-API nodes (BFL Flux Pro, Gemini
        Nano Banana, etc.) can call out to comfy.org — matching what the web
        UI sends. The token rides the body in addition to the ``Authorization``
        header because comfy_api_nodes reads it from ``extra_data``; this is a
        gateway-layer concern, not a header-vs-body choice.
        """

        def payload() -> dict[str, Any]:
            request_payload: dict[str, Any] = {"prompt": workflow, "client_id": client_id}
            merged_extra: dict[str, Any] = dict(extra_data or {})
            # Usage-source attribution rides extra_data too — the execution
            # record keeps it even when the HTTP header is dropped by proxies.
            merged_extra.setdefault("comfy_usage_source", "comfy-cli")
            # Partner-API nodes (BFL, Gemini, Bria, ByteDance, etc.) read the
            # caller's comfy.org credential out of extra_data. Rebuild this at
            # send time so an OAuth refresh updates both the header and body.
            if self.target.is_cloud:
                if self.target.auth_token:
                    merged_extra.setdefault("auth_token_comfy_org", self.target.auth_token)
                elif self.target.api_key:
                    merged_extra.setdefault("api_key_comfy_org", self.target.api_key)
            if merged_extra:
                request_payload["extra_data"] = merged_extra
            return request_payload

        resp = self._request("POST", ("prompt",), body_factory=payload, timeout=timeout)
        if not isinstance(resp, dict) or "prompt_id" not in resp:
            raise HTTPError(200, "missing prompt_id in response", json.dumps(resp) if resp else "")
        return SubmitResult(
            prompt_id=resp["prompt_id"],
            number=resp.get("number"),
            node_errors=resp.get("node_errors", {}) or {},
        )

    # ----- history -----

    def get_history(self, prompt_id: str, *, timeout: float | None = None) -> dict | None:
        """GET {prefix}/{history_path}/<prompt_id>.

        Returns the history record, or None if not yet present (404 is transient
        right after submit — the record is created when execution starts).
        """
        try:
            resp = self._request("GET", (self.target.history_path, prompt_id), timeout=timeout)
        except HTTPError as e:
            if e.status == 404:
                return None
            raise
        if not isinstance(resp, dict):
            return None
        if prompt_id in resp:
            return resp[prompt_id]
        if "outputs" in resp or "status" in resp or "execution_status" in resp:
            return resp
        return None

    # ----- jobs (cloud only has a dedicated endpoint; local synthesizes) -----

    def list_jobs(self, *, limit: int = 10, timeout: float | None = None) -> list[dict]:
        """List recent + running prompts.

        Cloud → GET {prefix}/jobs.
        Local → /queue (running+pending) merged with /history (recent completions).
        """
        if self.target.jobs_path:
            qs = f"?limit={int(limit)}" if limit else ""
            resp = self._request("GET", (f"{self.target.jobs_path}{qs}",), timeout=timeout)
            if isinstance(resp, dict):
                items = resp.get("jobs") or []
                return [j for j in items if isinstance(j, dict)][:limit]
            return []
        # Local: fall through to caller — jobs.py already has the merging logic.
        raise NotImplementedError("local list_jobs uses /queue + /history merge — call jobs._gather_jobs")

    def get_job_status(self, prompt_id: str, *, timeout: float | None = None) -> dict | None:
        """Cloud-only: GET {prefix}/job/<id>/status."""
        try:
            return self._request("GET", ("job", prompt_id, "status"), timeout=timeout)
        except HTTPError as e:
            if e.status == 404:
                return None
            raise

    # ----- polling helpers -----

    def wait_for_completion(
        self,
        prompt_id: str,
        *,
        poll_interval: float = 2.0,
        timeout: float = 600.0,
        progress_probe=None,
    ) -> dict:
        """Block until the prompt finishes; return the final history record.

        ``timeout`` is a *silence* deadline: it resets every time
        ``progress_probe()`` returns a value different from the previous one,
        so a job that keeps reporting forward progress can run indefinitely
        and only a server silent for ``timeout`` seconds aborts. When
        ``progress_probe`` is None it degrades to a wall-clock deadline.

        Polls ``get_history``. Quick transient blips are retried inside
        ``_request``; on top of that the loop itself absorbs 429/5xx poll
        failures (a rate-limit storm, or a plain 500 the in-request layer
        never retries) with exponential backoff — base 2s, cap 60s, jitter,
        ``Retry-After`` honored — and only ``_MAX_POLL_FAILURES`` CONSECUTIVE
        failures re-raise to the caller's error path. A failed poll says
        nothing about the job, which is usually still running. A little
        jitter on the interval keeps concurrent waiters from synchronizing
        into request bursts that trip cloud rate limits.
        """
        sentinel = object()
        last_signal = sentinel
        last_change = time.time()
        consecutive_failures = 0
        while True:
            try:
                record = self.get_history(prompt_id)
            except HTTPError as e:
                if e.status != 429 and not (500 <= e.status < 600):
                    raise
                consecutive_failures += 1
                if consecutive_failures >= _MAX_POLL_FAILURES:
                    raise
                time.sleep(self._poll_retry_delay(e, consecutive_failures - 1))
                continue
            consecutive_failures = 0
            if record and _looks_done(record):
                return record
            if progress_probe is not None:
                try:
                    signal = progress_probe()
                except Exception:  # noqa: BLE001 — a flaky probe must not abort the wait
                    signal = last_signal
                if signal != last_signal:
                    last_signal = signal
                    last_change = time.time()
            if time.time() - last_change >= timeout:
                raise TimeoutError(f"workflow {prompt_id} reported no progress for {timeout}s")
            nap = poll_interval + random.uniform(0, min(poll_interval, 1.0) * 0.5)
            time.sleep(nap)

    # ----- output URL helpers -----

    def view_url(self, image_info: dict) -> str:
        """Build a fetchable /view URL for an output image record."""
        params = {
            "filename": image_info.get("filename", ""),
            "subfolder": image_info.get("subfolder", ""),
            "type": image_info.get("type", "output"),
        }
        # Use the target's path_prefix so cloud goes to /api/view and local to /view.
        return f"{self.target.url('view')}?{urllib.parse.urlencode(params)}"

    def extract_outputs(self, record: dict) -> list[dict]:
        """Flatten a node-keyed history record into one dict per artifact.

        Each entry is ``{"node_id", "url", "filename", "type"}`` — the node
        association that flat URL lists drop. The record half of the flatten
        is the pure module-level :func:`extract_output_entries`; this method
        adds the fetchable ``url`` (which needs the client's Target).
        """
        return [
            {
                "node_id": entry["node_id"],
                "url": self.view_url(entry),
                "filename": entry["filename"],
                "type": entry["type"],
            }
            for entry in extract_output_entries(record)
        ]

    def extract_output_urls(self, record: dict) -> list[str]:
        return [o["url"] for o in self.extract_outputs(record)]


def extract_output_entries(record: dict) -> list[dict]:
    """Flatten a node-keyed history record into one entry per artifact —
    ``{"node_id", "filename", "subfolder", "type"}`` (all strings).

    Pure function, no Target needed: consumers that already hold output URLs
    (e.g. ``comfy download`` reading a state file) can join them back to
    producing nodes on the (filename, subfolder, type) triple — the same one
    :meth:`Client.view_url` encodes as query params. Ordering is stable:
    record ``outputs`` insertion order, then media-key order, then item order.
    """
    results: list[dict] = []
    outputs = record.get("outputs") or {}
    if not isinstance(outputs, dict):
        return results
    for node_id, node_output in outputs.items():
        if not isinstance(node_output, dict):
            continue
        for key in ("images", "gifs", "videos", "audio", "files"):
            items = node_output.get(key) or []
            if not isinstance(items, list):
                continue
            for item in items:
                if isinstance(item, dict) and "filename" in item:
                    results.append(
                        {
                            "node_id": str(node_id),
                            "filename": str(item.get("filename", "")),
                            "subfolder": str(item.get("subfolder", "")),
                            "type": str(item.get("type", "output")),
                        }
                    )
    return results


def _group_outputs(outputs: list[dict], item_map: dict | None) -> tuple[dict[str, list[str]], dict[str, list[str]]]:
    """Group ``Client.extract_outputs`` entries by node and by foreach item.

    Returns ``(outputs_by_node, outputs_by_item)``. ``item_map`` is the
    blueprint compose map ``{item: {"nodes": [ids], "save_node": id, …}}``
    stashed on the job state; a node belongs to an item when it appears in
    the item's ``nodes`` list or is its ``save_node``. When ``item_map`` is
    falsy, ``outputs_by_item`` is ``{}``. Items that produced nothing keep an
    explicit empty list — a pruned branch should be visible, not absent.
    URL ordering follows ``outputs`` ordering in both groupings.
    """
    by_node: dict[str, list[str]] = {}
    by_item: dict[str, list[str]] = {}

    node_to_item: dict[str, str] = {}
    if item_map:
        for item_id, entry in item_map.items():
            by_item[str(item_id)] = []
            if not isinstance(entry, dict):
                continue
            members = list(entry.get("nodes") or [])
            save_node = entry.get("save_node")
            if save_node is not None:
                members.append(save_node)
            for node_id in members:
                node_to_item[str(node_id)] = str(item_id)

    for entry in outputs:
        if not isinstance(entry, dict):
            continue
        node_id = entry.get("node_id")
        url = entry.get("url")
        if node_id is None or not url:
            continue
        node_id = str(node_id)
        by_node.setdefault(node_id, []).append(url)
        item = node_to_item.get(node_id)
        if item is not None:
            by_item[item].append(url)

    return by_node, by_item


def _looks_done(record: dict) -> bool:
    status = record.get("status") or record.get("execution_status") or {}
    if isinstance(status, dict):
        if status.get("completed") is True:
            return True
        if status.get("status_str") in {"success", "error", "failed"}:
            return True
    # Older deployments don't surface a structured status — presence of
    # outputs is enough to call it done.
    outputs = record.get("outputs") or {}
    return bool(outputs)
