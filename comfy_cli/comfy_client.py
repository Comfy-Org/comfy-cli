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
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Any

from comfy_cli.target import Target

_LOOPBACK_HOSTS = {"localhost", "127.0.0.1", "::1", "[::1]"}

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


class HTTPError(Exception):
    """Server returned a non-2xx response."""

    def __init__(self, status: int, message: str, body: str = ""):
        # Redact before super().__init__ so str(self) is safe to log.
        message = _redact(message)
        body = _redact(body)
        super().__init__(f"HTTP {status}: {message}")
        self.status = status
        self.message = message
        self.body = body


class Unauthenticated(Exception):
    """Target needs auth but no valid session is present."""


class _NoRedirectHandler(urllib.request.HTTPRedirectHandler):
    """Refuse to follow redirects.

    Following redirects with `Authorization: Bearer …` in flight risks
    replaying the token at the redirect target. ComfyUI gateways don't issue
    redirects under normal operation; if we ever see one it's almost certainly
    a misconfiguration or attack. Surface as a clear HTTPError instead.
    """

    def http_error_301(self, req, fp, code, msg, headers):
        raise urllib.error.HTTPError(req.full_url, code, "redirect refused", headers, fp)

    http_error_302 = http_error_303 = http_error_307 = http_error_308 = http_error_301


_OPENER = urllib.request.build_opener(_NoRedirectHandler())


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

    def __init__(self, target: Target, *, timeout: float = 30.0):
        self.target = target
        self.timeout = timeout
        if target.is_cloud and not (target.auth_token or target.api_key):
            raise Unauthenticated(
                "cloud target requires credentials — run `comfy cloud login` or set COMFY_CLOUD_API_KEY"
            )

    # ----- low-level -----

    def _try_refresh_token(self) -> bool:
        """Attempt to refresh the OAuth token. Returns True if successful."""
        if not self.target.is_cloud or not self.target.auth_token:
            return False
        # Only refresh OAuth tokens, not API keys
        if self.target.api_key:
            return False
        try:
            from comfy_cli.auth import store as auth_store
            from comfy_cli.cloud import CLIENT_ID, get_resource_url
            from comfy_cli.cloud.oauth import refresh_tokens

            session = auth_store.get_cloud_session()
            if session is None or not session.refresh_token:
                return False

            new_tokens = refresh_tokens(
                base_url=session.base_url,
                client_id=session.client_id or CLIENT_ID,
                refresh_token=session.refresh_token,
                resource=session.resource or get_resource_url(),
            )

            # Persist the refreshed tokens
            auth_store.save_cloud_session(
                base_url=session.base_url,
                resource=session.resource,
                client_id=session.client_id,
                scope=session.scope,
                access_token=new_tokens.access_token,
                refresh_token=new_tokens.refresh_token or session.refresh_token,
                token_type=new_tokens.token_type,
                expires_at=new_tokens.expires_at,
            )

            # Update the target's auth token in-place for subsequent requests.
            # Target is frozen, so we need to work around it.
            object.__setattr__(self.target, "auth_token", new_tokens.access_token)
            return True
        except Exception:  # noqa: BLE001 — refresh is best-effort
            return False

    def _request(
        self,
        method: str,
        path_parts: tuple[str, ...],
        *,
        body: dict | None = None,
        timeout: float | None = None,
        _retried: bool = False,
    ) -> Any:
        url = self.target.url(*path_parts)
        # Only enforce https-or-loopback when we're carrying a bearer token.
        if self.target.is_cloud and (self.target.auth_token or self.target.api_key):
            _assert_safe_url(url)
        data = json.dumps(body).encode("utf-8") if body is not None else None
        req = urllib.request.Request(url, data=data, method=method)
        req.add_header("Accept", "application/json")
        if data is not None:
            req.add_header("Content-Type", "application/json")
        # Cloud auth: prefer X-API-Key when set (easier to paste-and-test),
        # fall back to OAuth Bearer. Only attached on cloud targets so a stray
        # auth_token on a local target can't leak credentials to a plaintext
        # server.
        if self.target.is_cloud:
            if self.target.api_key:
                req.add_header("X-API-Key", self.target.api_key)
            elif self.target.auth_token:
                req.add_header("Authorization", f"Bearer {self.target.auth_token}")
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
                return self._request(method, path_parts, body=body, timeout=timeout, _retried=True)
            raise HTTPError(e.code, e.reason or "HTTP error", body_text) from e

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
        payload: dict[str, Any] = {"prompt": workflow, "client_id": client_id}
        merged_extra: dict[str, Any] = dict(extra_data or {})
        # Partner-API nodes (BFL, Gemini, Bria, ByteDance, etc.) read the
        # caller's comfy.org credential out of extra_data. We mirror what the
        # web UI sends: api_key_comfy_org for the X-API-Key path,
        # auth_token_comfy_org for the OAuth Bearer path.
        if self.target.is_cloud:
            if self.target.api_key:
                merged_extra.setdefault("api_key_comfy_org", self.target.api_key)
            elif self.target.auth_token:
                merged_extra.setdefault("auth_token_comfy_org", self.target.auth_token)
        if merged_extra:
            payload["extra_data"] = merged_extra
        resp = self._request("POST", ("prompt",), body=payload, timeout=timeout)
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

    def wait_for_completion(self, prompt_id: str, *, poll_interval: float = 1.0, timeout: float = 600.0) -> dict:
        """Block until the prompt finishes; return the final history record."""
        deadline = time.time() + timeout
        while True:
            record = self.get_history(prompt_id)
            if record and _looks_done(record):
                return record
            if time.time() >= deadline:
                raise TimeoutError(f"workflow {prompt_id} did not complete within {timeout}s")
            time.sleep(min(poll_interval, max(deadline - time.time(), 0.05)))

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

    def extract_output_urls(self, record: dict) -> list[str]:
        urls: list[str] = []
        outputs = record.get("outputs") or {}
        if not isinstance(outputs, dict):
            return urls
        for node_output in outputs.values():
            if not isinstance(node_output, dict):
                continue
            for key in ("images", "gifs", "videos", "audio", "files"):
                items = node_output.get(key) or []
                if not isinstance(items, list):
                    continue
                for item in items:
                    if isinstance(item, dict) and "filename" in item:
                        urls.append(self.view_url(item))
        return urls


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
