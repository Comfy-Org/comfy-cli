"""Shared cloud HTTP/URL error → structured-envelope mapping.

``comfy workflow`` and ``comfy jobs`` both talk to Comfy Cloud over ``urllib``
and must turn transport failures into the same structured error envelopes. This
helper is the single source of truth for that mapping so the two call sites
can't drift — historically only ``workflow`` mapped 401/403 to
``cloud_unauthorized`` (with an actionable ``comfy cloud login`` hint) while
``jobs`` emitted a generic ``cloud_http_error``; routing both through here
closes that gap.

The 404 branch is deliberately caller-parameterized: ``workflow`` surfaces
``workflow_not_found`` and ``jobs`` surfaces ``prompt_not_found``, each with its
own message/hint. Everything else — the bounded body read, the
401/403 → ``cloud_unauthorized`` branch, the 429 → ``cloud_rate_limited``
branch, the generic ``cloud_http_error``, and the ``URLError``/``OSError``
network hint — is shared. :func:`emit_status_error` is the 402/429-vs-generic split
on its own, for callers (``run``, ``cloud_http``) that map the other statuses
with their own vocabulary.
"""

from __future__ import annotations

import json
import re
import urllib.error

import typer

# Cap on the server error body we surface in ``details.body``. ``base_url`` is
# env-configurable, so a hostile or misbehaving endpoint must not be able to
# OOM the CLI with an unbounded error response.
_MAX_ERROR_BODY_BYTES = 1000
# Cap on the 403 body read for ``insufficient_scope`` detection. A JSON body
# cut at ``_MAX_ERROR_BODY_BYTES`` no longer parses, so detection reads more
# than ``details.body`` shows — but still a bounded amount.
_MAX_SCOPE_BODY_BYTES = 64 * 1024

# A 401 is unambiguously an authentication failure. A 403 is not — it is also
# how the server denies a forbidden resource, a quota, or an already-finished
# job, so pointing the user straight at re-login would be a misleading
# remediation. Send them to the server's own explanation instead.
_UNAUTHORIZED_HINTS = {
    401: "re-run `comfy cloud login`",
    403: "re-run `comfy cloud login` if your session expired; otherwise check `details.body` — the server may be denying access to this resource",
}


_INSUFFICIENT_SCOPE_MESSAGE = (
    "Your Comfy Cloud login predates a permission change; run `comfy cloud login` to re-authorize"
)
# One piece of a WWW-Authenticate field (RFC 7235 §2.1): an auth-param
# ``name = "quoted"`` / ``name = token``, or else a bare token, which starts a
# new challenge. Quoted values are consumed whole, so the scan never reads
# inside one such as ``error_description``.
_CHALLENGE_PART = re.compile(
    r"([!#$%&'*+\-.^_`|~0-9A-Za-z]+)"
    r'(?P<param>\s*=\s*(?:"(?P<quoted>(?:[^"\\]|\\.)*)"|(?P<token>[^\s,"]*)))?'
)
# The plain-text body the scope middleware writes: ``insufficient_scope`` or
# ``insufficient_scope: <scope>`` and nothing else. Anchored at both ends so a
# body that merely mentions the token in prose is not mistaken for one.
_BODY_SCOPE = re.compile(r"\s*insufficient_scope(?:\s*:\s*(?P<scope>[^\r\n]*?))?\s*\Z")
# Structured fields a JSON error body may carry the error in.
_BODY_SCOPE_FIELDS = ("message", "error", "code", "type")
# ``required_scope`` comes from the server, and ``base_url`` is configurable,
# so bound it and keep only RFC 6749 §3.3 scope-token characters.
_SCOPE_TOKEN = re.compile(r"[\x21\x23-\x5B\x5D-\x7E]+")
_MAX_REQUIRED_SCOPE_CHARS = 200


def _www_authenticate_values(headers) -> list[str]:
    """Every ``WWW-Authenticate`` value as ``str``; ``[]`` on any odd headers object.

    ``headers`` may also be the values themselves (``HTTPError.www_authenticate``).
    """
    if headers is None:
        return []
    if isinstance(headers, (list, tuple)):
        return [v for v in headers if isinstance(v, str) and v]
    try:
        get_all = getattr(headers, "get_all", None)
        values = get_all("WWW-Authenticate") if callable(get_all) else [headers.get("WWW-Authenticate")]
    except Exception:  # noqa: BLE001
        return []
    return [v for v in (values or []) if isinstance(v, str) and v]


def _challenges(value: str) -> list[tuple[str, dict[str, str]]]:
    """Split one ``WWW-Authenticate`` value into ``(scheme, params)`` challenges."""
    challenges: list[tuple[str, dict[str, str]]] = []
    for m in _CHALLENGE_PART.finditer(value):
        if m.group("param") is None:
            challenges.append((m.group(1).lower(), {}))
        elif challenges:
            raw = m.group("quoted")
            param = re.sub(r"\\(.)", r"\1", raw) if raw is not None else m.group("token")
            challenges[-1][1].setdefault(m.group(1).lower(), param)
    return challenges


def _bearer_scope_challenge(values: list[str]) -> tuple[bool, str | None]:
    """Whether a Bearer challenge says ``error=insufficient_scope``, and the ``scope`` that challenge names."""
    for value in values:
        for scheme, params in _challenges(value):
            if scheme == "bearer" and params.get("error") == "insufficient_scope":
                return True, params.get("scope")
    return False, None


def _body_scope_error(body: str) -> tuple[bool, str | None]:
    """Whether the body is an ``insufficient_scope`` error, and the scope it names.

    A JSON body is judged by its structured fields only; a plain body must be
    exactly the documented ``insufficient_scope[: <scope>]`` form.
    """
    candidates = [body]
    try:
        parsed = json.loads(body)
    except ValueError:
        parsed = None
    if isinstance(parsed, dict):
        candidates = [parsed[k] for k in _BODY_SCOPE_FIELDS if isinstance(parsed.get(k), str)]
        nested = parsed.get("error")
        if isinstance(nested, dict):
            candidates += [nested[k] for k in _BODY_SCOPE_FIELDS if isinstance(nested.get(k), str)]
    elif parsed is not None:
        candidates = [parsed] if isinstance(parsed, str) else []
    for text in candidates:
        m = _BODY_SCOPE.match(text)
        if m:
            return True, m.group("scope")
    return False, None


def _clean_scope(scope: str | None) -> str | None:
    if not scope:
        return None
    tokens = _SCOPE_TOKEN.findall(scope)
    if not tokens or " ".join(tokens) != " ".join(scope.split()):
        return None
    cleaned = " ".join(tokens)
    return cleaned if len(cleaned) <= _MAX_REQUIRED_SCOPE_CHARS else None


def insufficient_scope_error(status: int, body: str | None, headers=None, details: dict | None = None) -> dict | None:
    """The envelope fields for a 403 ``insufficient_scope``, or ``None`` for any other response.

    Comfy Cloud answers a token whose grant lacks a route's scope with
    ``403`` + ``WWW-Authenticate: Bearer error="insufficient_scope", scope="…"``
    (RFC 6750 §3.1) and an ``insufficient_scope: <scope>`` body. Refreshing
    cannot widen a grant, so the only fix is a new authorize flow — the
    envelope says so instead of the generic 403 hint. It never opens a
    browser: the caller may be non-interactive.
    """
    if status != 403:
        return None
    header_hit, header_scope = _bearer_scope_challenge(_www_authenticate_values(headers))
    body_hit, body_scope = _body_scope_error(body if isinstance(body, str) else "")
    if not header_hit and not body_hit:
        return None
    out_details = {**(details or {}), "status": 403, "reason": "insufficient_scope"}
    required_scope = _clean_scope(header_scope) or _clean_scope(body_scope)
    if required_scope:
        out_details["required_scope"] = required_scope
    return {
        "code": "cloud_unauthorized",
        "message": _INSUFFICIENT_SCOPE_MESSAGE,
        "hint": (
            "run `comfy cloud login` to re-authorize; refreshing the existing session cannot add the permission."
            " If you set COMFY_CLOUD_SCOPES or the `cloud_scopes` config, add the required scope to it first"
        ),
        "details": out_details,
    }


def _read_error_body(e: urllib.error.HTTPError) -> str:
    """Best-effort read of a bounded slice of the server's error body.

    A read that raises (reset or truncated stream) must not pre-empt the
    structured envelope this module exists to emit, so failures degrade to an
    empty body rather than escaping as an unhandled traceback.
    """
    try:
        return (e.read(_MAX_ERROR_BODY_BYTES) or b"").decode("utf-8", "replace")
    except Exception:
        return ""


def read_unauthorized_body(e: urllib.error.HTTPError) -> tuple[str, str]:
    """``(detection_body, display_body)`` for a 401/403, from one bounded read.

    Scope detection gets up to ``_MAX_SCOPE_BODY_BYTES`` so a structured
    ``insufficient_scope`` field past the display cap is still seen; only the
    ``details.body`` slice is capped at ``_MAX_ERROR_BODY_BYTES``. Read
    failures degrade to empty bodies, as in ``_read_error_body``.
    """
    try:
        raw = e.read(_MAX_SCOPE_BODY_BYTES) or b""
    except Exception:
        raw = b""
    return raw.decode("utf-8", "replace"), raw[:_MAX_ERROR_BODY_BYTES].decode("utf-8", "replace")


def retry_after_from_headers(headers) -> float | None:
    """The server's ``Retry-After`` (delta-seconds) from a urllib error's headers."""
    from comfy_cli.comfy_client import _parse_retry_after

    return _parse_retry_after(headers)


# How a 429 hint ends when the caller has nothing more specific to say. A 429
# means the server is throttling; it does not by itself prove the request had
# no effect, so a request that creates or changes something is checked before
# it is repeated.
_DEFAULT_RATE_LIMITED_NEXT_STEP = (
    "retry; if the request creates or changes something, check first that it did not already go through"
)


# A plan refusal (free generations used up, subscription required, a partner
# node or model that needs a paid plan) is HTTP 402 with a typed JSON body
# ({"error": {"type": ..., "message": ...}}). Waiting and retrying cannot change
# it, so it is ``cloud_payment_required``, never ``cloud_rate_limited``.
#
# Legacy: the cloud submit endpoint used to send these refusals as 429, telling
# them apart from throttling only by ``error.type``. Until every deployment
# sends 402, a 429 whose type is listed here is still a refusal. QUEUE_LIMIT
# (the workspace's queue is full) and FREE_TIER_UNAVAILABLE (temporarily off)
# are genuine backpressure and stay ``cloud_rate_limited``. Delete this set and
# the 429 branch in ``emit_status_error`` once the server no longer sends them.
_LEGACY_429_PLAN_REFUSAL_TYPES = frozenset(
    {
        "FREE_TIER_EXHAUSTED",
        "FREE_TIER_NOT_ALLOWED",
        "PAYMENT_REQUIRED",
        "CLOUD_SUBSCRIPTION_REQUIRED",
        "PARTNER_NODE_PAYMENT_REQUIRED",
        "MODEL_PAYMENT_REQUIRED",
    }
)


def _typed_error(details: dict) -> tuple[str | None, str | None]:
    """``(error.type, error.message)`` from a JSON error body; ``None`` for whatever is missing or malformed."""
    body = details.get("body")
    if not isinstance(body, str) or not body.strip():
        return None, None
    try:
        err = json.loads(body).get("error")
    except (ValueError, AttributeError):
        return None, None
    if not isinstance(err, dict):
        return None, None
    # A malformed body must degrade, not raise: an unhashable `type` ([] or {})
    # cannot be tested for set membership.
    type_ = err.get("type")
    message = err.get("message")
    return (
        type_ if isinstance(type_, str) and type_ else None,
        message if isinstance(message, str) and message else None,
    )


def payment_required_error(
    operation: str, status: int, reason: str | None, server_message: str | None, details: dict
) -> dict:
    """The ``cloud_payment_required`` envelope fields for a plan refusal (402, or a legacy typed 429)."""
    label = f"HTTP {status}, {reason}" if reason else f"HTTP {status}"
    out_details = {**details, "status": status}
    if reason:
        out_details["reason"] = reason
    return {
        "code": "cloud_payment_required",
        "message": (
            f"Comfy Cloud refused the {operation} ({label}): {server_message or 'the account plan does not allow this'}"
        ),
        "hint": (
            "this is not throttling, so retrying will not help and nothing was queued: tell the user the "
            "server's message; running this needs a plan that allows it"
        ),
        "details": out_details,
    }


def emit_status_error(
    renderer,
    *,
    status: int,
    retry_after: float | None,
    operation: str,
    message: str,
    hint: str | None,
    details: dict,
    rate_limited_next_step: str = _DEFAULT_RATE_LIMITED_NEXT_STEP,
    scope_body: str | None = None,
    www_authenticate: tuple[str, ...] | str | None = None,
) -> None:
    """Emit the envelope for a cloud HTTP status that has no caller-specific code.

    A 402 is a plan refusal: it gets the non-retryable
    ``cloud_payment_required`` with the server's message and ``error.type``. So
    does a 429 whose type is a legacy plan refusal
    (``_LEGACY_429_PLAN_REFUSAL_TYPES``). Any other 429 is throttling, which
    says nothing about whether the request is valid, so the generic ``cloud_http_error`` (whose callers' hints say "check the
    workflow is valid", "check `details.body`") would send an agent off to
    rewrite a request that may be fine. It gets ``cloud_rate_limited`` and a
    wait-then-next-step hint instead, with the server's ``Retry-After`` in
    ``details.retry_after`` when it sent one. Every other status keeps the
    caller's ``cloud_http_error`` envelope exactly as given.
    ``rate_limited_next_step`` finishes the 429 hint for a caller that knows
    more (``run``'s submit: check the job list before re-running; its poll: the
    job exists, so follow it rather than re-running).
    ``scope_body`` / ``www_authenticate`` let a caller whose ``details`` omit
    the body (``run``'s poll) still have a 403 ``insufficient_scope`` detected.
    """
    scope_error = insufficient_scope_error(
        status,
        scope_body if scope_body is not None else details.get("body"),
        (www_authenticate,) if isinstance(www_authenticate, str) else www_authenticate,
        details=details,
    )
    if scope_error is not None:
        renderer.error(**scope_error)
        return
    if status == 402:
        renderer.error(**payment_required_error(operation, 402, *_typed_error(details), details))
        return
    if status != 429:
        renderer.error(code="cloud_http_error", message=message, hint=hint, details=details)
        return
    reason, server_message = _typed_error(details)
    if reason in _LEGACY_429_PLAN_REFUSAL_TYPES:
        renderer.error(**payment_required_error(operation, 429, reason, server_message, details))
        return
    renderer.error(**rate_limited_error(operation, retry_after, details, next_step=rate_limited_next_step))


def rate_limited_error(
    operation: str,
    retry_after: float | None,
    details: dict,
    *,
    next_step: str = _DEFAULT_RATE_LIMITED_NEXT_STEP,
) -> dict:
    """The ``cloud_rate_limited`` envelope fields (``code``/``message``/``hint``/``details``).

    Shared by :func:`emit_status_error` and callers that record the error
    rather than render it (the job state file), so both carry the same shape:
    ``details.status`` is 429 and ``details.retry_after`` holds the server's
    ``Retry-After`` when it sent one.
    """
    rate_details = {**details, "status": 429}
    if retry_after is not None:
        rate_details["retry_after"] = int(retry_after) if float(retry_after).is_integer() else retry_after
        wait = f"wait {rate_details['retry_after']}s (`details.retry_after`)"
    else:
        wait = "wait a few seconds"
    return {
        "code": "cloud_rate_limited",
        "message": f"Comfy Cloud rate-limited the {operation} request (HTTP 429): too many requests",
        "hint": f"{wait}, then {next_step}",
        "details": rate_details,
    }


def handle_cloud_http_error(
    renderer,
    e: Exception,
    *,
    operation: str,
    not_found_code: str,
    not_found_message: str,
    not_found_hint: str,
    id_label: str,
    resource_id: str | None = None,
    not_found_details: dict | None = None,
) -> typer.Exit:
    """Map a cloud HTTP/URL failure to a structured error envelope.

    Emits the error via ``renderer.error`` and returns a ``typer.Exit`` for the
    caller to ``raise ... from e`` so the original traceback is preserved.

    Args:
        operation: short verb naming what failed (``"get"``, ``"cancel"``, …);
            used in messages and detail payloads.
        not_found_code / not_found_message / not_found_hint: the 404 envelope,
            which differs per caller (``workflow_not_found`` vs
            ``prompt_not_found``).
        id_label: detail key for ``resource_id`` (``"workflow_id"`` /
            ``"prompt_id"``).
        resource_id: the id being operated on, or ``None`` for id-less
            operations (e.g. ``list``).
        not_found_details: extra ``details`` keys for the 404 envelope only.
    """
    id_detail = {id_label: resource_id}
    if isinstance(e, urllib.error.HTTPError):
        if e.code == 404:
            renderer.error(
                code=not_found_code,
                message=not_found_message,
                hint=not_found_hint,
                details={**id_detail, "operation": operation, **(not_found_details or {})},
            )
        elif e.code in (401, 403):
            scope_body, body = read_unauthorized_body(e)
            details = {"status": e.code, "body": body, "operation": operation, **id_detail}
            scope_error = insufficient_scope_error(e.code, scope_body, getattr(e, "headers", None), details=details)
            if scope_error is not None:
                renderer.error(**scope_error)
            else:
                renderer.error(
                    code="cloud_unauthorized",
                    message=f"HTTP {e.code} during {operation}",
                    hint=_UNAUTHORIZED_HINTS[e.code],
                    details=details,
                )
        else:
            emit_status_error(
                renderer,
                status=e.code,
                retry_after=retry_after_from_headers(getattr(e, "headers", None)),
                operation=operation,
                message=f"HTTP {e.code} during {operation}",
                hint="check `details.body` for the server's message",
                details={"status": e.code, "body": _read_error_body(e), "operation": operation, **id_detail},
            )
    else:
        renderer.error(
            code="cloud_http_error",
            message=f"{operation} failed: {e}",
            hint="check network / `comfy cloud whoami`",
        )
    return typer.Exit(code=1)
