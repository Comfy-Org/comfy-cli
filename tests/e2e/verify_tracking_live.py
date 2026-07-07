#!/usr/bin/env python3
"""Live end-to-end verification that CLI telemetry actually reaches PostHog.

Unlike the unit suites (which mock the providers / SDK clients), this fires a
real event through the production ``track_event`` path and then reads it back
out of PostHog via the query API. It is the only check that proves the network
round-trip works — so it is opt-in and never runs in CI.

Required environment (fail-fast: missing any of these aborts):

    POSTHOG_PERSONAL_API_KEY   personal API key with project *read* scope
                               (NOT the phc_* project write key)
    POSTHOG_PROJECT_ID         numeric project id
    POSTHOG_QUERY_HOST         host that serves /api/projects/... query API,
                               e.g. https://us.posthog.com (the ingestion
                               proxy may not expose the query API)

Optional:
    POSTHOG_VERIFY_TIMEOUT     seconds to wait for ingestion (default 120)

Usage:
    POSTHOG_PERSONAL_API_KEY=phx_... POSTHOG_PROJECT_ID=12345 \
    POSTHOG_QUERY_HOST=https://us.posthog.com \
    python tests/e2e/verify_tracking_live.py
"""

from __future__ import annotations

import json
import os
import sys
import time
import urllib.error
import urllib.request
import uuid

SMOKETEST_EVENT = "cli_telemetry_smoketest"
_POLL_INTERVAL_SECONDS = 5


def _die(message: str) -> None:
    """Fail fast with a clear message and non-zero exit."""
    print(f"FAIL: {message}", file=sys.stderr)
    raise SystemExit(1)


def _require_env(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        _die(f"missing required env var {name} (see module docstring)")
    return value


def _send_smoketest_event(nonce: str) -> tuple[str, str]:
    """Fire one event through the real track_event path, PostHog pipe only.

    Returns (distinct_id, event_name). Raises (via _die) if telemetry is
    suppressed or no PostHog provider is available.
    """
    import comfy_cli.tracking as tracking

    if tracking._telemetry_disabled_by_env():
        _die("telemetry is opted out via DO_NOT_TRACK / COMFY_NO_TELEMETRY — unset it to verify")

    posthog_providers = [p for p in tracking.PROVIDERS if isinstance(p, tracking.PostHogProvider) and p.enabled]
    if not posthog_providers:
        _die("no enabled PostHog provider — check POSTHOG_API_KEY token")

    # Restrict the fan-out to PostHog so the smoke test doesn't pollute Mixpanel,
    # and force a session-only opt-in so we send without mutating persisted consent.
    tracking.PROVIDERS = posthog_providers
    tracking._session_only_tracking = True
    distinct_id = tracking._ensure_user_id()

    tracking.track_event(SMOKETEST_EVENT, {"nonce": nonce})
    tracking._flush_all_providers()
    return distinct_id, SMOKETEST_EVENT


def _build_query(nonce: str) -> dict:
    hogql = (
        "SELECT event, timestamp, properties.nonce AS nonce "
        "FROM events "
        f"WHERE event = '{SMOKETEST_EVENT}' AND properties.nonce = '{nonce}' "
        "AND timestamp > now() - INTERVAL 1 HOUR "
        "LIMIT 1"
    )
    return {"query": {"kind": "HogQLQuery", "query": hogql}}


def _query_posthog(host: str, project_id: str, api_key: str, body: dict) -> list:
    url = f"{host.rstrip('/')}/api/projects/{project_id}/query/"
    request = urllib.request.Request(
        url,
        data=json.dumps(body).encode("utf-8"),
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        detail = e.read().decode("utf-8", "replace")[:300]
        _die(f"PostHog query API returned HTTP {e.code}: {detail}")
    except urllib.error.URLError as e:
        _die(f"could not reach PostHog query API at {url}: {e.reason}")
    return payload.get("results", [])


def _poll_for_event(host: str, project_id: str, api_key: str, nonce: str, timeout: int) -> list | None:
    """Poll the query API until the nonce shows up or the deadline passes."""
    body = _build_query(nonce)
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        results = _query_posthog(host, project_id, api_key, body)
        if results:
            return results[0]
        remaining = int(deadline - time.monotonic())
        print(f"  ...not yet in PostHog, retrying ({remaining}s left)")
        time.sleep(_POLL_INTERVAL_SECONDS)
    return None


def main() -> None:
    api_key = _require_env("POSTHOG_PERSONAL_API_KEY")
    project_id = _require_env("POSTHOG_PROJECT_ID")
    query_host = _require_env("POSTHOG_QUERY_HOST")
    timeout = int(os.environ.get("POSTHOG_VERIFY_TIMEOUT", "120"))

    nonce = uuid.uuid4().hex
    print(f"Sending {SMOKETEST_EVENT} with nonce={nonce} ...")
    distinct_id, event_name = _send_smoketest_event(nonce)
    print(f"  dispatched (distinct_id={distinct_id}); waiting for ingestion (up to {timeout}s)")

    hit = _poll_for_event(query_host, project_id, api_key, nonce, timeout)
    if hit is None:
        _die(f"event with nonce={nonce} never appeared within {timeout}s — delivery NOT confirmed")

    print(f"PASS: telemetry round-trip confirmed — PostHog returned {hit}")


if __name__ == "__main__":
    main()
