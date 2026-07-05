"""Shared network-target helpers for CQL loaders."""

from __future__ import annotations

import ipaddress

__all__ = ["is_loopback_host"]


def is_loopback_host(hostname: str) -> bool:
    """True if *hostname* refers to a loopback interface.

    Expects an already-parsed hostname (e.g. ``urllib.parse.urlsplit(url).hostname``,
    which strips IPv6 brackets). Matches the literal name ``localhost`` and any
    address ``ipaddress`` classifies as loopback (127.0.0.0/8, ::1). Used to refuse
    SSRF fetches to non-loopback targets in local mode.

    NOTE: this deliberately includes the ipaddress fallback, so it must NOT be used
    for the require-HTTPS gates in comfy_client._assert_safe_url or
    oauth._assert_https_or_loopback — those need exact-literal membership.
    """
    h = (hostname or "").strip().lower()
    if h == "localhost":
        return True
    try:
        return ipaddress.ip_address(h).is_loopback
    except ValueError:
        return False
