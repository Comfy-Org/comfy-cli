"""Shared environment-variable backend resolver for ComfyUI host/port.

Supports four env vars (highest to lowest precedence within the env layer):
  COMFY_BACKEND=host:port     (combined, colon-separated)
  COMFY_HOST_PORT=host:port   (combined, same as COMFY_BACKEND)
  COMFY_HOST + COMFY_PORT     (separate)

CLI flags always take precedence over env vars; this helper is a fallback.
"""
from __future__ import annotations

import os
import re
from typing import Optional

# IPv6 address detection: contains multiple colons or starts with [
_IPV6_RE = re.compile(r"^\[.+\]$|^([0-9a-fA-F]{0,4}:){2,}")


def get_backend_from_env() -> tuple[Optional[str], Optional[int]]:
    """Resolve host/port from environment variables.

    Returns (host, port) where either may be None if not set.
    Handles IPv6 bracketed hosts (e.g. ``[::1]:8188``).
    Logs a warning on invalid port values instead of silently swallowing.
    """
    be = os.environ.get("COMFY_BACKEND") or os.environ.get("COMFY_HOST_PORT")
    if be:
        b = be.replace("http://", "").replace("https://", "").strip("/")
        host, port = _split_host_port(b)
        if port is not None and port is not None and not _valid_port(port):
            import logging
            logging.getLogger(__name__).warning(
                "Invalid port %r in COMFY_BACKEND/COMFY_HOST_PORT, ignoring", port
            )
            return host, None
        return host, port

    h = os.environ.get("COMFY_HOST")
    p = os.environ.get("COMFY_PORT")
    if h or p:
        port: Optional[int] = None
        if p:
            try:
                port = int(p)
            except ValueError:
                import logging
                logging.getLogger(__name__).warning(
                    "Invalid port %r in COMFY_PORT, ignoring", p
                )
                return h, None
            if not _valid_port(port):
                logging.getLogger(__name__).warning(
                    "Port %d out of range in COMFY_PORT, ignoring", port
                )
                return h, None
        return h, port

    return None, None


def _split_host_port(s: str) -> tuple[Optional[str], Optional[int]]:
    """Split a host:port string, handling IPv6 bracketed hosts.

    Examples:
      "127.0.0.1:8188"   -> ("127.0.0.1", 8188)
      "[::1]:8188"       -> ("[::1]", 8188)
      "[::1]"            -> ("[::1]", None)
      "localhost:8188"   -> ("localhost", 8188)
      "::1"              -> ("::1", None)  (bare IPv6, no port)
    """
    # Bracketed IPv6: [host]:port or [host]
    if s.startswith("["):
        end = s.find("]")
        if end == -1:
            return s, None  # malformed, return as-is
        host = s[:end + 1]  # keep brackets
        rest = s[end + 1:]
        if rest.startswith(":"):
            try:
                return host, int(rest[1:])
            except ValueError:
                return host, None
        return host, None

    # Bare IPv6 (multiple colons, no brackets) — no port
    if s.count(":") > 1:
        return s, None

    # Normal host:port
    if ":" in s:
        h, p = s.rsplit(":", 1)
        try:
            return h, int(p)
        except ValueError:
            return h, None
    return s, None


def _valid_port(port: int) -> bool:
    """Check if a port number is in valid range."""
    return 1 <= port <= 65535
