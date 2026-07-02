"""Shared host/port parsing + resolution for local ComfyUI commands.

Both ``comfy run`` and every ``comfy jobs`` subcommand accept a ``--host`` /
``--port`` pair (and a combined ``host[:port]`` string), fall back to the
persisted ``config.background`` server, then to ``DEFAULT_HOST`` /
``DEFAULT_PORT``. They also feed the resolved host straight into URLs like
``http://{host}:{port}/prompt`` / ``ws://{host}:{port}/ws``, so the value must
be validated (no URL-injection characters) and IPv6 literals bracketed. This
module is the single home for that logic; callers should not re-implement it.
"""

from __future__ import annotations

import typer

from comfy_cli.config_manager import ConfigManager

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8188
_UNSAFE_HOST_CHARS = frozenset("/@?#")


def validate_host(host: str) -> str:
    """Reject host values that could cause URL injection."""
    if any(c in host for c in _UNSAFE_HOST_CHARS):
        raise typer.BadParameter(f"invalid host: {host!r} (contains URL-special characters)")
    return host


def parse_host_port_arg(value: str) -> tuple[str, int | None]:
    """Split a user-typed combined ``host[:port]`` string, IPv6-aware.

    Accepts: ``'host'``, ``'host:port'``, ``'[::1]'``, ``'[::1]:8188'``, bare
    ``'::1'``. Returns ``(host, port_or_None)``. Raises ``typer.BadParameter``
    on a non-numeric port or an unterminated bracket.
    """
    v = value.strip()
    if v.startswith("["):
        end = v.find("]")
        if end == -1:
            raise typer.BadParameter(f"invalid host: {value!r} (unterminated '[')")
        host = v[1:end]
        rest = v[end + 1 :]
        if rest.startswith(":") and rest[1:]:
            return host, _to_port(rest[1:], value)
        return host, None
    if v.count(":") == 1:  # exactly one colon -> host:port
        h, p = v.split(":")
        return h, (_to_port(p, value) if p else None)
    # zero colons -> hostname only; 2+ colons -> bare IPv6 literal (no port)
    return v, None


def _to_port(s: str, original: str) -> int:
    try:
        return int(s)
    except ValueError:
        raise typer.BadParameter(f"invalid port in {original!r}: {s!r} is not a number")


def resolve_host_port(host: str | None, port: int | None) -> tuple[str, int]:
    """Fill host/port from ``config.background`` then defaults; validate and
    bracket IPv6 literals so callers building ``'http://{host}:{port}'`` get a
    well-formed URL (e.g. ``'::1'`` -> ``'[::1]'``)."""
    cfg = ConfigManager()
    bg = cfg.background
    if not host and bg is not None:
        host = bg[0]
    if not port and bg is not None:
        port = bg[1]
    h = validate_host(host or DEFAULT_HOST)
    if ":" in h and not h.startswith("["):
        h = f"[{h}]"
    return (h, int(port or DEFAULT_PORT))
