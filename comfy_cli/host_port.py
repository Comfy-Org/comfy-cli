"""Shared host/port parsing + resolution for local ComfyUI commands.

Both ``comfy run`` and every ``comfy jobs`` subcommand accept a ``--host`` /
``--port`` pair, fall back to the persisted ``config.background`` server, then
to ``DEFAULT_HOST`` / ``DEFAULT_PORT``. In addition, ``comfy run`` accepts a
combined ``host[:port]`` string (parsed via ``parse_host_port_arg``); the
``comfy jobs`` subcommands only take the separate ``--host`` / ``--port``
options. They all feed the resolved host straight into URLs like
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
    # Whitespace/control chars (notably CR/LF) never appear in a real host and
    # are the canonical header/URL-injection vectors, so reject them too.
    if any(c.isspace() or ord(c) < 0x20 or ord(c) == 0x7F for c in host):
        raise typer.BadParameter(f"invalid host: {host!r} (contains whitespace or control characters)")
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
        if rest:
            # Only a ``:port`` suffix is allowed after the bracket; anything
            # else (e.g. ``[::1]8188``) is a typo we must not silently drop.
            if not rest.startswith(":"):
                raise typer.BadParameter(f"invalid host: {value!r} (unexpected text after ']')")
            if rest[1:]:
                return _require_host(host, value), _to_port(rest[1:], value)
        return _require_host(host, value), None
    if v.count(":") == 1:  # exactly one colon -> host:port
        h, p = v.split(":")
        return _require_host(h, value), (_to_port(p, value) if p else None)
    # zero colons -> hostname only; 2+ colons -> bare IPv6 literal (no port)
    return _require_host(v, value), None


def _require_host(host: str, original: str) -> str:
    """Reject an empty host so ``:8188`` / ``[]:8188`` error instead of
    silently retargeting the request to the default/background server."""
    if not host:
        raise typer.BadParameter(f"invalid host: {original!r} (empty host)")
    return host


def _to_port(s: str, original: str) -> int:
    try:
        port = int(s)
    except ValueError:
        raise typer.BadParameter(f"invalid port in {original!r}: {s!r} is not a number")
    if not (1 <= port <= 65535):
        raise typer.BadParameter(f"invalid port in {original!r}: {port} is out of range (1-65535)")
    return port


def resolve_host_port(host: str | None, port: int | None) -> tuple[str, int]:
    """Resolve host/port by precedence — explicit flag > ``COMFY_LOCAL_URL``
    env > ``config.background`` > defaults — then validate and bracket IPv6
    literals so callers building ``'http://{host}:{port}'`` get a well-formed
    URL (e.g. ``'::1'`` -> ``'[::1]'``)."""
    from comfy_cli.local_address import resolve_local_host_port

    cfg = ConfigManager()
    host, port = resolve_local_host_port(host, port, background=cfg.background)
    h = validate_host(host)
    if ":" in h and not h.startswith("["):
        h = f"[{h}]"
    return (h, int(port))
