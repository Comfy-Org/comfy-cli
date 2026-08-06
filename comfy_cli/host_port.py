"""Shared host/port parsing + resolution for local ComfyUI commands.

``comfy run``, every ``comfy jobs`` subcommand, ``comfy validate``, the
``comfy nodes`` subcommands, and ``comfy upload`` accept a ``--host`` /
``--port`` pair. ``run``/``jobs``/``validate``/``nodes`` resolve it here —
falling back to the persisted ``config.background`` server, then to
``DEFAULT_HOST`` / ``DEFAULT_PORT``; ``comfy upload`` only borrows
:func:`validate_host` and hands the pair to ``target.resolve_target``, which
skips the background server (env override, then the defaults). In addition,
``comfy run`` and ``comfy validate`` accept a combined ``host[:port]`` string
(parsed via ``parse_host_port_arg``); the ``comfy jobs`` / ``comfy nodes``
subcommands and ``comfy upload`` only take the separate ``--host`` / ``--port``
options. All of them feed the resolved host straight into URLs like
``http://{host}:{port}/prompt`` / ``ws://{host}:{port}/ws``, so the value must
be validated (no URL-injection characters) and IPv6 literals bracketed. This
module is the single home for that logic; callers should not re-implement it.
"""

from __future__ import annotations

import ipaddress
import urllib.parse

import typer

from comfy_cli.config_manager import ConfigManager

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8188
_UNSAFE_HOST_CHARS = frozenset("/@?#")


def validate_host(host: str) -> str:
    """Reject host values that could cause URL injection."""
    # An empty/blank host is not "no host": callers resolve it as absent
    # (``host or env or DEFAULT_HOST``), so ``--host ""`` — a wrapper
    # interpolating an unset variable — would silently retarget the request at
    # a server the caller never named. ``_require_host`` rejects it on the
    # parse path for the same reason; reject it here too so both agree.
    if not host.strip():
        raise typer.BadParameter(f"invalid host: {host!r} (empty host)")
    # ``urllib.request.Request._parse`` percent-decodes the host it splits out
    # of the URL, so a ``%0d%0a`` payload only becomes a real CRLF *after* this
    # guard. Check the decoded form too rather than trusting the literal value.
    for candidate in (host, urllib.parse.unquote(host)):
        if any(c in candidate for c in _UNSAFE_HOST_CHARS):
            raise typer.BadParameter(f"invalid host: {host!r} (contains URL-special characters)")
        # Whitespace/control chars (notably CR/LF) never appear in a real host
        # and are the canonical header/URL-injection vectors, so reject them.
        if any(c.isspace() or ord(c) < 0x20 or ord(c) == 0x7F for c in candidate):
            raise typer.BadParameter(f"invalid host: {host!r} (contains whitespace or control characters)")
    _reject_embedded_port(host)
    return host


def _reject_embedded_port(host: str) -> None:
    """Reject a colon in a host unless it's a genuine IPv6 literal.

    Callers bracket any colon-bearing host into ``http://[{host}]:{port}``, so
    a combined ``'127.0.0.1:8188'`` would silently become
    ``http://[127.0.0.1:8188]:8188`` — the embedded port dropped and the
    request aimed at a bogus address. Only ``comfy run`` accepts the combined
    form, and it splits it with :func:`parse_host_port_arg` before we see it.
    """
    if ":" not in host:
        return
    literal = host[1:-1] if host.startswith("[") and host.endswith("]") else host
    try:
        ipaddress.IPv6Address(literal)
    except ValueError:
        raise typer.BadParameter(
            f"invalid host: {host!r} (contains ':'; pass the port separately with --port, or give a valid IPv6 literal)"
        ) from None


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


def validate_port(port: int) -> int:
    """Reject a ``--port`` outside the TCP range.

    ``resolve_local_host_port`` resolves the port as ``port or env or bg or
    DEFAULT``, so a falsy ``--port 0`` is indistinguishable from "not passed"
    and silently resolves to some *other* server; an out-of-range ``--port
    99999`` is worse, flowing straight into ``http://{host}:99999`` to fail at
    connect time with no hint that the flag was the problem. Reject both here,
    matching the range check :func:`_to_port` already applies to the port half
    of a combined ``host:port`` string.
    """
    if not (1 <= port <= 65535):
        raise typer.BadParameter(f"invalid port: {port} is out of range (1-65535)")
    return port


def resolve_host_port(host: str | None, port: int | None) -> tuple[str, int]:
    """Resolve host/port by precedence — explicit flag > ``COMFY_LOCAL_URL``
    env > ``config.background`` > defaults — then validate and bracket IPv6
    literals so callers building ``'http://{host}:{port}'`` get a well-formed
    URL (e.g. ``'::1'`` -> ``'[::1]'``).

    An explicitly-passed ``host``/``port`` is validated *before* the precedence
    chain runs, because that chain treats every falsy value as "not passed" and
    would otherwise swallow the bad input and resolve to a different server.
    This function owns the range check for the separate ``--port`` flag on
    behalf of every caller (``run``, ``templates run``, ``jobs``, ``validate``,
    ``nodes``), so call sites must not re-implement it — they only need to keep
    an explicit ``--port`` distinguishable from an absent one (``port is None``,
    never ``not port``) so a ``--port 0`` reaches this guard.
    """
    from comfy_cli.env_checker import _bracket_host
    from comfy_cli.local_address import resolve_local_host_port

    # ``--host ""`` is not "no host": a wrapper interpolating an unset variable
    # would silently retarget the request at the env/background/default server.
    # ``validate_host`` already rejects a blank host (see its comment) — it just
    # never sees one, because the ``host or …`` fallback below drops it first.
    if host is not None and not host.strip():
        validate_host(host)
    if port is not None:
        validate_port(port)
    cfg = ConfigManager()
    host, port = resolve_local_host_port(host, port, background=cfg.background)
    # Validate BEFORE bracketing: ``validate_host``'s unsafe-char set does not
    # include ``[``/``]``, so it must see the unbracketed value.
    h = validate_host(host)
    # Bracketing is delegated to the shared ``_bracket_host`` choke point.
    return (_bracket_host(h), int(port))
