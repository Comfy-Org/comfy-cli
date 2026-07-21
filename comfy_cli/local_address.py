"""Resolve the local ComfyUI address, honoring the ``COMFY_LOCAL_URL`` env var.

comfy-cli's local target defaults to ``127.0.0.1:8188`` unless a per-command
``--host`` / ``--port`` is passed or a comfy-cli-launched background server was
recorded in config. ``COMFY_LOCAL_URL`` is the process-wide override that
points EVERY local command at a different address — e.g. a ComfyUI started
*outside* comfy-cli on ``:8189`` — without threading a flag through each
invocation. It is the local-address analogue of :data:`comfy_cli.where.ENV_DEFAULT`
(``COMFY_WHERE``), which picks the backend but not its address.

Two entry points:

- :func:`parse_local_url` — parse the env var's value (``http://host:port``,
  ``host:port``, or ``http://host``; scheme optional and only ``http``; IPv6
  literals bracketed) into ``(host, port)``. Raises ``ValueError`` on garbage.
- :func:`resolve_local_host_port` — the precedence resolver: explicit flag >
  ``COMFY_LOCAL_URL`` env > ``config.background`` > ``127.0.0.1:8188``, with
  host and port resolved independently.

A malformed ``COMFY_LOCAL_URL`` is *ignored with a one-line stderr warning*
rather than raised, so a typo in the env var can't hard-break every command.
"""

from __future__ import annotations

import os
import sys
from collections.abc import Mapping

ENV_LOCAL_URL = "COMFY_LOCAL_URL"
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8188

# URL-injection-unsafe host characters (mirrors ``comfy_cli.host_port``).
_UNSAFE_HOST_CHARS = frozenset("/@?#")

# Bad COMFY_LOCAL_URL values already warned about this process, so a resolver
# invoked at several sites per command doesn't spam identical warnings.
_warned: set[str] = set()


def parse_local_url(value: str) -> tuple[str, int]:
    """Parse a local ComfyUI address into ``(host, port)``.

    Accepts ``http://host:port``, ``host:port``, or ``http://host`` (port
    defaults to :data:`DEFAULT_PORT`). The scheme is optional and, when
    present, must be ``http``. Bracketed IPv6 literals (``[::1]:8189``) are
    supported and the brackets are stripped from the returned host. Raises
    ``ValueError`` on any input that isn't a well-formed local address.
    """
    v = value.strip()
    if not v:
        raise ValueError("empty value")

    if "://" in v:
        scheme, _, rest = v.partition("://")
        if scheme.lower() != "http":
            raise ValueError(f"unsupported scheme {scheme!r}: only 'http' is supported")
        v = rest

    # Drop any path / query / fragment — only the authority carries host:port.
    for sep in ("/", "?", "#"):
        i = v.find(sep)
        if i != -1:
            v = v[:i]
    if not v:
        raise ValueError("missing host")

    return _split_host_port(v)


def _split_host_port(authority: str) -> tuple[str, int]:
    """Split ``host[:port]`` (IPv6-aware) into a validated ``(host, port)``."""
    if authority.startswith("["):
        end = authority.find("]")
        if end == -1:
            raise ValueError("unterminated '[' in IPv6 literal")
        host = authority[1:end]
        rest = authority[end + 1 :]
        if rest:
            # Only a ``:port`` suffix may follow the bracket; anything else
            # (e.g. ``[::1]8188``) is a typo we must not silently drop.
            if not rest.startswith(":"):
                raise ValueError("unexpected text after ']'")
            port = _to_port(rest[1:]) if rest[1:] else DEFAULT_PORT
        else:
            port = DEFAULT_PORT
        return _validate_host(host), port
    if authority.count(":") == 1:  # host:port
        h, p = authority.split(":")
        return _validate_host(h), (_to_port(p) if p else DEFAULT_PORT)
    # zero colons -> hostname only; 2+ colons -> bare IPv6 literal (no port)
    return _validate_host(authority), DEFAULT_PORT


def _to_port(s: str) -> int:
    try:
        port = int(s)
    except ValueError:
        raise ValueError(f"invalid port {s!r}: not a number") from None
    if not (1 <= port <= 65535):
        raise ValueError(f"invalid port {port}: out of range (1-65535)")
    return port


def _validate_host(host: str) -> str:
    """Reject hosts that could corrupt a URL built as ``http://{host}:{port}``."""
    if not host:
        raise ValueError("empty host")
    if any(c in host for c in _UNSAFE_HOST_CHARS):
        raise ValueError(f"invalid host {host!r}: contains URL-special characters")
    if any(c.isspace() or ord(c) < 0x20 or ord(c) == 0x7F for c in host):
        raise ValueError(f"invalid host {host!r}: contains whitespace or control characters")
    return host


def _env_local_host_port(env: Mapping[str, str] | None = None) -> tuple[str | None, int | None]:
    """Parse ``COMFY_LOCAL_URL`` from the environment.

    Returns ``(host, port)`` from a valid value, or ``(None, None)`` when the
    var is unset/empty or malformed. A malformed value is ignored with a
    one-line stderr warning (deduplicated per process) so a typo can't
    hard-break every command.
    """
    e = env if env is not None else os.environ
    raw = e.get(ENV_LOCAL_URL)
    if not raw or not raw.strip():
        return None, None
    try:
        return parse_local_url(raw)
    except ValueError as exc:
        if raw not in _warned:
            _warned.add(raw)
            print(f"warning: ignoring invalid {ENV_LOCAL_URL}={raw!r}: {exc}", file=sys.stderr)
        return None, None


def resolve_local_host_port(
    host: str | None,
    port: int | None,
    background: tuple | None = None,
    *,
    env: Mapping[str, str] | None = None,
) -> tuple[str, int]:
    """Resolve the local ComfyUI ``(host, port)`` by precedence.

    Precedence, resolved *independently* for host and port:
    explicit flag > ``COMFY_LOCAL_URL`` env > ``config.background`` >
    :data:`DEFAULT_HOST` / :data:`DEFAULT_PORT`. So an explicit ``--port`` with
    no ``--host`` still picks up the env var's host, and vice-versa.

    ``background`` is the persisted ``ConfigManager().background`` tuple
    (``(host, port)``) or ``None``. The returned host is *raw* (unbracketed);
    callers building a URL bracket IPv6 literals themselves.
    """
    env_host, env_port = _env_local_host_port(env)
    bg_host = background[0] if background else None
    bg_port = background[1] if background else None
    resolved_host = host or env_host or bg_host or DEFAULT_HOST
    resolved_port = port or env_port or bg_port or DEFAULT_PORT
    return resolved_host, int(resolved_port)
