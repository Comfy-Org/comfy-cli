"""Detect whether the CLI is being driven by a human or an agent.

The detection flips three defaults:
    - Output mode (agents → JSON, humans → pretty)
    - Confirmation prompts (skipped for agents)
    - Pretty banner (suppressed for agents)

Signals (in priority order — most specific first):
    1. ``COMFY_USER_AGENT=<label>`` → explicit override, agentic, label preserved.
    2. ``AI_AGENT`` truthy → agentic, kind="agent".
    3. ``CLAUDECODE`` truthy → Claude Code session, kind="claude-code".
    4. stdout is not a TTY (or is missing/closed) → agentic, kind="pipe".
    5. otherwise → kind="user".

Claude Code is checked after AI_AGENT because AI_AGENT is the generic
contract any agent framework can set, while CLAUDECODE is Claude Code's
own env var (set in every Bash subprocess it spawns). Checking it
explicitly means analytics can distinguish "Claude Code" from "some
other agent" without requiring Claude Code users to set AI_AGENT.

This module also derives the ``comfy_usage_source`` attribution value
(``usage_source`` below) that rides every workflow submission, so
server-side partner-node billing can tell a human-driven CLI run apart
from an agent-driven one.

Tested in ``tests/comfy_cli/output/test_caller.py``.
"""

from __future__ import annotations

import os
import sys
from collections.abc import Mapping
from dataclasses import dataclass
from functools import lru_cache


@dataclass(frozen=True)
class Caller:
    kind: str  # "user" | "claude-code" | "pipe" | "agent" | <custom>
    agentic: bool
    source_env: str | None  # which env var triggered the detection, for debug


def _truthy(value: str | None) -> bool:
    if value is None:
        return False
    return value.strip().lower() not in {"", "0", "false", "no", "off"}


def stream_is_tty(stream: object) -> bool:
    """True only when *stream* is a live TTY. Never raises — that is the point.

    ``sys.stdout.isatty()`` assumes stdout is a live stream, but a process's
    standard streams are not guaranteed to be one. Under ``pythonw``, a Windows
    service, or a detached/daemonised parent they can be ``None``; after a
    wrapper closes or replaces them they can be an already-closed file, an
    object with no ``isatty`` at all, or one backed by a revoked file
    descriptor. Those raise ``AttributeError``, ``ValueError``, ``OSError``
    (``EBADF`` / ``WinError 6``) and ``TypeError`` respectively — and a
    non-conforming replacement stream can raise anything at all, since
    ``isatty`` is just an arbitrary attribute on an arbitrary object.

    So the handler is deliberately broad rather than a list of the failures we
    happened to think of. A process with no usable stream is by definition not
    a human at a terminal, so every failure means the same thing: not a TTY.

    This is the shared, fail-safe probe for every standard-stream TTY check on
    the startup path — ``detect_caller`` below, ``Renderer.resolve``, and
    ``tracking.prompt_tracking_consent``. All three run before argument
    parsing, so an escaping exception would take down every command, including
    ``--help`` and runs with tracking disabled.
    """
    if stream is None:
        return False
    try:
        # The attribute LOOKUP is inside the try, not just the call: on a proxy
        # or lazy wrapper stream, `isatty` can be a property or come from a
        # `__getattr__`, either of which can raise. `getattr(..., None)` only
        # swallows AttributeError, so a lookup that raised ValueError/OSError
        # would escape a function whose whole contract is "never raises".
        isatty = getattr(stream, "isatty", None)
        if isatty is None:
            return False
        return bool(isatty())
    except Exception:
        return False


def _stdout_is_tty() -> bool:
    """``stream_is_tty`` against the live ``sys.stdout``, re-read on each call
    so a test or wrapper that swaps the stream is honoured."""
    return stream_is_tty(getattr(sys, "stdout", None))


def detect_caller(
    env: Mapping[str, str] | None = None,
    *,
    is_tty: bool | None = None,
) -> Caller:
    e = env if env is not None else os.environ

    # 1. Explicit override — custom agent frameworks self-attribute here.
    explicit = e.get("COMFY_USER_AGENT")
    if explicit and _truthy(explicit):
        return Caller(kind=explicit.strip().lower(), agentic=True, source_env="COMFY_USER_AGENT")

    # 2. Generic agent marker — any framework can set this.
    if _truthy(e.get("AI_AGENT")):
        return Caller(kind="agent", agentic=True, source_env="AI_AGENT")

    # 3. Claude Code — sets CLAUDECODE=1 in every shell it spawns.
    if _truthy(e.get("CLAUDECODE")):
        return Caller(kind="claude-code", agentic=True, source_env="CLAUDECODE")

    # 4. Non-TTY — piped, backgrounded, or CI. Probed lazily, only once the
    #    env-var branches above have all declined: an explicitly-attributed
    #    caller is answered without ever touching stdout.
    tty = is_tty if is_tty is not None else _stdout_is_tty()
    if not tty:
        return Caller(kind="pipe", agentic=True, source_env=None)

    # 5. Human at a terminal.
    return Caller(kind="user", agentic=False, source_env=None)


# A custom COMFY_USER_AGENT is arbitrary user-supplied text, and the value
# derived from it rides a request body the server bills against, so bound it
# the way `tracking._sanitize_caller_kind` bounds `caller_kind` rather than
# forwarding an unbounded label.
_USAGE_SOURCE_MAX_LEN = 64


def usage_source_for(caller: Caller) -> str:
    """Map a caller to its ``comfy_usage_source`` attribution value.

    A human at a terminal keeps the bare ``comfy-cli``, so dashboards and
    margin queries that exact-match ``usage_source = 'comfy-cli'`` keep
    working — they narrow to human-driven CLI use, which is what those
    consumers already assume. Every agentic caller gets the slash-scoped
    ``comfy-cli/<kind>`` (``comfy-cli/comfy-mcp``, ``comfy-cli/claude-code``,
    ``comfy-cli/agent``, ``comfy-cli/pipe``), matching the
    ``comfy-cloud-mcp/<tool>`` convention already in production; a
    ``usage_source LIKE 'comfy-cli%'`` prefix query covers the whole family.

    The human case keys off ``agentic``, not ``kind == "user"``: a caller that
    self-attributes ``COMFY_USER_AGENT=user`` produces the *string* "user"
    while being exactly the agentic case, and a kind test would let it pass
    itself off as a human. This mirrors ``tracking._caller_kind_is_custom``.
    """
    if not caller.agentic:
        return "comfy-cli"
    return f"comfy-cli/{caller.kind[:_USAGE_SOURCE_MAX_LEN]}"


@lru_cache(maxsize=1)
def usage_source() -> str:
    """The ``comfy_usage_source`` value for this process.

    Computed once — ``detect_caller`` is env + isatty only, but the answer
    cannot change within a process, matching how ``tracking`` computes
    ``_caller_kind`` at import. Tests that vary the caller patch the name
    where it is *used* (the call sites import it directly) and/or call
    ``usage_source.cache_clear()``.
    """
    return usage_source_for(detect_caller())
