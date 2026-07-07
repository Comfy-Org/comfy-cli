"""Detect whether the CLI is being driven by a human or an agent.

The detection flips three defaults:
    - Output mode (agents → JSON, humans → pretty)
    - Confirmation prompts (skipped for agents)
    - Pretty banner (suppressed for agents)

Signals (in priority order — most specific first):
    1. ``COMFY_USER_AGENT=<label>`` → explicit override, agentic, label preserved.
    2. ``AI_AGENT`` truthy → agentic, kind="agent".
    3. ``CLAUDECODE`` truthy → Claude Code session, kind="claude-code".
    4. stdout is not a TTY → agentic, kind="pipe".
    5. otherwise → kind="user".

Claude Code is checked after AI_AGENT because AI_AGENT is the generic
contract any agent framework can set, while CLAUDECODE is Claude Code's
own env var (set in every Bash subprocess it spawns). Checking it
explicitly means analytics can distinguish "Claude Code" from "some
other agent" without requiring Claude Code users to set AI_AGENT.

Tested in ``tests/comfy_cli/output/test_caller.py``.
"""

from __future__ import annotations

import os
import sys
from collections.abc import Mapping
from dataclasses import dataclass


@dataclass(frozen=True)
class Caller:
    kind: str  # "user" | "claude-code" | "pipe" | "agent" | <custom>
    agentic: bool
    source_env: str | None  # which env var triggered the detection, for debug


def _truthy(value: str | None) -> bool:
    if value is None:
        return False
    return value.strip().lower() not in {"", "0", "false", "no", "off"}


def detect_caller(
    env: Mapping[str, str] | None = None,
    *,
    is_tty: bool | None = None,
) -> Caller:
    e = env if env is not None else os.environ
    tty = is_tty if is_tty is not None else sys.stdout.isatty()

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

    # 4. Non-TTY — piped, backgrounded, or CI.
    if not tty:
        return Caller(kind="pipe", agentic=True, source_env=None)

    # 5. Human at a terminal.
    return Caller(kind="user", agentic=False, source_env=None)
