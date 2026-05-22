"""Caller detection: env vars + TTY → agentic mode.

Priority: COMFY_USER_AGENT > AI_AGENT > CLAUDECODE > non-TTY > user.
All agentic callers flip the same three defaults (JSON, skip prompts,
no banner). The ``kind`` field is for analytics/logging only.
"""

from comfy_cli.caller import detect_caller


def test_tty_no_env_is_user():
    c = detect_caller(env={}, is_tty=True)
    assert c.kind == "user"
    assert c.agentic is False
    assert c.source_env is None


def test_no_tty_is_pipe():
    """Piped into another process or backgrounded → agentic, kind="pipe"."""
    c = detect_caller(env={}, is_tty=False)
    assert c.agentic is True
    assert c.kind == "pipe"
    assert c.source_env is None


def test_ai_agent_env_var_forces_agentic_even_on_tty():
    c = detect_caller(env={"AI_AGENT": "1"}, is_tty=True)
    assert c.agentic is True
    assert c.kind == "agent"
    assert c.source_env == "AI_AGENT"


def test_ai_agent_explicit_off_falls_through():
    """A user who set AI_AGENT=0 doesn't want the agentic path."""
    c = detect_caller(env={"AI_AGENT": "0"}, is_tty=True)
    assert c.agentic is False
    assert c.kind == "user"


def test_claude_code_detected_on_tty():
    """CLAUDECODE=1 is set in every shell Claude Code spawns."""
    c = detect_caller(env={"CLAUDECODE": "1"}, is_tty=True)
    assert c.agentic is True
    assert c.kind == "claude-code"
    assert c.source_env == "CLAUDECODE"


def test_claude_code_off_falls_through():
    c = detect_caller(env={"CLAUDECODE": "0"}, is_tty=True)
    assert c.agentic is False
    assert c.kind == "user"


def test_ai_agent_wins_over_claudecode():
    """AI_AGENT is more specific (explicitly set by agent frameworks)."""
    c = detect_caller(env={"AI_AGENT": "1", "CLAUDECODE": "1"}, is_tty=True)
    assert c.kind == "agent"
    assert c.source_env == "AI_AGENT"


def test_user_agent_override_carries_label():
    """COMFY_USER_AGENT lets a wrapper self-attribute with a specific label."""
    c = detect_caller(env={"COMFY_USER_AGENT": "my-bot"}, is_tty=True)
    assert c.agentic is True
    assert c.kind == "my-bot"
    assert c.source_env == "COMFY_USER_AGENT"


def test_user_agent_override_wins_over_everything():
    c = detect_caller(env={"COMFY_USER_AGENT": "harness", "AI_AGENT": "1", "CLAUDECODE": "1"}, is_tty=True)
    assert c.kind == "harness"
    assert c.source_env == "COMFY_USER_AGENT"
