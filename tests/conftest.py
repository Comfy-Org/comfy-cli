import pytest

_COLOR_FORCING_ENV_VARS = ("FORCE_COLOR", "NO_COLOR", "CLICOLOR", "CLICOLOR_FORCE")


@pytest.fixture(autouse=True)
def _neutralize_forced_color(monkeypatch):
    """Strip color-forcing env vars so rich/click fall back to real tty
    detection (no tty under pytest's capture -> no color), matching CI.

    A shell that exports ``FORCE_COLOR`` for nicer everyday output otherwise
    makes every pretty-mode assertion in the suite fail non-deterministically,
    since output that should be plain comes back wrapped in ANSI codes (or
    vice versa for ``NO_COLOR``).
    """
    for var in _COLOR_FORCING_ENV_VARS:
        monkeypatch.delenv(var, raising=False)


#: Env vars ``comfy_cli.caller.detect_caller`` reads to decide the caller is an
#: agent rather than a person (``caller.py`` branches 1-3).
_AGENT_ATTRIBUTION_ENV_VARS = ("COMFY_USER_AGENT", "AI_AGENT", "CLAUDECODE")


@pytest.fixture(autouse=True)
def _neutralize_agent_attribution(monkeypatch):
    """Strip agent-attribution env vars so ``detect_caller`` sees a plain
    caller, matching CI.

    Same hazard as the color vars above, from the other direction: an AI coding
    tool exports these into every shell it spawns, so running the suite from
    inside one flips code paths that branch on ``caller.agentic``. `comfy
    code-search` adds an ``X-Comfy-Code-Search-Origin: agent`` header and
    passes ``agent_origin=True`` (``command/code_search.py:685``, ``:748``),
    and five ``TestCodeSearchCLI`` cases then fail on the extra kwarg while
    passing in CI, where no such var is set.

    Only the three env-var branches are neutralized. Branch 4 (non-tty) still
    reports an agentic caller under pytest's capture, which is what CI sees
    too, so tests that depend on it keep working.
    """
    for var in _AGENT_ATTRIBUTION_ENV_VARS:
        monkeypatch.delenv(var, raising=False)
