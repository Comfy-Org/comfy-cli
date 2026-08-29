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
