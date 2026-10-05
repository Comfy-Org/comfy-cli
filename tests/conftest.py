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


@pytest.fixture(autouse=True)
def _no_cloud_model_asset_lookup():
    """Keep the suite offline and deterministic: a model value the catalog
    lacks is NOT looked up in a Cloud asset library unless a test installs a
    lookup itself (``comfy_cli.cql.model_assets.set_lookup``)."""
    from comfy_cli.cql import model_assets

    model_assets.set_lookup(None)
    yield
    model_assets.reset()
