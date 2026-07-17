"""The telemetry SDKs must not be imported just by starting the CLI (BE-3289).

`mixpanel` depends on pydantic, which loads the compiled `pydantic_core`
extension. A `comfy` process that has that extension loaded holds the .pyd/.so
open for its whole lifetime, and on Windows an open DLL cannot be replaced — so
`comfy install`, which shells out to `uv pip install` against its own
interpreter's environment, could not upgrade pydantic_core:

    error: failed to remove file `...\\pydantic_core\\_pydantic_core.cp312-win_amd64.pyd`:
    Access is denied. (os error 5)

uv exited 2 having already unlinked the distribution's metadata, so every later
`comfy` invocation died with `ImportError: cannot import name '__version__' from
'pydantic_core' (unknown location)`.

These run in subprocesses on purpose: the assertion is about what a *fresh*
interpreter loads, and the pytest process has already imported plenty on its own.
"""

from __future__ import annotations

import subprocess
import sys

# The SDKs themselves plus the transitive dependency that actually did the damage.
_FORBIDDEN = ("mixpanel", "posthog", "pydantic", "pydantic_core")


def _modules_after(source: str) -> set[str]:
    """Run *source* in a fresh interpreter; return its top-level sys.modules names."""
    script = f"{source}\nimport sys, json\nprint(json.dumps(sorted(m.split('.')[0] for m in sys.modules)))"
    proc = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        check=True,
    )
    import json

    return set(json.loads(proc.stdout.strip().splitlines()[-1]))


def test_importing_tracking_does_not_load_telemetry_sdks():
    loaded = _modules_after("import comfy_cli.tracking")
    assert not loaded & set(_FORBIDDEN), (
        f"comfy_cli.tracking imported {sorted(loaded & set(_FORBIDDEN))} at module scope; "
        "these must stay lazy so `comfy install` can upgrade pydantic_core on Windows"
    )


def test_importing_the_cli_entrypoint_does_not_load_telemetry_sdks():
    # cmdline is what `comfy` actually runs, and it is the import chain in the
    # BE-3289 traceback (cmdline -> tracking -> mixpanel -> pydantic).
    loaded = _modules_after("import comfy_cli.cmdline")
    assert not loaded & set(_FORBIDDEN), (
        f"comfy_cli.cmdline imported {sorted(loaded & set(_FORBIDDEN))} at module scope; "
        "these must stay lazy so `comfy install` can upgrade pydantic_core on Windows"
    )


def test_atexit_flush_does_not_construct_providers():
    """The atexit hook must not be the thing that imports the SDKs at exit."""
    loaded = _modules_after("import comfy_cli.tracking as t\nt._flush_all_providers()")
    assert not loaded & set(_FORBIDDEN)


def test_providers_are_built_on_first_dispatch():
    """The deferral must be a deferral, not a removal — telemetry still works."""
    loaded = _modules_after(
        "import comfy_cli.tracking as t\n"
        "assert t.PROVIDERS is None, 'providers built before any send'\n"
        "providers = t._get_providers()\n"
        "assert t.PROVIDERS is providers, 'accessor did not cache onto the module global'\n"
        "assert len(providers) == 2, f'expected mixpanel+posthog, got {providers!r}'\n"
        "assert t._get_providers() is providers, 'accessor rebuilt providers on second call'\n"
    )
    # Constructing them is precisely what pulls the SDKs in.
    assert {"mixpanel", "posthog"} <= loaded


def test_broken_sdk_import_degrades_instead_of_crashing_the_command():
    """Deferring the import moves it onto the user's command path, so the failure it
    used to hit at startup can now surface mid-run. A half-removed pydantic_core is
    exactly the state this change exists to prevent, but a user can still land in it
    from an earlier bad install — telemetry must go quiet, not take the command down.
    """
    _modules_after(
        "import builtins\n"
        "_real = builtins.__import__\n"
        "def boom(name, *a, **k):\n"
        "    if name in ('mixpanel', 'posthog'):\n"
        "        raise ImportError(\"cannot import name '__version__' from 'pydantic_core'\")\n"
        "    return _real(name, *a, **k)\n"
        "builtins.__import__ = boom\n"
        "import comfy_cli.tracking as t\n"
        "t.MIXPANEL_TOKEN = 'tok'\n"
        "t.POSTHOG_TOKEN = 'tok'\n"
        # The send path must survive: _get_providers() is called in _dispatch's loop
        # header, outside the per-provider try/except that only guards .track().
        "t._dispatch('evt', {}, distinct_id='abc')\n"
        "assert t._get_providers() == [], f'expected no providers, got {t._get_providers()!r}'\n"
        # Cached, so a doomed import isn't retried on every later event.
        "assert t.PROVIDERS == []\n"
        "t._flush_all_providers()\n"
    )


def test_one_broken_sdk_does_not_silence_the_other():
    """Providers are built independently: an unusable PostHog SDK must not throw away
    a perfectly good Mixpanel provider (and vice versa)."""
    _modules_after(
        "import builtins\n"
        "_real = builtins.__import__\n"
        "def boom(name, *a, **k):\n"
        "    if name == 'posthog':\n"
        "        raise ImportError('posthog is broken')\n"
        "    return _real(name, *a, **k)\n"
        "builtins.__import__ = boom\n"
        "import comfy_cli.tracking as t\n"
        "t.MIXPANEL_TOKEN = 'tok'\n"
        "t.POSTHOG_TOKEN = 'tok'\n"
        "providers = t._get_providers()\n"
        "names = [type(p).__name__ for p in providers]\n"
        "assert names == ['MixpanelProvider'], f'lost the healthy provider: {names!r}'\n"
    )


def test_concurrent_first_send_builds_one_set_of_providers():
    """Module-level init was single-shot via the import lock; the lazy accessor
    has to keep that. A racing build would strand a PostHog client, and its
    unflushed event queue, in the list that lost."""
    _modules_after(
        "import comfy_cli.tracking as t\n"
        "from concurrent.futures import ThreadPoolExecutor\n"
        "with ThreadPoolExecutor(max_workers=8) as ex:\n"
        "    got = [f.result() for f in [ex.submit(t._get_providers) for _ in range(8)]]\n"
        "assert all(g is got[0] for g in got), 'threads saw different provider lists'\n"
        "assert t.PROVIDERS is got[0]\n"
    )
