import os

import pytest


@pytest.fixture(autouse=True)
def _preserve_cwd():
    """Restore the working directory after every test.

    Several functions in comfy_cli.command.install (execute,
    pip_install_comfyui_dependencies) call os.chdir() as a side effect.
    Without this fixture the changed CWD leaks into subsequent tests and
    can cause hard-to-debug failures.
    """
    original = os.getcwd()
    yield
    os.chdir(original)


@pytest.fixture(autouse=True)
def _reset_renderer_singleton():
    """The Renderer is a process-wide singleton (output.set_renderer).

    A CliRunner test that invokes the app installs a renderer keyed to the
    runner's piped streams; a later direct-call test would otherwise inherit
    that JSON-mode renderer and have its capsys captures come up empty.

    Reset before AND after every test so we always start from a clean default
    (pretty mode, looking up current sys.stdout / sys.stderr).
    """
    from comfy_cli.output.renderer import reset_renderer_for_testing

    reset_renderer_for_testing()
    yield
    reset_renderer_for_testing()


@pytest.fixture(autouse=True)
def _isolate_ambient_api_keys(monkeypatch):
    """Clear the API keys a developer or CI shell may export.

    ``comfy build`` and ``comfy deploy`` send ``COMFY_CLOUD_API_KEY`` ahead of any
    sign-in, so a key left in the shell would turn every signed-out test into a
    keyed one. A test that needs a key sets it itself.
    """
    monkeypatch.delenv("COMFY_CLOUD_API_KEY", raising=False)
    monkeypatch.delenv("COMFY_API_KEY", raising=False)
    # Handed-in tokens outrank a stored sign-in, so one left in the shell would
    # turn every signed-in test into a token one.
    monkeypatch.delenv("COMFY_CLOUD_AUTH_TOKEN", raising=False)
    monkeypatch.delenv("COMFY_BUILDER_TOKEN", raising=False)


@pytest.fixture(autouse=True)
def _isolate_config_path(tmp_path, monkeypatch):
    """Redirect the CLI's config dir to a per-test tmp path.

    Without this, tests that exercise ``comfy set-default``, ``comfy auth
    set-base-url``, ``comfy auth set-cloud-key``, etc. write to the user's
    real ``~/Library/Application Support/comfy-cli/`` (or platform
    equivalent) — wiping persisted state between runs. We saw this in
    practice: ``where_default`` and ``cloud_base_url`` kept disappearing
    mid-session because tests were clobbering them.
    """
    from comfy_cli import constants
    from comfy_cli.config_manager import ConfigManager
    from comfy_cli.utils import reset_singleton_for_testing

    fake_root = tmp_path / "comfy-cli-config"
    fake_root.mkdir(mode=0o700, parents=True, exist_ok=True)
    # constants.DEFAULT_CONFIG is a dict keyed by OS; patch all entries so
    # whichever ``get_os()`` resolves to lands in our tmp dir.
    for k in list(constants.DEFAULT_CONFIG.keys()):
        monkeypatch.setitem(constants.DEFAULT_CONFIG, k, str(fake_root))
    # ConfigManager is a @singleton that reads the config dir once, at
    # construction — so repointing the dir alone leaves the FIRST test's
    # in-memory config serving every later one. Any test that runs the CLI
    # entrypoint writes `setup_nudged` into it, which then breaks the
    # onboarding tests' "fresh config" premise hundreds of tests later.
    reset_singleton_for_testing(ConfigManager)
    yield fake_root
    reset_singleton_for_testing(ConfigManager)


@pytest.fixture(autouse=True)
def _isolate_object_info_cache_dir(tmp_path, monkeypatch):
    """Redirect the ``object_info`` disk cache to a per-test tmp dir.

    ``resilient_load_object_info`` (comfy_cli.cql.loader) reads/writes
    ``~/.cache/comfy-cli/object_info-*.json`` (or ``$XDG_CACHE_HOME``) as a
    side effect of every cache-first fetch. Now that `comfy run`'s UI-convert
    and preflight-validate call sites route through it too, any test that
    exercises those paths would otherwise read stale state from — or write
    real dumps into — the developer's actual cache directory.
    """
    fake = tmp_path / "comfy-cli-cache"
    fake.mkdir(mode=0o700, parents=True, exist_ok=True)
    monkeypatch.setenv("XDG_CACHE_HOME", str(fake))
    monkeypatch.delenv("COMFY_CACHE_DIR", raising=False)
    yield fake


@pytest.fixture(autouse=True)
def _isolate_jobs_state_dir(tmp_path, monkeypatch):
    """Redirect ``jobs_state.state_dir`` to a per-test tmp dir.

    Without this, tests that exercise ``comfy run`` mock-write state files
    against a ``MagicMock`` prompt_id, polluting the user's real state dir
    with garbage files. The defensive check in ``jobs_state.write`` now
    also raises on non-string prompt_ids, but pinning the dir keeps the
    real one untouched even when a future test forgets.
    """
    from comfy_cli import jobs_state

    fake = tmp_path / "jobs"
    fake.mkdir(mode=0o700, parents=True, exist_ok=True)
    monkeypatch.setattr(jobs_state, "state_dir", lambda: fake)
    yield fake


@pytest.fixture(autouse=True)
def _pin_usage_source_to_human(monkeypatch):
    """Pin the submit paths' ``comfy_usage_source`` to the human-caller value.

    ``caller.usage_source()`` is derived from the live environment, which under
    pytest is not a stable input: stdout is captured (so ``detect_caller``
    answers "pipe"), and a run driven by an agent harness exports
    ``CLAUDECODE`` / ``AI_AGENT`` / ``COMFY_USER_AGENT`` into the test process.
    Left alone, the same payload assertion would pass in a terminal and fail in
    CI or under an agent.

    So pin the baseline to the human-at-a-terminal value — the one that stayed
    the bare ``comfy-cli`` — at the two places that send it. The patch is
    deliberately applied to the *importing* modules rather than to
    ``caller.detect_caller``: a global detect_caller patch also rewrites what
    ``tracking`` and ``renderer`` see. A test that wants a different caller
    re-patches the same names (see ``cloud/test_client.py``,
    ``command/test_run.py``); ``caller.usage_source_for`` is tested directly in
    ``output/test_caller.py`` and needs no pinning.
    """
    from comfy_cli import comfy_client
    from comfy_cli.command.run import execution

    for module in (comfy_client, execution):
        monkeypatch.setattr(module, "usage_source", lambda: "comfy-cli")
    yield


@pytest.fixture
def pretty_no_stdout(capsys):
    """Assert pretty-mode commands write nothing to stdout.

    Opt-in. Use in tests that exercise pretty-mode rendering to pin the
    envelope contract: pretty mode is supposed to send all output to stderr
    so the one-JSON-on-stdout invariant for JSON consumers can't be
    silently violated by a stray ``print()`` or ``rich.print()`` call.

    Usage:

        def test_my_pretty_thing(pretty_no_stdout):
            my_command(...)
            # fixture asserts on teardown
    """
    yield
    captured = capsys.readouterr()
    if captured.out.strip():
        raise AssertionError(
            "Pretty mode wrote to stdout — the envelope contract requires "
            "that to be empty (all output goes to stderr in pretty mode):\n"
            f"  stdout = {captured.out!r}\n"
        )


@pytest.fixture(autouse=True)
def _isolate_comfy_desktop(tmp_path, monkeypatch):
    """Point Comfy Desktop detection at an empty per-test dir.

    ``comfy which`` / ``comfy env`` (and ``comfy build init``) read Desktop's
    ``installations.json`` and ``settings.json`` from the real app-data dir.
    On a developer machine with Desktop installed, every test touching those
    commands would otherwise see that machine's installs. A test that wants a
    Desktop install builds one with the ``comfy_desktop`` fixture.
    """
    from comfy_cli import desktop_install

    monkeypatch.setattr(desktop_install, "desktop_app_data_dir", lambda: tmp_path / "no-comfy-desktop")
    monkeypatch.setattr(desktop_install, "legacy_config_dir", lambda: None)
    yield


@pytest.fixture
def comfy_desktop(tmp_path, monkeypatch):
    """A Comfy Desktop 2 install laid out the way the bug report found it.

    An install adopted from Desktop 1: the ComfyUI checkout under
    ``<installPath>/ComfyUI``, the data (models, custom_nodes, venv) in the
    Desktop 1 base path (``config.json`` → ``basePath``, copied into the
    record as ``adoptedBaseDir``), and a shared model library listed in
    ``settings.json`` → ``modelsDirs`` alongside the data dir's own
    ``models/``. Detection is pointed at it; the returned dict names each part.
    """
    import json

    from comfy_cli import desktop_install

    install_path = tmp_path / "ComfyUI-Installs" / "ComfyUI"
    code = install_path / "ComfyUI"
    code.mkdir(parents=True)
    (code / "main.py").write_text("")
    data = tmp_path / "Documents" / "ComfyUI"
    for sub in ("models/checkpoints", "custom_nodes", "user"):
        (data / sub).mkdir(parents=True)
    python = data / ".venv" / "bin" / "python3"
    python.parent.mkdir(parents=True)
    python.write_text("")
    shared = tmp_path / "ComfyUI-Shared" / "models"
    shared.mkdir(parents=True)
    legacy = tmp_path / "Application Support" / "ComfyUI"
    legacy.mkdir(parents=True)
    (legacy / "config.json").write_text(json.dumps({"basePath": str(data)}))
    record = {
        "id": "inst-adopted",
        "name": "ComfyUI",
        "sourceId": "standalone",
        "installPath": str(install_path),
        "adopted": True,
        "adoptedBaseDir": str(data),
        "adoptedPythonPath": str(python),
        "comfyVersion": {"commit": "b53f77a7", "baseTag": "v0.34.2", "commitsAhead": 0},
        "useSharedModels": True,
        "status": "installed",
        "lastLaunchedAt": 1792188627164,
    }
    cloud = {"id": "inst-cloud", "name": "Comfy Cloud", "sourceId": "cloud", "remoteUrl": "https://cloud.comfy.org/"}
    app = tmp_path / "Application Support" / "Comfy Desktop"
    app.mkdir(parents=True)
    (app / "installations.json").write_text(json.dumps([record, cloud]))
    (app / "settings.json").write_text(json.dumps({"modelsDirs": [str(shared), str(data / "models")]}))

    monkeypatch.setattr(desktop_install, "desktop_app_data_dir", lambda: app)
    monkeypatch.setattr(desktop_install, "legacy_config_dir", lambda: legacy)
    return {
        "app": app,
        "legacy": legacy,
        "install_path": install_path,
        "code": code,
        "data": data,
        "python": python,
        "shared": shared,
        "record": record,
    }
