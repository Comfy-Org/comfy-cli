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

    fake_root = tmp_path / "comfy-cli-config"
    fake_root.mkdir(mode=0o700, parents=True, exist_ok=True)
    # constants.DEFAULT_CONFIG is a dict keyed by OS; patch all entries so
    # whichever ``get_os()`` resolves to lands in our tmp dir.
    for k in list(constants.DEFAULT_CONFIG.keys()):
        monkeypatch.setitem(constants.DEFAULT_CONFIG, k, str(fake_root))
    yield fake_root


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
