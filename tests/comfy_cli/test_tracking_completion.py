"""Regression tests for GitHub #506 — telemetry must stay inert during shell
tab-completion.

Under fish/bash/zsh, every tab-completion of a ``comfy ...`` command re-invokes
the CLI with a ``_<PROG_NAME>_COMPLETE`` environment variable set. No command
runs on that path, so no telemetry is ever sent — standing up the PostHog client
(a background thread + network setup) there is pure waste. These tests lock in
that the telemetry providers are never constructed while a completion env var is
present, at both the construction gateway (:func:`_get_providers`) and end to end
through the real Typer completion resolution.
"""

from unittest.mock import MagicMock, patch

import pytest

import comfy_cli.tracking as tracking_mod


class TestInShellCompletionDetection:
    @pytest.mark.parametrize(
        "env_var",
        [
            # Typer's own completion instruction var, per entrypoint / shell.
            "_COMFY_COMPLETE",  # `comfy`
            "_COMFY_CLI_COMPLETE",  # `comfy-cli`
            "_COMFYCLI_COMPLETE",  # `comfycli`
            "_SOMEOTHERTOOL_COMPLETE",  # a user alias
        ],
    )
    def test_completion_env_var_is_detected(self, monkeypatch, env_var):
        monkeypatch.setenv(env_var, "complete_fish")
        assert tracking_mod._in_shell_completion() is True

    @pytest.mark.parametrize("shell_instruction", ["complete_fish", "complete_bash", "complete_zsh"])
    def test_detected_across_shells(self, monkeypatch, shell_instruction):
        # The value differs per shell; detection keys off the var name, so any of
        # them trips the guard.
        monkeypatch.setenv("_COMFY_COMPLETE", shell_instruction)
        assert tracking_mod._in_shell_completion() is True

    @pytest.mark.parametrize(
        "name",
        [
            "PATH",
            "AUTO_COMPLETE",  # no leading underscore
            "_COMFY_HOME",  # underscore-prefixed but not a completion var
            "COMPLETE",
        ],
    )
    def test_non_completion_env_is_not_detected(self, monkeypatch, name):
        _clear_completion_env(monkeypatch)
        monkeypatch.setenv(name, "1")
        assert tracking_mod._in_shell_completion() is False


class TestGetProvidersUnderCompletion:
    def test_no_providers_constructed_during_completion(self, monkeypatch):
        """The core acceptance criterion: with a completion env var set,
        :func:`_get_providers` must return no providers and never invoke a
        provider factory (which is what imports the SDKs and builds the PostHog
        client)."""
        monkeypatch.setenv("_COMFY_COMPLETE", "complete_fish")
        # Start from a clean cache so we exercise the construction path.
        monkeypatch.setattr(tracking_mod, "PROVIDERS", None)

        def _boom(*_a, **_k):
            raise AssertionError("telemetry provider constructed during shell completion")

        with (
            patch.object(tracking_mod, "MixpanelProvider", _boom),
            patch.object(tracking_mod, "PostHogProvider", _boom),
        ):
            providers = tracking_mod._get_providers()

        assert providers == []
        # The guard is intentionally uncached so a real invocation is unaffected.
        assert tracking_mod.PROVIDERS is None

    def test_providers_constructed_when_not_completing(self, monkeypatch):
        """Control: without a completion env var, providers build as before."""
        _clear_completion_env(monkeypatch)
        monkeypatch.setattr(tracking_mod, "PROVIDERS", None)

        mp, ph = MagicMock(name="mixpanel"), MagicMock(name="posthog")
        with (
            patch.object(tracking_mod, "MixpanelProvider", MagicMock(return_value=mp)),
            patch.object(tracking_mod, "PostHogProvider", MagicMock(return_value=ph)),
        ):
            providers = tracking_mod._get_providers()

        assert providers == [mp, ph]


class TestCompletionEndToEnd:
    @pytest.mark.parametrize("shell_instruction", ["complete_fish", "complete_bash", "complete_zsh"])
    def test_real_completion_never_constructs_posthog(self, monkeypatch, tmp_path, shell_instruction):
        """Drive the actual Typer completion resolution the way a shell does and
        assert the PostHog client is never constructed. This is the end-to-end
        proof of GitHub #506's fix across fish/bash/zsh."""
        # Isolate config so completion can't touch the real user config.
        monkeypatch.setenv("COMFY_HOME", str(tmp_path))
        monkeypatch.setenv("HOME", str(tmp_path))
        # Typer's completion protocol: instruction + the words resolved so far.
        monkeypatch.setenv("_COMFY_COMPLETE", shell_instruction)
        monkeypatch.setenv("_TYPER_COMPLETE_ARGS", "comfy ")
        monkeypatch.setenv("_TYPER_COMPLETE_FISH_ACTION", "get-args")
        monkeypatch.setattr("sys.argv", ["comfy"])
        monkeypatch.setattr(tracking_mod, "PROVIDERS", None)

        constructed = MagicMock(side_effect=AssertionError("PostHog client constructed during completion"))
        from comfy_cli.cmdline import main

        with patch.object(tracking_mod.PostHogProvider, "__init__", constructed):
            with pytest.raises(SystemExit) as exc:
                main()

        # Completion resolves successfully (exit 0) without ever touching PostHog.
        assert exc.value.code == 0
        constructed.assert_not_called()


def _clear_completion_env(monkeypatch):
    """Remove any ``_*_COMPLETE`` var the ambient environment might carry so the
    negative/control cases start from a known non-completion state."""
    import os

    for key in list(os.environ):
        if key.startswith("_") and key.endswith("_COMPLETE"):
            monkeypatch.delenv(key, raising=False)
