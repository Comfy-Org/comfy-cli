from unittest.mock import MagicMock, patch

import pytest

from comfy_cli import constants
from comfy_cli.config_manager import ConfigManager

# Unwrap the singleton to get fresh ConfigManager instances per test.
_ConfigManagerCls = ConfigManager.__closure__[0].cell_contents


@pytest.fixture
def tracking_module(tmp_path):
    """Yield comfy_cli.tracking with a fresh tmp-path ConfigManager and a single
    mocked TelemetryProvider in PROVIDERS so tests can assert on the fan-out.

    Exposes the mock as ``tracking_mod.provider`` for assertions.
    """
    config_dir = tmp_path / "comfy-cli"
    config_dir.mkdir()
    with patch.object(_ConfigManagerCls, "get_config_path", return_value=str(config_dir)):
        cfg = _ConfigManagerCls()

    import comfy_cli.tracking as tracking_mod

    fake_provider = MagicMock()
    fake_provider.enabled = True
    # Mirror MixpanelProvider's no-op-on-missing-distinct-id behavior so opt-out
    # paths look identical from the test's perspective.
    fake_provider.track.return_value = None

    with (
        patch.object(tracking_mod, "config_manager", cfg),
        patch.object(tracking_mod, "user_id", None),
        patch.object(tracking_mod, "cli_version", "test-cli-version"),
        patch.object(tracking_mod, "tracing_id", "test-tracing-id"),
        patch.object(tracking_mod, "PROVIDERS", [fake_provider]),
        patch.object(tracking_mod, "_session_only_tracking", False),
    ):
        # Stash the mock on the module for convenient access from tests
        # without changing the fixture return contract.
        tracking_mod.provider = fake_provider  # type: ignore[attr-defined]
        try:
            yield tracking_mod
        finally:
            del tracking_mod.provider


def _last_track_call(provider):
    args, kwargs = provider.track.call_args
    # Provider.track(event_name, distinct_id=..., properties=...)
    event_name = args[0] if args else kwargs.get("event_name")
    distinct_id = kwargs.get("distinct_id", args[1] if len(args) > 1 else None)
    properties = kwargs.get("properties", args[2] if len(args) > 2 else {})
    return event_name, distinct_id, properties


class TestTrackEvent:
    def test_short_circuits_when_disabled(self, tracking_module):
        tracking_module.config_manager.set(constants.CONFIG_KEY_ENABLE_TRACKING, "False")
        tracking_module.track_event("some_event")
        tracking_module.provider.track.assert_not_called()

    def test_short_circuits_when_not_configured(self, tracking_module):
        tracking_module.track_event("some_event")
        tracking_module.provider.track.assert_not_called()

    def test_fires_when_enabled(self, tracking_module):
        tracking_module.config_manager.set(constants.CONFIG_KEY_ENABLE_TRACKING, "True")
        tracking_module.track_event("some_event", {"k": "v"})
        tracking_module.provider.track.assert_called_once()
        event_name, _, properties = _last_track_call(tracking_module.provider)
        assert event_name == "some_event"
        assert properties["k"] == "v"
        assert "cli_version" in properties
        assert "tracing_id" in properties

    def test_properties_default_to_empty_dict(self, tracking_module):
        tracking_module.config_manager.set(constants.CONFIG_KEY_ENABLE_TRACKING, "True")
        tracking_module.track_event("some_event")
        tracking_module.provider.track.assert_called_once()
        _, _, properties = _last_track_call(tracking_module.provider)
        assert set(properties.keys()) == {"cli_version", "tracing_id"}

    def test_swallows_provider_errors(self, tracking_module):
        tracking_module.config_manager.set(constants.CONFIG_KEY_ENABLE_TRACKING, "True")
        tracking_module.provider.track.side_effect = RuntimeError("boom")
        tracking_module.track_event("some_event")
        tracking_module.provider.track.assert_called_once()


class TestTrackCommandRedaction:
    """track_command must redact secret-bearing kwargs before they reach the tracking system."""

    def test_api_key_value_is_redacted(self, tracking_module):
        tracking_module.config_manager.set(constants.CONFIG_KEY_ENABLE_TRACKING, "True")

        @tracking_module.track_command()
        def some_cmd(workflow, api_key=None):
            return None

        some_cmd(workflow="wf.json", api_key="sk-supersecret")

        tracking_module.provider.track.assert_called_once()
        _, _, properties = _last_track_call(tracking_module.provider)
        assert properties["api_key"] == "<redacted>"
        assert properties["workflow"] == "wf.json"
        assert "sk-supersecret" not in str(properties)

    def test_api_key_none_stays_none(self, tracking_module):
        # When the user didn't pass --api-key (or set $COMFY_API_KEY), we still
        # want to be able to see in the analytics that it was absent — not a
        # "<redacted>" sentinel that would imply they did pass one.
        tracking_module.config_manager.set(constants.CONFIG_KEY_ENABLE_TRACKING, "True")

        @tracking_module.track_command()
        def some_cmd(workflow, api_key=None):
            return None

        some_cmd(workflow="wf.json", api_key=None)

        _, _, properties = _last_track_call(tracking_module.provider)
        assert properties["api_key"] is None

    def test_publish_token_and_changelog_values_are_redacted(self, tracking_module):
        tracking_module.config_manager.set(constants.CONFIG_KEY_ENABLE_TRACKING, "True")

        @tracking_module.track_command("publish")
        def publish(token=None, changelog=None, changelog_file=None):
            return None

        publish(token="pat-supersecret", changelog="## 1.0\n- fix things", changelog_file=None)

        _, _, properties = _last_track_call(tracking_module.provider)
        assert properties["token"] == "<redacted>"
        assert properties["changelog"] == "<redacted>"
        assert properties["changelog_file"] is None
        assert "pat-supersecret" not in str(properties)
        assert "fix things" not in str(properties)

    def test_set_civitai_api_token_is_redacted(self, tracking_module):
        tracking_module.config_manager.set(constants.CONFIG_KEY_ENABLE_TRACKING, "True")

        @tracking_module.track_command("model")
        def download(url, set_civitai_api_token=None, set_hf_api_token=None):
            return None

        download(url="https://example.com", set_civitai_api_token="civ-real-token")

        tracking_module.provider.track.assert_called_once()
        _, _, properties = _last_track_call(tracking_module.provider)
        assert properties["set_civitai_api_token"] == "<redacted>"
        assert "civ-real-token" not in str(properties)

    def test_set_hf_api_token_is_redacted(self, tracking_module):
        tracking_module.config_manager.set(constants.CONFIG_KEY_ENABLE_TRACKING, "True")

        @tracking_module.track_command("model")
        def download(url, set_civitai_api_token=None, set_hf_api_token=None):
            return None

        download(url="https://example.com", set_hf_api_token="hf_real-token")

        tracking_module.provider.track.assert_called_once()
        _, _, properties = _last_track_call(tracking_module.provider)
        assert properties["set_hf_api_token"] == "<redacted>"
        assert "hf_real-token" not in str(properties)

    def test_bare_token_kwarg_is_redacted(self, tracking_module):
        tracking_module.config_manager.set(constants.CONFIG_KEY_ENABLE_TRACKING, "True")

        @tracking_module.track_command()
        def some_cmd(workflow, token=None):
            return None

        some_cmd(workflow="wf.json", token="my-secret-token")

        tracking_module.provider.track.assert_called_once()
        _, _, properties = _last_track_call(tracking_module.provider)
        assert properties["token"] == "<redacted>"
        assert "my-secret-token" not in str(properties)

    def test_underscore_ctx_is_excluded(self, tracking_module):
        import click

        tracking_module.config_manager.set(constants.CONFIG_KEY_ENABLE_TRACKING, "True")

        @tracking_module.track_command("model")
        def download(_ctx, url, set_civitai_api_token=None):
            return None

        ctx = click.Context(click.Command("download"))
        download(_ctx=ctx, url="https://example.com")

        tracking_module.provider.track.assert_called_once()
        _, _, properties = _last_track_call(tracking_module.provider)
        assert "_ctx" not in properties
        assert properties["url"] == "https://example.com"

    def test_non_serializable_value_is_excluded(self, tracking_module):
        tracking_module.config_manager.set(constants.CONFIG_KEY_ENABLE_TRACKING, "True")

        @tracking_module.track_command()
        def some_cmd(workflow, callback=None):
            return None

        some_cmd(workflow="wf.json", callback=lambda x: x)

        tracking_module.provider.track.assert_called_once()
        _, _, properties = _last_track_call(tracking_module.provider)
        assert "callback" not in properties
        assert properties["workflow"] == "wf.json"

    def test_url_query_string_is_scrubbed(self, tracking_module):
        # CivitAI download links carry the API key as `?token=`.
        tracking_module.config_manager.set(constants.CONFIG_KEY_ENABLE_TRACKING, "True")

        @tracking_module.track_command("model")
        def download(url=None, relative_path=None):
            return None

        download(url="https://civitai.com/api/download/models/12345?token=civ-url-secret")

        _, _, properties = _last_track_call(tracking_module.provider)
        assert properties["url"] == "https://civitai.com/api/download/models/12345"
        assert "civ-url-secret" not in str(properties)

    def test_url_without_query_is_unchanged(self, tracking_module):
        tracking_module.config_manager.set(constants.CONFIG_KEY_ENABLE_TRACKING, "True")

        @tracking_module.track_command("model")
        def download(url=None):
            return None

        download(url="https://huggingface.co/org/repo/resolve/main/m.safetensors")

        _, _, properties = _last_track_call(tracking_module.provider)
        assert properties["url"] == "https://huggingface.co/org/repo/resolve/main/m.safetensors"


class TestSensitiveNameMatcher:
    @pytest.mark.parametrize(
        "name",
        [
            "api_key",
            "token",
            "password",
            "secret",
            "changelog",
            "set_civitai_api_token",
            "set_hf_api_token",
            "access_token",
            "client_secret",
            "admin_password",
            "API_KEY",
            "Set_HF_Api_Token",
        ],
    )
    def test_matches(self, name):
        import comfy_cli.tracking as tm

        assert tm._is_sensitive(name) is True

    @pytest.mark.parametrize("name", ["url", "workflow", "changelog_file", "max_tokens", "tokenizer", "relative_path"])
    def test_does_not_match(self, name):
        import comfy_cli.tracking as tm

        assert tm._is_sensitive(name) is False


class TestCliParamNameDriftGate:
    """BE-992 happened because credential flags were added after the redaction
    set was written. Walk the real CLI tree so the next one cannot land
    unredacted."""

    # Params whose names merely contain a credential-ish substring but are
    # reviewed as safe to track verbatim go here.
    ALLOWLIST = frozenset()

    def test_credentialish_cli_params_are_redacted(self):
        import click
        from typer.main import get_command

        import comfy_cli.tracking as tm
        from comfy_cli.cmdline import app

        suspicious = ("token", "secret", "password", "api_key", "apikey", "credential")

        def walk(cmd, path):
            if isinstance(cmd, click.Group):
                for name, sub in cmd.commands.items():
                    yield from walk(sub, [*path, name])
                return
            for param in cmd.params:
                if param.name:
                    yield " ".join(path), param.name

        offenders = sorted(
            {
                (path, pname)
                for path, pname in walk(get_command(app), ["comfy"])
                if any(s in pname.lower() for s in suspicious)
                and pname not in self.ALLOWLIST
                and not tm._is_sensitive(pname)
            }
        )
        assert offenders == [], f"credential-looking CLI params not redacted by _is_sensitive: {offenders}"


class TestTrackCommandRealTyperWiring:
    def test_model_download_kwargs_are_filtered_and_redacted(self, tracking_module):
        # `model download` is the command whose `_ctx` + credential kwarg
        # combination motivated BE-992; invoke it through Typer for real so
        # the Click context actually lands in the tracked kwargs.
        from typer.testing import CliRunner

        import comfy_cli.command.models.models as models

        tracking_module.config_manager.set(constants.CONFIG_KEY_ENABLE_TRACKING, "True")

        with (
            patch.object(models, "config_manager", MagicMock()),
            patch.object(models, "check_civitai_url", side_effect=RuntimeError("halt after tracking")),
        ):
            result = CliRunner().invoke(
                models.app,
                [
                    "download",
                    "--url",
                    "https://example.com/model.safetensors?token=url-secret",
                    "--set-civitai-api-token",
                    "civ-secret",
                ],
            )

        # The command body aborted at the patched helper, after tracking fired.
        assert isinstance(result.exception, RuntimeError)

        tracking_module.provider.track.assert_called_once()
        event_name, _, properties = _last_track_call(tracking_module.provider)
        assert event_name == "model:download"
        assert "_ctx" not in properties
        assert properties["set_civitai_api_token"] == "<redacted>"
        assert "civ-secret" not in str(properties)
        assert properties["url"] == "https://example.com/model.safetensors"
        assert "url-secret" not in str(properties)


class TestInitTrackingRoundTrip:
    """End-to-end: init_tracking() writes the string "False"/"True", and track_event honors it.

    Regression for a prior bug where track_event used config_manager.get(), which returned
    the raw string "False" (a truthy value), so disabling via this code path had no effect.
    """

    def test_disable_is_respected_by_track_event(self, tracking_module):
        tracking_module.init_tracking(False)
        tracking_module.track_event("some_event")
        tracking_module.provider.track.assert_not_called()

    def test_enable_is_respected_by_track_event(self, tracking_module):
        tracking_module.init_tracking(True)
        tracking_module.provider.track.reset_mock()
        tracking_module.track_event("some_event")
        tracking_module.provider.track.assert_called_once()

    def test_disable_persists_as_parseable_bool(self, tracking_module):
        tracking_module.init_tracking(False)
        assert tracking_module.config_manager.get_bool(constants.CONFIG_KEY_ENABLE_TRACKING) is False

    def test_enable_generates_user_id(self, tracking_module):
        assert tracking_module.config_manager.get(constants.CONFIG_KEY_USER_ID) is None
        tracking_module.init_tracking(True)
        generated_user_id = tracking_module.config_manager.get(constants.CONFIG_KEY_USER_ID)
        assert generated_user_id is not None
        assert tracking_module.user_id == generated_user_id
        _, distinct_id, _ = _last_track_call(tracking_module.provider)
        assert distinct_id == generated_user_id

    def test_disable_does_not_generate_user_id(self, tracking_module):
        tracking_module.init_tracking(False)
        assert tracking_module.config_manager.get(constants.CONFIG_KEY_USER_ID) is None

    def test_install_event_fires_once_across_calls(self, tracking_module):
        tracking_module.init_tracking(True)
        assert tracking_module.provider.track.call_count == 1
        tracking_module.init_tracking(True)
        assert tracking_module.provider.track.call_count == 1


class TestPromptTrackingConsent:
    def test_enables_session_only_when_stdin_not_tty(self, tracking_module):
        with (
            patch.object(tracking_module.sys.stdin, "isatty", return_value=False),
            patch.object(tracking_module.sys.stdout, "isatty", return_value=True),
            patch.object(tracking_module.ui, "prompt_confirm_action") as mock_prompt,
        ):
            tracking_module.prompt_tracking_consent()
        mock_prompt.assert_not_called()
        assert tracking_module.config_manager.get_bool(constants.CONFIG_KEY_ENABLE_TRACKING) is None
        assert tracking_module._session_only_tracking is True
        assert tracking_module.user_id is not None

    def test_enables_session_only_when_stdout_not_tty(self, tracking_module):
        with (
            patch.object(tracking_module.sys.stdin, "isatty", return_value=True),
            patch.object(tracking_module.sys.stdout, "isatty", return_value=False),
            patch.object(tracking_module.ui, "prompt_confirm_action") as mock_prompt,
        ):
            tracking_module.prompt_tracking_consent()
        mock_prompt.assert_not_called()
        assert tracking_module.config_manager.get_bool(constants.CONFIG_KEY_ENABLE_TRACKING) is None
        assert tracking_module._session_only_tracking is True

    def test_session_only_tracking_fires_track_event(self, tracking_module):
        with (
            patch.object(tracking_module.sys.stdin, "isatty", return_value=False),
            patch.object(tracking_module.sys.stdout, "isatty", return_value=False),
        ):
            tracking_module.prompt_tracking_consent()
        tracking_module.track_event("some_event", {"k": "v"})
        tracking_module.provider.track.assert_called_once()
        event_name, distinct_id, _ = _last_track_call(tracking_module.provider)
        assert event_name == "some_event"
        assert distinct_id is not None

    def test_session_only_persists_user_id(self, tracking_module):
        with (
            patch.object(tracking_module.sys.stdin, "isatty", return_value=False),
            patch.object(tracking_module.sys.stdout, "isatty", return_value=False),
        ):
            tracking_module.prompt_tracking_consent()
        persisted = tracking_module.config_manager.get(constants.CONFIG_KEY_USER_ID)
        assert persisted is not None
        assert persisted == tracking_module.user_id

    def test_session_only_survives_unwritable_config(self, tracking_module):
        # Read-only / missing config dir (fresh CI, restricted sandbox) must
        # not crash the caller mid-typer-callback — otherwise an agent gets
        # a Python traceback instead of a structured `failed` event.
        with (
            patch.object(tracking_module.sys.stdin, "isatty", return_value=False),
            patch.object(tracking_module.sys.stdout, "isatty", return_value=False),
            patch.object(tracking_module.config_manager, "set", side_effect=PermissionError("read-only fs")),
        ):
            tracking_module.prompt_tracking_consent()
        # In-memory state is still correct so this process tracks normally.
        assert tracking_module._session_only_tracking is True
        assert tracking_module.user_id is not None

    def test_session_only_reuses_existing_user_id(self, tracking_module):
        existing_id = "existing-uuid-from-prior-run"
        tracking_module.config_manager.set(constants.CONFIG_KEY_USER_ID, existing_id)
        with (
            patch.object(tracking_module, "user_id", existing_id),
            patch.object(tracking_module.sys.stdin, "isatty", return_value=False),
            patch.object(tracking_module.sys.stdout, "isatty", return_value=False),
        ):
            tracking_module.prompt_tracking_consent()
            assert tracking_module.user_id == existing_id
        assert tracking_module.config_manager.get(constants.CONFIG_KEY_USER_ID) == existing_id

    def test_prompts_when_both_are_tty(self, tracking_module):
        with (
            patch.object(tracking_module.sys.stdin, "isatty", return_value=True),
            patch.object(tracking_module.sys.stdout, "isatty", return_value=True),
            patch.object(tracking_module.ui, "prompt_confirm_action", return_value=False) as mock_prompt,
        ):
            tracking_module.prompt_tracking_consent()
        mock_prompt.assert_called_once()
        assert tracking_module.config_manager.get_bool(constants.CONFIG_KEY_ENABLE_TRACKING) is False
        assert tracking_module._session_only_tracking is False

    def test_skip_prompt_bypasses_tty_check(self, tracking_module):
        with (
            patch.object(tracking_module.sys.stdin, "isatty", return_value=False),
            patch.object(tracking_module.sys.stdout, "isatty", return_value=False),
            patch.object(tracking_module.ui, "prompt_confirm_action") as mock_prompt,
        ):
            tracking_module.prompt_tracking_consent(skip_prompt=True, default_value=False)
        mock_prompt.assert_not_called()
        assert tracking_module.config_manager.get_bool(constants.CONFIG_KEY_ENABLE_TRACKING) is False
        assert tracking_module._session_only_tracking is False

    def test_no_op_when_already_configured(self, tracking_module):
        tracking_module.config_manager.set(constants.CONFIG_KEY_ENABLE_TRACKING, "True")
        with (
            patch.object(tracking_module.sys.stdin, "isatty", return_value=False),
            patch.object(tracking_module.sys.stdout, "isatty", return_value=False),
            patch.object(tracking_module.ui, "prompt_confirm_action") as mock_prompt,
        ):
            tracking_module.prompt_tracking_consent()
        mock_prompt.assert_not_called()
        assert tracking_module.config_manager.get_bool(constants.CONFIG_KEY_ENABLE_TRACKING) is True
        assert tracking_module._session_only_tracking is False

    def test_session_only_is_idempotent(self, tracking_module):
        with (
            patch.object(tracking_module.sys.stdin, "isatty", return_value=False),
            patch.object(tracking_module.sys.stdout, "isatty", return_value=False),
        ):
            tracking_module.prompt_tracking_consent()
            first_user_id = tracking_module.user_id
            tracking_module.prompt_tracking_consent()
            assert tracking_module.user_id == first_user_id


class TestEnvVarOptOut:
    @pytest.mark.parametrize("env_var", ["DO_NOT_TRACK", "COMFY_NO_TELEMETRY"])
    def test_env_var_blocks_track_event_even_when_config_enabled(self, tracking_module, monkeypatch, env_var):
        tracking_module.config_manager.set(constants.CONFIG_KEY_ENABLE_TRACKING, "True")
        monkeypatch.setenv(env_var, "1")
        tracking_module.track_event("some_event", {"k": "v"})
        tracking_module.provider.track.assert_not_called()

    @pytest.mark.parametrize("env_var", ["DO_NOT_TRACK", "COMFY_NO_TELEMETRY"])
    def test_env_var_blocks_track_event_under_session_only(self, tracking_module, monkeypatch, env_var):
        monkeypatch.setenv(env_var, "1")
        with patch.object(tracking_module, "_session_only_tracking", True):
            tracking_module.track_event("some_event")
        tracking_module.provider.track.assert_not_called()

    @pytest.mark.parametrize("falsy", ["", "0"])
    def test_falsy_values_do_not_block(self, tracking_module, monkeypatch, falsy):
        tracking_module.config_manager.set(constants.CONFIG_KEY_ENABLE_TRACKING, "True")
        monkeypatch.setenv("DO_NOT_TRACK", falsy)
        monkeypatch.setenv("COMFY_NO_TELEMETRY", falsy)
        tracking_module.track_event("some_event")
        tracking_module.provider.track.assert_called_once()

    @pytest.mark.parametrize("env_var", ["DO_NOT_TRACK", "COMFY_NO_TELEMETRY"])
    def test_env_var_short_circuits_consent_prompt(self, tracking_module, monkeypatch, env_var):
        monkeypatch.setenv(env_var, "1")
        with (
            patch.object(tracking_module.sys.stdin, "isatty", return_value=True),
            patch.object(tracking_module.sys.stdout, "isatty", return_value=True),
            patch.object(tracking_module.ui, "prompt_confirm_action") as mock_prompt,
        ):
            tracking_module.prompt_tracking_consent()
        mock_prompt.assert_not_called()
        assert tracking_module.config_manager.get_bool(constants.CONFIG_KEY_ENABLE_TRACKING) is None

    @pytest.mark.parametrize("env_var", ["DO_NOT_TRACK", "COMFY_NO_TELEMETRY"])
    def test_env_var_blocks_non_tty_auto_enable_and_user_id_persist(self, tracking_module, monkeypatch, env_var):
        # Reporter's core concern (issue #462): in CI/Docker the non-TTY
        # branch silently persisted a UUID. Env var must skip that path.
        monkeypatch.setenv(env_var, "1")
        with (
            patch.object(tracking_module.sys.stdin, "isatty", return_value=False),
            patch.object(tracking_module.sys.stdout, "isatty", return_value=False),
        ):
            tracking_module.prompt_tracking_consent()
        assert tracking_module._session_only_tracking is False
        assert tracking_module.config_manager.get(constants.CONFIG_KEY_USER_ID) is None

    def test_env_var_does_not_overwrite_existing_consent(self, tracking_module, monkeypatch):
        # On-disk consent flag must survive an env-var-suppressed run so a
        # subsequent invocation without the env var keeps the user's choice.
        tracking_module.config_manager.set(constants.CONFIG_KEY_ENABLE_TRACKING, "True")
        monkeypatch.setenv("DO_NOT_TRACK", "1")
        tracking_module.prompt_tracking_consent()
        assert tracking_module.config_manager.get_bool(constants.CONFIG_KEY_ENABLE_TRACKING) is True


class TestTelemetryDisabledByEnvHelper:
    @pytest.fixture(autouse=True)
    def _clear_both(self, monkeypatch):
        monkeypatch.delenv("DO_NOT_TRACK", raising=False)
        monkeypatch.delenv("COMFY_NO_TELEMETRY", raising=False)

    def test_unset_returns_false(self, tracking_module):
        import comfy_cli.tracking as tm

        assert tm._telemetry_disabled_by_env() is False

    @pytest.mark.parametrize("env_var", ["DO_NOT_TRACK", "COMFY_NO_TELEMETRY"])
    @pytest.mark.parametrize(
        "value,expected",
        [
            # consoledonottrack.com spec: empty or "0" allows tracking; anything else opts out.
            ("", False),
            ("0", False),
            ("1", True),
            ("true", True),
            ("yes", True),
            ("00", True),
            ("false", True),
        ],
    )
    def test_value_semantics(self, tracking_module, monkeypatch, env_var, value, expected):
        import comfy_cli.tracking as tm

        monkeypatch.setenv(env_var, value)
        assert tm._telemetry_disabled_by_env() is expected

    def test_either_var_alone_is_sufficient(self, tracking_module, monkeypatch):
        import comfy_cli.tracking as tm

        monkeypatch.setenv("COMFY_NO_TELEMETRY", "1")
        assert tm._telemetry_disabled_by_env() is True
        monkeypatch.delenv("COMFY_NO_TELEMETRY")
        monkeypatch.setenv("DO_NOT_TRACK", "1")
        assert tm._telemetry_disabled_by_env() is True
