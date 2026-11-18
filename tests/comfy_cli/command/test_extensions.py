"""Tests for `comfy extension list|disable|enable`.

The commands read and write the frontend's `Comfy.Extension.DisabledPacks`
setting through the running ComfyUI's `/settings/{id}` endpoint, and read the
served scripts from `/extensions`. All HTTP is mocked at the
`comfy_client._OPENER` seam, like `test_system.py`.
"""

from __future__ import annotations

import json
import urllib.error
from unittest.mock import patch

import pytest
import typer

from comfy_cli import comfy_client
from comfy_cli.caller import Caller
from comfy_cli.command import extensions as ext_cmd
from comfy_cli.output.renderer import OutputMode, Renderer, reset_renderer_for_testing, set_renderer
from comfy_cli.target import Target

LOCAL_TARGET = Target(kind="local", base_url="http://127.0.0.1:8188", path_prefix="", host="127.0.0.1", port=8188)
SETTING_URL = "http://127.0.0.1:8188/settings/Comfy.Extension.DisabledPacks"

SERVED = [
    "/extensions/core/clipspace.js",
    "/extensions/core/groupNode.js",
    "/extensions/agent-cushions/cushions.js",
    "/extensions/agent-cushions/sidebar.js",
    "/extensions/ComfyUI-VideoHelperSuite/videoinfo.js",
    "/extensions/My%20Pack/main.js",
]


@pytest.fixture(autouse=True)
def reset_singleton():
    reset_renderer_for_testing()
    yield
    reset_renderer_for_testing()


class _FakeResp:
    def __init__(self, body: bytes):
        self._body = body

    def read(self, n: int | None = None) -> bytes:
        return self._body if n is None else self._body[:n]

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class _FakeComfy:
    """A ComfyUI that serves `/extensions` and one settings key, and records writes."""

    def __init__(self, served=SERVED, disabled=None, users=None, frontend="1.59.6", changed_before_write=None):
        self.served = served
        self.disabled = disabled
        self.users = users if users is not None else {"storage": "server", "migrated": True}
        self.frontend = frontend
        self.changed_before_write = changed_before_write
        self.setting_reads = 0
        self.writes: list = []
        self.user_headers: list = []

    def open(self, req, timeout=None):
        url = req.full_url
        method = req.get_method()
        if url == "http://127.0.0.1:8188/users" and method == "GET":
            return _FakeResp(json.dumps(self.users).encode())
        if url == "http://127.0.0.1:8188/system_stats" and method == "GET":
            return _FakeResp(json.dumps({"system": {"required_frontend_version": self.frontend}}).encode())
        if url.startswith(SETTING_URL):
            self.user_headers.append(req.get_header("Comfy-user"))
        if url == SETTING_URL and method == "GET":
            self.setting_reads += 1
            if self.setting_reads == 2 and self.changed_before_write is not None:
                self.disabled = self.changed_before_write
            return _FakeResp(json.dumps(self.disabled).encode())
        if url == "http://127.0.0.1:8188/extensions" and method == "GET":
            return _FakeResp(json.dumps(self.served).encode())
        if url == SETTING_URL and method == "GET":
            return _FakeResp(json.dumps(self.disabled).encode())
        if url == SETTING_URL and method == "POST":
            value = json.loads(req.data.decode())
            self.writes.append(value)
            self.disabled = value
            return _FakeResp(b"")
        raise AssertionError(f"unexpected request {method} {url}")


def _force_renderer() -> Renderer:
    r = Renderer.resolve(
        is_stdout_tty=False,
        env={},
        caller=Caller(kind="user", agentic=False, source_env=None),
        json_flag=True,
    )
    r.mode = OutputMode.JSON
    set_renderer(r)
    return r


def _envelope(capsys: pytest.CaptureFixture[str]) -> dict:
    captured = capsys.readouterr().out
    assert captured.strip(), "no envelope on stdout"
    return json.loads(captured.strip().splitlines()[-1])


def _run(fn, fake, *args, **kwargs):
    renderer = _force_renderer()
    with (
        patch("comfy_cli.command.extensions._target", return_value=LOCAL_TARGET),
        patch.object(comfy_client._OPENER, "open", side_effect=fake.open),
    ):
        fn(renderer, *args, **kwargs)


class TestExtensionPath:
    def test_groups_scripts_by_folder(self):
        assert ext_cmd.extension_path("/extensions/agent-cushions/cushions.js") == "/extensions/agent-cushions/"

    def test_skips_core(self):
        assert ext_cmd.extension_path("/extensions/core/clipspace.js") is None

    def test_skips_scripts_outside_a_folder(self):
        assert ext_cmd.extension_path("/extensions/loose.js") is None
        assert ext_cmd.extension_path("/scripts/app.js") is None


class TestList:
    def test_lists_folders_with_state(self, capsys):
        fake = _FakeComfy(disabled=["/extensions/agent-cushions/"])
        _run(ext_cmd.list_execute, fake)
        env = _envelope(capsys)
        assert env["ok"] is True
        assert env["command"] == "extension list"
        rows = {row["folder"]: row for row in env["data"]["extensions"]}
        assert set(rows) == {"agent-cushions", "ComfyUI-VideoHelperSuite", "My Pack"}
        assert rows["agent-cushions"] == {
            "folder": "agent-cushions",
            "path": "/extensions/agent-cushions/",
            "scripts": 2,
            "enabled": False,
            "served": True,
        }
        assert rows["My Pack"]["path"] == "/extensions/My%20Pack/"
        assert rows["ComfyUI-VideoHelperSuite"]["enabled"] is True
        assert fake.writes == []

    def test_unset_setting_means_nothing_disabled(self, capsys):
        _run(ext_cmd.list_execute, _FakeComfy(disabled=None))
        env = _envelope(capsys)
        assert all(row["enabled"] for row in env["data"]["extensions"])

    def test_shows_disabled_folders_that_are_no_longer_served(self, capsys):
        _run(ext_cmd.list_execute, _FakeComfy(disabled=["/extensions/removed-pack/"]))
        rows = {row["folder"]: row for row in _envelope(capsys)["data"]["extensions"]}
        assert rows["removed-pack"]["served"] is False
        assert rows["removed-pack"]["enabled"] is False
        assert rows["removed-pack"]["scripts"] == 0

    def test_server_down_is_server_not_running(self, capsys):
        renderer = _force_renderer()
        with (
            patch("comfy_cli.command.extensions._target", return_value=LOCAL_TARGET),
            patch.object(comfy_client._OPENER, "open", side_effect=urllib.error.URLError("Connection refused")),
        ):
            with pytest.raises(typer.Exit) as exc_info:
                ext_cmd.list_execute(renderer)
        assert exc_info.value.exit_code == 1
        assert _envelope(capsys)["error"]["code"] == "server_not_running"

    def test_malformed_setting_is_reported_not_guessed(self, capsys):
        with pytest.raises(typer.Exit):
            _run(ext_cmd.list_execute, _FakeComfy(disabled="oops"))
        assert _envelope(capsys)["error"]["code"] == "invalid_response"


class TestDisable:
    def test_adds_the_folder_path_and_keeps_existing_entries(self, capsys):
        fake = _FakeComfy(disabled=["/extensions/other/"])
        _run(ext_cmd.toggle_execute, fake, "agent-cushions", enabled=False)
        assert fake.writes == [["/extensions/other/", "/extensions/agent-cushions/"]]
        env = _envelope(capsys)
        assert env["command"] == "extension disable"
        assert env["changed"] is True
        assert env["data"]["path"] == "/extensions/agent-cushions/"
        assert env["data"]["enabled"] is False
        assert "Reload" in env["data"]["note"]

    def test_accepts_the_path_form(self, capsys):
        fake = _FakeComfy(disabled=[])
        _run(ext_cmd.toggle_execute, fake, "/extensions/agent-cushions/", enabled=False)
        assert fake.writes == [["/extensions/agent-cushions/"]]

    def test_matches_a_percent_encoded_folder_by_its_name(self, capsys):
        fake = _FakeComfy(disabled=[])
        _run(ext_cmd.toggle_execute, fake, "My Pack", enabled=False)
        assert fake.writes == [["/extensions/My%20Pack/"]]

    def test_already_disabled_writes_nothing(self, capsys):
        fake = _FakeComfy(disabled=["/extensions/agent-cushions/"])
        _run(ext_cmd.toggle_execute, fake, "agent-cushions", enabled=False)
        assert fake.writes == []
        assert _envelope(capsys)["changed"] is False

    def test_unknown_folder_is_refused(self, capsys):
        fake = _FakeComfy(disabled=[])
        with pytest.raises(typer.Exit):
            _run(ext_cmd.toggle_execute, fake, "agent-cushion", enabled=False)
        assert fake.writes == []
        env = _envelope(capsys)
        assert env["error"]["code"] == "invalid_argument"
        assert "agent-cushions" in env["error"]["hint"]

    def test_core_cannot_be_disabled(self, capsys):
        fake = _FakeComfy(disabled=[])
        with pytest.raises(typer.Exit):
            _run(ext_cmd.toggle_execute, fake, "core", enabled=False)
        assert fake.writes == []

    @pytest.mark.parametrize("bad", ["", ".", "..", "a/b", "a\\b"])
    def test_path_like_names_are_refused(self, capsys, bad):
        fake = _FakeComfy(disabled=[])
        with pytest.raises(typer.Exit):
            _run(ext_cmd.toggle_execute, fake, bad, enabled=False)
        assert fake.writes == []
        assert _envelope(capsys)["error"]["code"] == "invalid_argument"

    def test_malformed_setting_is_never_overwritten(self, capsys):
        fake = _FakeComfy(disabled={"not": "a list"})
        with pytest.raises(typer.Exit):
            _run(ext_cmd.toggle_execute, fake, "agent-cushions", enabled=False)
        assert fake.writes == []
        assert _envelope(capsys)["error"]["code"] == "invalid_response"


class TestReviewFindings:
    def test_non_canonical_disabled_entries_do_not_crash_or_match(self, capsys):
        fake = _FakeComfy(disabled=["my-pack", "", "/foo", "/x/agent-cushions/bar"])
        with pytest.raises(typer.Exit):
            _run(ext_cmd.toggle_execute, fake, "my-pack", enabled=True)
        assert _envelope(capsys)["error"]["code"] == "invalid_argument"
        assert fake.writes == []

    def test_enable_of_an_unknown_folder_is_refused(self, capsys):
        fake = _FakeComfy(disabled=["/extensions/agent-cushions/"])
        with pytest.raises(typer.Exit):
            _run(ext_cmd.toggle_execute, fake, "agent-cushion", enabled=True)
        assert fake.writes == []
        assert _envelope(capsys)["error"]["code"] == "invalid_argument"

    def test_path_form_with_extra_slashes_is_normalized(self, capsys):
        fake = _FakeComfy(disabled=[])
        _run(ext_cmd.toggle_execute, fake, "/extensions/agent-cushions//", enabled=False)
        assert fake.writes == [["/extensions/agent-cushions/"]]

    def test_non_json_reply_is_invalid_response(self, capsys):
        renderer = _force_renderer()
        with (
            patch("comfy_cli.command.extensions._target", return_value=LOCAL_TARGET),
            patch.object(comfy_client._OPENER, "open", side_effect=lambda req, timeout=None: _FakeResp(b"<html>")),
        ):
            with pytest.raises(typer.Exit):
                ext_cmd.list_execute(renderer)
        assert _envelope(capsys)["error"]["code"] == "invalid_response"

    def test_http_status_is_not_reported_as_server_down(self, capsys):
        def _raise(req, timeout=None):
            if req.full_url.endswith("/users"):
                return _FakeResp(b'{"storage": "server"}')
            raise urllib.error.HTTPError(req.full_url, 403, "Forbidden", {}, None)

        renderer = _force_renderer()
        with (
            patch("comfy_cli.command.extensions._target", return_value=LOCAL_TARGET),
            patch.object(comfy_client._OPENER, "open", side_effect=_raise),
        ):
            with pytest.raises(typer.Exit):
                ext_cmd.list_execute(renderer)
        err = _envelope(capsys)["error"]
        assert err["code"] == "client_error"
        assert err["details"]["status"] == 403

    def test_a_change_made_before_the_write_is_kept(self, capsys):
        fake = _FakeComfy(disabled=[], changed_before_write=["/extensions/ComfyUI-VideoHelperSuite/"])
        _run(ext_cmd.toggle_execute, fake, "agent-cushions", enabled=False)
        assert fake.writes == [["/extensions/ComfyUI-VideoHelperSuite/", "/extensions/agent-cushions/"]]

    def test_multi_user_without_user_is_refused(self, capsys):
        fake = _FakeComfy(disabled=[], users={"storage": "server", "users": {"alice_1": "alice"}})
        with pytest.raises(typer.Exit):
            _run(ext_cmd.toggle_execute, fake, "agent-cushions", enabled=False)
        assert fake.writes == []
        err = _envelope(capsys)["error"]
        assert err["code"] == "usage_error"
        assert "alice_1" in err["hint"]

    def test_multi_user_with_user_sends_the_header(self, capsys):
        fake = _FakeComfy(disabled=[], users={"storage": "server", "users": {"alice_1": "alice"}})
        _run(ext_cmd.toggle_execute, fake, "agent-cushions", enabled=False, user="alice_1")
        assert fake.writes == [["/extensions/agent-cushions/"]]
        assert set(fake.user_headers) == {"alice_1"}

    def test_old_frontend_pin_warns(self, capsys):
        fake = _FakeComfy(disabled=[], frontend="1.58.5")
        _run(ext_cmd.toggle_execute, fake, "agent-cushions", enabled=False)
        env = _envelope(capsys)
        assert env["ok"] is True
        assert "1.59" in env["data"]["warnings"][0]

    def test_current_frontend_pin_does_not_warn(self, capsys):
        _run(ext_cmd.toggle_execute, _FakeComfy(disabled=[]), "agent-cushions", enabled=False)
        assert "warnings" not in _envelope(capsys)["data"]

    def test_messages_are_not_markup_escaped_in_json(self, capsys):
        with pytest.raises(typer.Exit):
            _run(ext_cmd.toggle_execute, _FakeComfy(disabled=[]), "[pack]", enabled=False)
        assert "\\[" not in _envelope(capsys)["error"]["message"]


class TestEnable:
    def test_removes_only_that_folder(self, capsys):
        fake = _FakeComfy(disabled=["/extensions/other/", "/extensions/agent-cushions/"])
        _run(ext_cmd.toggle_execute, fake, "agent-cushions", enabled=True)
        assert fake.writes == [["/extensions/other/"]]
        env = _envelope(capsys)
        assert env["command"] == "extension enable"
        assert env["changed"] is True
        assert env["data"]["enabled"] is True

    def test_enables_a_folder_that_is_no_longer_served(self, capsys):
        fake = _FakeComfy(disabled=["/extensions/removed-pack/"])
        _run(ext_cmd.toggle_execute, fake, "removed-pack", enabled=True)
        assert fake.writes == [[]]

    def test_already_enabled_writes_nothing(self, capsys):
        fake = _FakeComfy(disabled=[])
        _run(ext_cmd.toggle_execute, fake, "agent-cushions", enabled=True)
        assert fake.writes == []
        assert _envelope(capsys)["changed"] is False
