"""An unexpected exception inside a command still ends the --json stream with an envelope.

`comfy --json workflow set-widget ...` could escape with a Python traceback
and no envelope. Edit commands catch only ValueError, and nothing above them
turned any other exception into `ok:false`. So a --json caller got a stack
dump, an empty stdout and exit 1, which it cannot tell from a transport
failure.

These drive the real CLI, because the fix lives at the click entrypoint
(`_RootGroup.invoke`), the same place usage errors become envelopes.
"""

from __future__ import annotations

import json

import click
import pytest
import typer
from typer.testing import CliRunner

from comfy_cli import workflow_ops
from comfy_cli.cmdline import _internal_error_message, app


def _boom(*_a, **_kw):
    raise KeyError("widgets_values")


@pytest.fixture
def workflow_file(tmp_path):
    p = tmp_path / "wf.json"
    p.write_text(json.dumps({"nodes": [{"id": 1, "type": "VAELoader", "widgets_values": ["a"]}], "links": []}))
    return p


@pytest.fixture
def object_info(tmp_path):
    oi = tmp_path / "oi.json"
    oi.write_text(
        json.dumps(
            {
                "VAELoader": {
                    "input": {"required": {"vae_name": [["a", "b"], {}]}},
                    "input_order": {"required": ["vae_name"]},
                    "output": ["VAE"],
                    "output_name": ["VAE"],
                    "category": "loaders",
                    "display_name": "Load VAE",
                    "description": "",
                    "output_node": False,
                    "python_module": "nodes",
                }
            }
        )
    )
    return oi


def _set_widget(mode_flag: str, workflow_file, object_info):
    return CliRunner().invoke(
        app,
        [mode_flag, "workflow", "set-widget", str(workflow_file), "1.vae_name", "b", "--input", str(object_info)],
        env={"NO_COLOR": "1", "COLUMNS": "400"},
    )


def test_a_crash_in_a_command_ends_the_json_stream_with_an_envelope(monkeypatch, workflow_file, object_info):
    monkeypatch.setattr(workflow_ops, "set_widget", _boom)
    result = _set_widget("--json", workflow_file, object_info)
    assert result.exit_code == 1, result.output
    lines = [ln for ln in result.stdout.splitlines() if ln.strip()]
    assert lines, "stdout must not be empty"
    envelope = json.loads(lines[-1])
    assert envelope["ok"] is False
    err = envelope["error"]
    assert err["code"] == "internal_error", err
    assert err["details"]["exception"] == "KeyError", err
    assert "widgets_values" in err["message"], err
    assert envelope["command"] == "workflow set-widget", envelope


def test_pretty_mode_prints_no_envelope(monkeypatch, workflow_file, object_info):
    monkeypatch.setattr(workflow_ops, "set_widget", _boom)
    result = _set_widget("--no-json", workflow_file, object_info)
    assert result.exit_code == 1
    assert isinstance(result.exception, KeyError), "pretty mode keeps the crash as it was"
    assert "internal_error" not in result.stdout


def test_a_handled_refusal_is_not_relabelled(workflow_file, object_info):
    """typer.Exit after a renderer.error is control flow, not a crash."""
    result = CliRunner().invoke(
        app,
        ["--json", "workflow", "set-widget", str(workflow_file), "1.nope", "b", "--input", str(object_info)],
        env={"NO_COLOR": "1", "COLUMNS": "400"},
    )
    envelopes = [json.loads(ln) for ln in result.stdout.splitlines() if ln.strip().startswith("{")]
    assert [e["error"]["code"] for e in envelopes] == ["workflow_edit_invalid"], envelopes


def test_the_envelope_carries_a_short_traceback_tail(monkeypatch, workflow_file, object_info):
    monkeypatch.setattr(workflow_ops, "set_widget", _boom)
    envelope = json.loads(_set_widget("--json", workflow_file, object_info).stdout.strip().splitlines()[-1])
    frames = envelope["error"]["details"]["traceback"]
    assert 1 <= len(frames) <= 3, frames
    # file:line:func only, innermost last — no source text.
    assert frames[-1].endswith(":_boom"), frames
    assert all(f.count(":") >= 2 and "raise" not in f for f in frames), frames


def test_the_message_is_capped_and_secrets_are_redacted(monkeypatch, workflow_file, object_info):
    def leak(*_a, **_kw):
        raise RuntimeError(
            "GET https://api.comfy.org/x?api_key=sk-SECRET123&x=1 failed "
            "Authorization: Bearer eyJSECRETTOKEN token=abc123secret " + "y" * 2000
        )

    monkeypatch.setattr(workflow_ops, "set_widget", leak)
    err = json.loads(_set_widget("--json", workflow_file, object_info).stdout.strip().splitlines()[-1])["error"]
    msg = err["message"]
    assert len(msg) <= 520, len(msg)
    for secret in ("sk-SECRET123", "eyJSECRETTOKEN", "abc123secret"):
        assert secret not in json.dumps(err), err
    assert "https://api.comfy.org/x" in msg, "keep the URL path, drop only the query"


@pytest.mark.parametrize(
    "exc_factory, expected_exit",
    [
        pytest.param(lambda: typer.Exit(0), 0, id="typer-exit-0"),
        pytest.param(lambda: click.Abort(), 1, id="click-abort"),
    ],
)
def test_click_control_flow_is_never_relabelled(monkeypatch, workflow_file, object_info, exc_factory, expected_exit):
    """typer.Exit / Abort raised with NO prior envelope are control flow, not crashes."""

    def raise_it(*_a, **_kw):
        raise exc_factory()

    monkeypatch.setattr(workflow_ops, "set_widget", raise_it)
    result = _set_widget("--json", workflow_file, object_info)
    assert "internal_error" not in result.stdout, result.stdout
    assert result.exit_code == expected_exit, result.output


@pytest.mark.parametrize(
    "header",
    [
        "Authorization: Basic dXNlcjpwYXNz",
        "Authorization: Bearer abc.def-ghi",
        "authorization=Token dXNlcjpwYXNz",
        "Proxy-Authorization: Digest username=dXNlcjpwYXNz",
        "headers={'Authorization': 'Basic dXNlcjpwYXNz'}",
        'Authorization: Digest username="alice", realm="x", response="deadbeef00"',
        "headers={'Authorization': 'Digest username=\"alice\", response=\"deadbeef00\"'}",
    ],
)
def test_an_authorization_header_is_masked_scheme_and_credential(monkeypatch, workflow_file, object_info, header):
    """The auth scheme and its credential are one value. Masking only the
    scheme word (``Basic``) would leave the credential in the envelope."""

    def leak(*_a, **_kw):
        raise RuntimeError(f"request failed status=401\n{header}")

    monkeypatch.setattr(workflow_ops, "set_widget", leak)
    err = json.loads(_set_widget("--json", workflow_file, object_info).stdout.strip().splitlines()[-1])["error"]
    dumped = json.dumps(err)
    for secret in ("dXNlcjpwYXNz", "abc.def-ghi", "deadbeef00", "alice"):
        assert secret not in dumped, err
    assert "status=401" in err["message"], "text before the header survives"


@pytest.mark.parametrize(
    "text, kept",
    [
        (
            "{'Authorization': 'Bearer tok123', 'X-Request-Id': 'req-abc-789', 'Content-Type': 'application/json'}",
            ("'X-Request-Id': 'req-abc-789'", "'Content-Type': 'application/json'}"),
        ),
        (
            '{"Authorization": "Basic dG9rMTIz", "X-Request-Id": "req-abc-789"}',
            ('"X-Request-Id": "req-abc-789"}',),
        ),
        (
            "headers={'Authorization': 'Digest username=\"tok123\"'} status=401",
            ("} status=401",),
        ),
        ("Authorization: Token tok123\nX-Request-Id: req-abc-789", ("X-Request-Id: req-abc-789",)),
        (
            json.dumps({"Authorization": 'Digest username="x", response="tok123"', "X-Request-Id": "req-abc-789"}),
            ('"X-Request-Id": "req-abc-789"}',),
        ),
    ],
)
def test_an_authorization_value_is_masked_once_and_what_follows_survives(
    monkeypatch, workflow_file, object_info, text, kept
):
    """Masking the header value must not eat the rest of the line: the other
    headers and trailing text after it are what a --json caller acts on."""

    def leak(*_a, **_kw):
        raise RuntimeError(text)

    monkeypatch.setattr(workflow_ops, "set_widget", leak)
    err = json.loads(_set_widget("--json", workflow_file, object_info).stdout.strip().splitlines()[-1])["error"]
    for secret in ("tok123", "dG9rMTIz"):
        assert secret not in err["message"], err
    for fragment in kept:
        assert fragment in err["message"], err
    assert err["message"].count("***") == 1, err


def test_a_crash_in_the_root_callback_still_emits_the_envelope(monkeypatch, workflow_file, object_info):
    """The root callback crashing before it installs the --json renderer must
    still end stdout with an envelope, resolved from the parsed root flags."""
    from comfy_cli import cmdline

    class _BrokenConfig:
        def get_cli_version(self):
            raise OSError("config unreadable")

    monkeypatch.setattr(cmdline, "ConfigManager", _BrokenConfig)
    result = _set_widget("--json", workflow_file, object_info)
    assert result.exit_code == 1, result.output
    lines = [ln for ln in result.stdout.splitlines() if ln.strip()]
    assert lines, "stdout must not be empty"
    envelope = json.loads(lines[-1])
    assert envelope["ok"] is False
    assert envelope["error"]["code"] == "internal_error", envelope
    assert envelope["error"]["details"]["exception"] == "OSError", envelope


@pytest.mark.parametrize(
    ("message", "secret", "kept"),
    [
        ('body={\\"session_id\\": \\"abc123\\", \\"Cookie\\": \\"sid=xyz\\"}', "abc123", "Cookie"),
        ('body={"api_key_comfy_org":"sk-LIVE"}', "sk-LIVE", "body="),
        ("X-API-Key: sk-LIVE", "sk-LIVE", "X-API-Key:"),
        ('body={\\"api_key_comfy_org\\": \\"sk-LIVE\\"}', "sk-LIVE", "body="),
        ('api_key=\\"sk-LIVE\\" request=req-1', "sk-LIVE", "request=req-1"),
        ('api_key="ab\\"cd-LEAK" request=req-1', "cd-LEAK", "request=req-1"),
        ("headers={'cookie': None, 'x-request-id': 'req-abc'}", "None", "x-request-id"),
        ("Cookie: sid=a; remember_me=LONGTOKEN", "LONGTOKEN", "Cookie:"),
        ('Cookie: pref="x"; auth=LEAK', "LEAK", "Cookie:"),
        ("headers={'token': b'sk-LIVE'}", "sk-LIVE", "headers="),
        ("headers={'token': {'access': 'LEAK'}}", "LEAK", "headers="),
        ("params={'api_key': ['sk-LIVE']}", "sk-LIVE", "params="),
        ("params={'api_key': ['sk-LIVE', 'sk-BACKUP']}", "sk-BACKUP", "params="),
        ("api_key=[['sk-A'], 'sk-BACKUP']", "sk-BACKUP", "api_key="),
        ("api_key=[\n  'sk-BACKUP'", "sk-BACKUP", "api_key="),
        ("api_key=[\n'sk-BACKUP'] request=req-1", "sk-BACKUP", "request=req-1"),
        ("headers={'token': bytearray(b'sk-LIVE')}", "sk-LIVE", "headers="),
        ("token=ApiKey(value='sk-LIVE'", "sk-LIVE", "token="),
        ("api_key=ApiKey(value='sk-LIVE') request=req-1", "sk-LIVE", "request=req-1"),
        (
            "api_key=ApiKey(value=SecretStr('sk-A'), backup='sk-B') request=req-1",
            "sk-B",
            "request=req-1",
        ),
        (
            "api_key=Credentials(note='old (expired)', value='sk-LIVE') request=req-1",
            "sk-LIVE",
            "request=req-1",
        ),
        ("api_key=ApiKey(value='sk)LEAK') request=req-1", "sk)LEAK", "request=req-1"),
        (r"body={\"api_key\": [\"sk-A\", \"sk-B\"]}", "sk-B", "body="),
        (
            "Set-Cookie: sid=abc; expires=Wed, 09 Jun 2021 10:18:14 GMT, remember_me=LEAK",
            "LEAK",
            "Set-Cookie:",
        ),
        ('Cookie: "sid=abc", remember_me=LEAK', "LEAK", "Cookie:"),
        ("Set-Cookie: sid=abc;\n remember_me=LEAKTOKEN", "LEAKTOKEN", "Set-Cookie:"),
        ("headers={'Set-Cookie': ['sid=abc', 'remember_me=LEAK']}", "LEAK", "headers="),
        ("cookie=abc123sid path=/x", "abc123sid", "cookie="),
        ("token=Bearer-sk-LIVE request=req-1", "sk-LIVE", "request=req-1"),
        ('body={\\"x-api-key\\": 123456789}', "123456789", "body="),
        ('body={\\"password\\": \\"a\\nb\\"}', "a\\nb", "body="),
        (r"password=C:\Users\bob", r"Users\bob", "password="),
        ("GET https://alice:p@ssword@example.com/x failed", "p@ssword", "example.com/x"),
        ("GET https://user:pa,ss@example.com/x failed", "pa,ss", "example.com/x"),
    ],
)
def test_internal_error_scrubber_handles_reviewed_secret_shapes(message, secret, kept):
    scrubbed = _internal_error_message(RuntimeError(message))
    assert secret not in scrubbed
    assert kept in scrubbed


def test_internal_error_scrubber_handles_unprintable_exception():
    class UnprintableError(Exception):
        def __str__(self):
            raise RuntimeError("cannot stringify")

    assert _internal_error_message(UnprintableError()) == "UnprintableError: unprintable exception"


def test_internal_error_scrubber_does_not_treat_query_at_as_userinfo():
    message = "amqp://host?opt=user@example.com"
    assert message in _internal_error_message(RuntimeError(message))


def test_internal_error_scrubber_preserves_ordinary_identifier_diagnostics():
    message = "sigma=0.8 max_tokens=100 sidecar=on signal: 9"
    assert message in _internal_error_message(RuntimeError(message))


def test_internal_error_scrubber_marks_truncated_messages():
    message = "x" * 380 + " https://alice:password@example.com/" + "y" * 200
    scrubbed = _internal_error_message(RuntimeError(message))
    assert "password" not in scrubbed
    assert scrubbed.endswith("…")


def test_internal_error_scrubber_drops_a_credential_split_at_the_input_cap():
    message = "Bearer " + "A" * 3_800 + " https://alice:" + "S" * 300 + "@example.com/x"
    scrubbed = _internal_error_message(RuntimeError(message))
    assert "alice:" not in scrubbed
    assert "S" * 20 not in scrubbed
    assert scrubbed.endswith("…")


def test_internal_error_scrubber_preserves_text_after_an_unquoted_value():
    scrubbed = _internal_error_message(RuntimeError("bad option 'token=abc' given; retry later"))
    assert "abc" not in scrubbed
    assert "bad option 'token=***' given; retry later" in scrubbed


def test_internal_error_scrubber_preserves_explanation_after_an_unquoted_value():
    scrubbed = _internal_error_message(RuntimeError("token=abc123 (expired at 12:00)"))
    assert "abc123" not in scrubbed
    assert "token=*** (expired at 12:00)" in scrubbed


def test_internal_error_scrubber_does_not_cross_a_newline_to_find_a_container():
    message = "invalid token:\n  (line 3) unexpected character\n  node id: 7"
    scrubbed = _internal_error_message(RuntimeError(message))
    assert message in scrubbed


def test_internal_error_scrubber_handles_unterminated_escaped_container_with_backslashes():
    secret = "sk-LIVE" + "\\" * 1_000
    scrubbed = _internal_error_message(RuntimeError(r"body={\"api_key\": [\"" + secret))
    assert "sk-LIVE" not in scrubbed


def test_internal_error_scrubber_bounds_long_non_secret_key_scans():
    import time

    message = "a_" * 2_000
    started = time.perf_counter()
    _internal_error_message(RuntimeError(message))
    assert time.perf_counter() - started < 0.2
