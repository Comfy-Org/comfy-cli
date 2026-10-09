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
from comfy_cli.cmdline import _INTERNAL_ERROR_SCRUB_INPUT_CAP, _internal_error_message, app


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
        ("auth=('alice', 'hunter2') request=req-1", "hunter2", "request=req-1"),
        ("cookies={'remember_me': 'LEAKTOKEN'} request=req-1", "LEAKTOKEN", "request=req-1"),
        ("API key: sk-LIVE request=req-1", "sk-LIVE", "request=req-1"),
        ("private_key=sk-LIVE request=req-1", "sk-LIVE", "request=req-1"),
        ("signing_key=sk-LIVE request=req-1", "sk-LIVE", "request=req-1"),
        ("access_key=sk-LIVE request=req-1", "sk-LIVE", "request=req-1"),
        ("aws_access_key_id=sk-LIVE request=req-1", "sk-LIVE", "request=req-1"),
        (r"body={\"private_key\": \"sk-LIVE\"}", "sk-LIVE", "body="),
        ("access token: sk-LIVE request=req-1", "sk-LIVE", "request=req-1"),
        ("token=<ApiKey value='sk-LIVE'> request=req-1", "sk-LIVE", "request=req-1"),
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
    message = (
        "sigma=0.8 max_tokens=100 sidecar=on signal: 9 "
        "input_token_count=8192 signature_algorithm=ed25519 password_length=32"
    )
    assert message in _internal_error_message(RuntimeError(message))


@pytest.mark.parametrize(
    "message",
    [
        "missing required key: num_inference_steps",
        "--param must be key=value, got 'x'",
        "key: Foo(value)",
        'primary_key: "user-42"',
        "foreign_key=value sort_key=value cache_key=value hash-key=value",
        "primary_key_id=42 foreign_key_id=7 cache_key_value=data sort_key_id=3 node_primary_key=1",
        "partitionKey=us-east-1 objectKey: artifact idempotencyKey=req-1 translationKey=home.title nodeSortKey=4",
        "methodSignature=(self, x) -> None",
    ],
)
def test_internal_error_scrubber_preserves_standalone_key_diagnostics(message):
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


def test_internal_error_scrubber_drops_split_userinfo_after_earlier_redaction_contracts_text():
    message = "Bearer " + "A" * 3_400 + " https://alice:" + "S" * 700 + "@example.com"
    scrubbed = _internal_error_message(RuntimeError(message))
    assert "alice:" not in scrubbed
    assert "S" * 20 not in scrubbed
    assert scrubbed.endswith("…")


@pytest.mark.parametrize(
    "tail",
    [
        "Invalid URL 'https://alice:" + "S" * 700,
        "url=https://alice:" + "S" * 700,
        "db=postgres://alice:p@ss" + "S" * 700,
    ],
)
def test_internal_error_scrubber_drops_any_anchorless_userinfo_tail(tail):
    message = "Bearer " + "A" * 3_400 + " " + tail
    scrubbed = _internal_error_message(RuntimeError(message))
    assert "alice:" not in scrubbed
    assert "S" * 20 not in scrubbed


@pytest.mark.parametrize(
    "message",
    [
        "secret key: sk-LIVE",
        "signing key = sk-LIVE",
        "private key: -----BEGIN " + "RSA PRIVATE KEY-----",
        "clientSecret=sk-LIVE",
        "ClientSecret=sk-LIVE",
        "privateKey=sk-LIVE",
        "secretAccessKey=sk-LIVE",
        "consumer_key=sk-LIVE",
        "subscription_key=sk-LIVE",
        "Ocp-Apim-Subscription-Key: sk-LIVE",
        "master_key=sk-LIVE",
        "hmac_key=sk-LIVE",
        "app_key=sk-LIVE",
        "encryption_key=sk-LIVE",
        "masterKey=sk-LIVE",
        "subscriptionKey=sk-LIVE",
        "hmacKey=sk-LIVE",
        "appKey=sk-LIVE",
        "encryptionKey=sk-LIVE",
        "apiKeyBackup=sk-LIVE",
        "accessTokenV2=sk-LIVE",
        "clientSecretBackup=sk-LIVE",
        "api_key_backup=sk-LIVE",
        "token_v2=sk-LIVE",
        "api_key_2=sk-LIVE",
        "private_key_passphrase=hunter2",
        "passwd=hunter2",
        "pwd=hunter2",
    ],
)
def test_internal_error_scrubber_masks_spaced_and_camel_case_credentials(message):
    scrubbed = _internal_error_message(RuntimeError(message))
    assert "sk-LIVE" not in scrubbed
    assert "PRIVATE " + "KEY-----" not in scrubbed


@pytest.mark.parametrize(
    "message",
    [
        "credentials=BasicAuth('alice', 'hunter2')",
        "credentials=('alice', 'hunter2')",
        "creds={'username': 'alice', 'password': 'hunter2'}",
        "jwt=eyJhbGciOiJIUzI1NiJ9.secret",
        "oauth=oauth-secret",
    ],
)
def test_internal_error_scrubber_masks_bare_credential_heads(message):
    scrubbed = _internal_error_message(RuntimeError(message))
    assert "alice" not in scrubbed
    assert "hunter2" not in scrubbed
    assert "eyJhbGci" not in scrubbed
    assert "oauth-secret" not in scrubbed


def test_internal_error_scrubber_masks_unlabelled_pem_and_preserves_following_diagnostic():
    message = (
        "invalid key: -----BEGIN OPENSSH PRIVATE "
        "KEY-----\nb3BlbnNzaC1rZXk=\n-----END OPENSSH PRIVATE "
        "KEY-----\nCaused by: HTTP 502 from the proxy"
    )

    scrubbed = _internal_error_message(RuntimeError(message))

    assert "b3BlbnNzaC1rZXk" not in scrubbed
    assert "-----BEGIN" not in scrubbed
    assert "invalid key: ***" in scrubbed
    assert "Caused by: HTTP 502 from the proxy" in scrubbed


def test_internal_error_scrubber_masks_pem_without_any_key_assignment():
    message = "could not deserialize -----BEGIN PRIVATE " + "KEY-----\nc2VjcmV0"
    scrubbed = _internal_error_message(RuntimeError(message))
    assert "c2VjcmV0" not in scrubbed
    assert "could not deserialize ***" in scrubbed


def test_internal_error_scrubber_masks_single_line_pem_body():
    message = "could not deserialize -----BEGIN PRIVATE " + "KEY-----MIIEvQsecret"
    scrubbed = _internal_error_message(RuntimeError(message))
    assert "MIIEvQsecret" not in scrubbed
    assert "could not deserialize ***" in scrubbed


def test_internal_error_scrubber_preserves_diagnostics_after_single_line_pem():
    message = (
        "invalid key: -----BEGIN PRIVATE KEY-----MIISECRET-----END PRIVATE KEY-----\nCaused by: HTTP 502 from the proxy"
    )
    scrubbed = _internal_error_message(RuntimeError(message))
    assert "MIISECRET" not in scrubbed
    assert "invalid key: ***" in scrubbed
    assert "Caused by: HTTP 502 from the proxy" in scrubbed


def test_internal_error_scrubber_masks_cr_only_pem():
    message = "invalid key: -----BEGIN PRIVATE KEY-----\rMIISECRET\r-----END PRIVATE KEY-----\rrequest=req-1"
    scrubbed = _internal_error_message(RuntimeError(message))
    assert "MIISECRET" not in scrubbed
    assert "request=req-1" in scrubbed


@pytest.mark.parametrize("label", ["PGP PRIVATE KEY BLOCK", "PGP SECRET KEY BLOCK", "OPENVPN STATIC KEY V1"])
def test_internal_error_scrubber_masks_armored_private_key_labels(label):
    message = f"could not import -----BEGIN {label}-----\nc2VjcmV0"
    scrubbed = _internal_error_message(RuntimeError(message))
    assert "c2VjcmV0" not in scrubbed
    assert "could not import ***" in scrubbed


def test_internal_error_scrubber_preserves_host_ports_and_complete_userinfo_at_the_input_cap():
    cases = [
        (" retrying https://api.comfy.org:8443", "RuntimeError: Bearer *** retrying https://api.comfy.org:8443…"),
        (" retrying http://localhost:8188/prompt", "RuntimeError: Bearer *** retrying http://localhost:8188/prompt…"),
        (" retrying https://alice:secret@example.com", "RuntimeError: Bearer *** retrying …"),
    ]
    for tail, expected in cases:
        prefix = "Bearer " + "A" * (
            _INTERNAL_ERROR_SCRUB_INPUT_CAP - len("RuntimeError: ") - len("Bearer ") - len(tail)
        )
        scrubbed = _internal_error_message(RuntimeError(prefix + tail + " overflow"))
        assert scrubbed == expected


@pytest.mark.parametrize(
    "tail",
    [
        " https://alice:p@ssw0",
        " redis://:SECRET",
        " https://alice:p:ss",
    ],
)
def test_internal_error_scrubber_drops_incomplete_userinfo_that_looks_like_a_host(tail):
    prefix = "Bearer " + "A" * (_INTERNAL_ERROR_SCRUB_INPUT_CAP - len("RuntimeError: ") - len("Bearer ") - len(tail))
    scrubbed = _internal_error_message(RuntimeError(prefix + tail + " overflow"))
    assert tail.strip() not in scrubbed
    assert scrubbed.endswith("…")


@pytest.mark.parametrize(
    "tail",
    [
        " https://ghp_LIVE_TOKEN",
        " https://john.doe:123456",
        " https://alice:p@secret.example",
    ],
)
def test_internal_error_scrubber_drops_ambiguous_userinfo_at_the_input_cap(tail):
    prefix = "Bearer " + "A" * (_INTERNAL_ERROR_SCRUB_INPUT_CAP - len("RuntimeError: ") - len("Bearer ") - len(tail))
    scrubbed = _internal_error_message(RuntimeError(prefix + tail + " overflow"))
    assert tail.strip() not in scrubbed
    assert scrubbed.endswith("…")


@pytest.mark.parametrize(
    "message",
    [
        "postgres://alice:pa#ss@db/x",
        "https://alice:pa#ss@example.com",
        "redis://:p?ss@cache",
    ],
)
def test_internal_error_scrubber_masks_userinfo_password_punctuation(message):
    scrubbed = _internal_error_message(RuntimeError(message))
    assert "pa#ss" not in scrubbed
    assert "p?ss" not in scrubbed
    assert "://***@" in scrubbed


@pytest.mark.parametrize(
    ("message", "leaked"),
    [
        ("https://alice:pa?ss@example.com/x", "alice:pa"),
        ("https://alice:pa/ss@example.com/x", "pa/ss"),
        ("https://alice:p@ssword@example.com/x", "ssword"),
    ],
)
def test_internal_error_scrubber_masks_http_userinfo_before_query_parsing(message, leaked):
    scrubbed = _internal_error_message(RuntimeError(message))
    assert leaked not in scrubbed
    assert "://***@" in scrubbed


@pytest.mark.parametrize(
    "tail",
    [
        " https://alice:pa?ss",
        " https://alice:pa#ss",
        " https://alice:pa/ss",
        " https://john.doe:12345",
    ],
)
def test_internal_error_scrubber_drops_truncated_punctuation_and_port_shaped_passwords(tail):
    prefix = "Bearer " + "A" * (_INTERNAL_ERROR_SCRUB_INPUT_CAP - len("RuntimeError: ") - len("Bearer ") - len(tail))
    scrubbed = _internal_error_message(RuntimeError(prefix + tail + " overflow"))
    assert tail.strip() not in scrubbed


def test_internal_error_scrubber_preserves_bare_ipv6_at_the_input_cap():
    tail = " http://[::1]"
    prefix = "Bearer " + "A" * (_INTERNAL_ERROR_SCRUB_INPUT_CAP - len("RuntimeError: ") - len("Bearer ") - len(tail))
    scrubbed = _internal_error_message(RuntimeError(prefix + tail + " overflow"))
    assert scrubbed.endswith("http://[::1]…")


def test_internal_error_scrubber_handles_non_ascii_digit_port_without_crashing():
    tail = " https://example.com:²"
    prefix = "Bearer " + "A" * (_INTERNAL_ERROR_SCRUB_INPUT_CAP - len("RuntimeError: ") - len("Bearer ") - len(tail))
    assert _internal_error_message(RuntimeError(prefix + tail + " overflow")).endswith("…")


@pytest.mark.parametrize(
    "key",
    [
        "databasePassword",
        "githubToken",
        "sshPrivateKey",
        "stripeSecret",
        "AccountKey",
        "SharedAccessKey",
        "api_keys",
        "secrets",
        "passwords",
        "access_tokens",
        "private_keys",
        "private_key_b64",
        "signing_key_hex",
        "encryption_key_v2",
        "aws_access_key_id_prod",
    ],
)
def test_internal_error_scrubber_masks_extended_credential_key_vocabulary(key):
    scrubbed = _internal_error_message(RuntimeError(f"{key}=LIVE-CREDENTIAL"))
    assert "LIVE-CREDENTIAL" not in scrubbed


@pytest.mark.parametrize("key", ["appMonkey", "apiKeyboardLayout", "max_tokens", "maxTokens"])
def test_internal_error_scrubber_preserves_non_secret_key_like_diagnostics(key):
    message = f"{key}=visible"
    assert message in _internal_error_message(RuntimeError(message))


def test_armored_scrubber_keeps_unicode_offsets_and_masks_the_next_block():
    message = (
        "C:/İlkİz/bundle.pem: -----BEGIN CERTIFICATE-----cert-----END CERTIFICATE-----\n"
        "-----BEGIN PRIVATE KEY-----MIIE-LIVE"
    )
    scrubbed = _internal_error_message(RuntimeError(message))
    assert "MIIE-LIVE" not in scrubbed
    assert scrubbed.count("***") == 2


def test_internal_error_scrubber_masks_triple_quoted_secret():
    scrubbed = _internal_error_message(RuntimeError('password="""hunter2""" request=req-1'))
    assert "hunter2" not in scrubbed
    assert 'password="""***""" request=req-1' in scrubbed


def test_internal_error_scrubber_masks_cookie_assignment_through_semicolons():
    scrubbed = _internal_error_message(RuntimeError("cookie=sid=abc; remember_me=LONGTOKEN"))
    assert "abc" not in scrubbed
    assert "LONGTOKEN" not in scrubbed


def test_internal_error_scrubber_masks_folded_authorization_header():
    scrubbed = _internal_error_message(RuntimeError("Authorization: Basic\r\n dXNlcjpwYXNz\r\nrequest=req-1"))
    assert "dXNlcjpwYXNz" not in scrubbed
    assert "request=req-1" in scrubbed


def test_internal_error_scrubber_handles_long_non_secret_camel_names_without_backtracking():
    name = "ComfyApiNodeExecutionContextManager" + "ProfileSettings" * 40
    scrubbed = _internal_error_message(RuntimeError(f"{name} has no attribute x"))
    assert "ComfyApiNodeExecutionContextManager" in scrubbed


def test_internal_error_scrubber_keeps_a_long_compact_diagnostic():
    scrubbed = _internal_error_message(RuntimeError("x" * 5_000))
    assert len(scrubbed) == 500
    assert scrubbed.startswith("RuntimeError: " + "x" * 100)
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


@pytest.mark.parametrize(
    ("message", "secrets"),
    [
        ("token=ApiKey(\nvalue='sk-LIVE')", ("sk-LIVE",)),
        ("api_key=Outer(Inner('sk-A'), value='sk-LIVE'\n  retrying", ("sk-A", "sk-LIVE", "retrying")),
    ],
)
def test_internal_error_scrubber_masks_the_tail_of_multiline_constructors(message, secrets):
    scrubbed = _internal_error_message(RuntimeError(message))
    assert scrubbed.endswith("=***")
    assert not any(secret in scrubbed for secret in secrets)


def test_internal_error_scrubber_masks_a_multiline_quoted_private_key():
    message = 'private_key="-----BEGIN PRIVATE KEY-----\nMIIEvQ-LIVE\n-----END PRIVATE KEY-----" request=req-1'
    scrubbed = _internal_error_message(RuntimeError(message))
    assert "MIIEvQ-LIVE" not in scrubbed
    assert 'private_key="***" request=req-1' in scrubbed


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


@pytest.mark.parametrize("message", ["token" + "_a" * 60 + " missing", "token" + "_" * 60 + "x"])
def test_internal_error_scrubber_bounds_secret_heads_with_long_tails(message):
    import time

    started = time.perf_counter()
    _internal_error_message(RuntimeError(message))
    assert time.perf_counter() - started < 0.2
