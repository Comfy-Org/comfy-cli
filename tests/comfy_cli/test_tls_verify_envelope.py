"""A TLS trust failure is its own diagnosis, on every surface that can hit it.

`CERTIFICATE_VERIFY_FAILED` used to arrive as a generic transport error, whose
hint — "check the builder URL and your access" — points at the two things that
are demonstrably fine: `curl` to the same host succeeds, and the credential was
never sent. The cause is local, the remedy is a CA bundle, and neither was
anywhere in the envelope.

These assert the rendered ENVELOPE rather than the classifier, because the
defect was that a correct diagnosis existed and did not reach `error.hint`.
"""

from __future__ import annotations

import ssl
import urllib.error

import pytest

import comfy_cli.http as http_mod
from comfy_cli.deploy_api_errors import transport_error


@pytest.fixture(autouse=True)
def _known_trust_store(monkeypatch):
    """Pin the store so the assertions describe the fixture, not the machine."""
    http_mod.trust_store.cache_clear()
    http_mod.ssl_context.cache_clear()
    monkeypatch.delenv(http_mod.CA_FILE_ENV_VAR, raising=False)
    monkeypatch.delenv(http_mod.CA_DIR_ENV_VAR, raising=False)
    yield
    http_mod.trust_store.cache_clear()
    http_mod.ssl_context.cache_clear()


def _verify_failure() -> urllib.error.URLError:
    return urllib.error.URLError(
        ssl.SSLCertVerificationError(
            "[SSL: CERTIFICATE_VERIFY_FAILED] certificate verify failed: unable to get local issuer certificate"
        )
    )


class _Capture:
    """The renderer's error surface, with its registered-hint fallback.

    Reproduced rather than mocked away: the fallback is exactly what buried the
    specific hint under the generic one, so a test that skipped it could not see
    the defect.
    """

    def __init__(self) -> None:
        self.calls: list[dict] = []

    def error(self, *, code, message, hint=None, details=None, **_):
        if not (hint and hint.strip()):
            from comfy_cli import error_codes

            registered = error_codes.get(code)
            hint = registered.hint if registered is not None else None
        self.calls.append({"code": code, "message": message, "hint": hint, "details": details})


def test_the_deploy_surface_reports_the_trust_store_in_the_hint():
    """Given a verify failure, When deploy renders it, Then the hint names the store.

    `DeployAPIError` carried no `hint`, so the specific text could only ride in
    `details` under a key no consumer reads, while `error.hint` showed the
    registry's generic line.
    """
    # Given
    error = transport_error("get", _verify_failure())
    capture = _Capture()

    # When
    capture.error(code=error.code, message=str(error), hint=error.hint, details=error.details)

    # Then
    rendered = capture.calls[0]
    assert rendered["code"] == "tls_verify_failed"
    assert "CERTIFICATE_VERIFY_FAILED" in rendered["message"]
    assert rendered["hint"] == http_mod.tls_trust_hint()
    assert "certifi" in rendered["hint"] or http_mod.CA_FILE_ENV_VAR in rendered["hint"]
    assert rendered["details"] == {"operation": "get"}


def test_the_builder_surface_reports_the_same_hint():
    """Given the same failure, When the builder renders it, Then the two agree.

    One change produced two TLS surfaces; they must not disagree about the
    remedy.
    """
    # Given
    from comfy_cli.command.build import _report_builder_error

    capture = _Capture()

    # When
    _report_builder_error(capture, _verify_failure())

    # Then
    rendered = capture.calls[0]
    assert rendered["code"] == "tls_verify_failed"
    assert rendered["hint"] == http_mod.tls_trust_hint()


def test_an_ordinary_transport_failure_keeps_its_own_code():
    """Given an unrelated URLError, When rendered, Then nothing is relabelled."""
    # Given / When
    error = transport_error("get", urllib.error.URLError("connection refused"))

    # Then
    assert error.code == "deploy_server_error"
    assert error.hint is None


def test_a_bogus_trust_store_is_named_in_the_hint(tmp_path, monkeypatch):
    """Given an unusable SSL_CERT_FILE, When a verify failure renders, Then it says so.

    The fail-closed path: the caller pinned a trust root, it could not be loaded,
    and the hint has to name that rather than suggesting a store the caller
    deliberately overrode.
    """
    # Given
    missing = tmp_path / "nope.pem"
    monkeypatch.setenv(http_mod.CA_FILE_ENV_VAR, str(missing))
    http_mod.trust_store.cache_clear()
    http_mod.ssl_context.cache_clear()

    # When
    error = transport_error("get", _verify_failure())

    # Then
    assert error.code == "tls_verify_failed"
    assert str(missing) in (error.hint or "")
