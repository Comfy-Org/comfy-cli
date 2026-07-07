"""Unit tests for the shared loopback-host SSRF guard helper."""

from __future__ import annotations

import pytest

from comfy_cli.cql._net import is_loopback_host


@pytest.mark.parametrize(
    "host,expected",
    [
        ("localhost", True),
        ("LOCALHOST", True),
        ("127.0.0.1", True),
        ("127.0.0.2", True),
        ("127.5.5.5", True),
        ("::1", True),
        ("0.0.0.0", False),
        ("10.0.0.5", False),
        ("example.com", False),
        ("", False),
        ("not a host", False),
    ],
)
def test_is_loopback_host(host, expected):
    assert is_loopback_host(host) is expected
