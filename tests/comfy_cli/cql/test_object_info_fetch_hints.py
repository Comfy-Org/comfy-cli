"""A ``cql_no_graph`` envelope names the cause it actually hit.

``_load_from_target`` has two failure modes that a generic hint actively
misdirects: the 64 MiB fetch cap and the local-mode loopback guard. Both used
to render ``details["hint"]``'s fallback — "pass --input <path>, or start the
server with ``comfy launch``" — which tells someone whose server is already up
to start one. ``comfy-mcp`` issue #252 is that report: every ``nodes`` verb
failed on a 101.9 MiB catalog and the error blamed a server that was running.

The hint-override mechanism already existed and was read at three call sites;
no raise site had ever set it, so the fallback was the only hint the CLI could
emit. These tests drive the real CLI, because the lift-and-strip is in the
command layer and the fallback is the command's own string.
"""

from __future__ import annotations

import io
import json

import pytest
from typer.testing import CliRunner

from comfy_cli import cmdline
from comfy_cli.cql import engine

CAP = engine._MAX_OBJECT_INFO_BYTES


@pytest.fixture(autouse=True)
def _no_ambient_routing(monkeypatch: pytest.MonkeyPatch):
    """``--where`` must be the only routing input (see test_error_envelope_where).

    ``COMFY_OBJECT_INFO_FILE`` is cleared and ``COMFY_LOCAL_URL`` is pinned for
    a sharper reason: ``resilient_load_object_info`` returns the catalog named
    by that variable BEFORE it fetches, so a developer or CI shell that exports
    it hands these tests a success envelope and ``_error_block`` fails on a
    pass. An ambient ``COMFY_LOCAL_URL`` is the same hazard from the other
    side -- a non-loopback value sends the oversize test into the loopback
    branch instead.
    """
    monkeypatch.delenv("COMFY_WHERE", raising=False)
    monkeypatch.delenv("COMFY_OBJECT_INFO_FILE", raising=False)
    monkeypatch.setenv("COMFY_LOCAL_URL", "http://127.0.0.1:8188")


class _Resp(io.BytesIO):
    def __enter__(self) -> _Resp:
        return self

    def __exit__(self, *_exc: object) -> bool:
        return False


def _error_block(argv: list[str]) -> dict:
    result = CliRunner().invoke(cmdline.app, argv)
    lines = [ln for ln in (result.stdout or "").splitlines() if ln.strip()]
    assert lines, f"no envelope on stdout (rc={result.exit_code}, exc={result.exception!r})"
    envelope = json.loads(lines[-1])
    assert envelope.get("ok") is False, f"expected a failure envelope, got {envelope}"
    return envelope["error"]


def test_oversize_catalog_hint_names_the_cap_and_not_comfy_launch(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(engine._opener, "open", lambda req, timeout=None: _Resp(b"x" * (CAP + 1)))

    err = _error_block(["--json", "nodes", "ls", "--where", "local"])

    assert err["code"] == "cql_no_graph"
    assert "64 MiB fetch cap" in err["hint"]
    assert "--input object_info.json" in err["hint"]
    # The whole point of the report: the server was up. The hint may name
    # `comfy launch` only to rule it out, never as the step to take.
    assert "`comfy launch` is not the fix" in err["hint"]
    assert "start the server" not in err["hint"]


def test_non_loopback_local_hint_offers_both_real_routes(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("COMFY_LOCAL_URL", "http://192.0.2.10:8188")

    err = _error_block(["--json", "nodes", "ls", "--where", "local"])

    assert err["code"] == "cql_no_graph"
    assert "--where cloud" in err["hint"]
    assert "--input object_info.json" in err["hint"]
    assert "comfy launch" not in err["hint"]
    assert "start the server" not in err["hint"]


def test_hint_is_not_rendered_twice_in_one_envelope(monkeypatch: pytest.MonkeyPatch):
    """A cause-specific hint reaches ``error.hint`` and never ``error.details``."""
    monkeypatch.setattr(engine._opener, "open", lambda req, timeout=None: _Resp(b"x" * (CAP + 1)))

    err = _error_block(["--json", "nodes", "ls", "--where", "local"])

    assert err["hint"]
    assert "hint" not in (err.get("details") or {})
    # The details a consumer already relied on are untouched.
    assert (err.get("details") or {}).get("url", "").endswith("/object_info")


def test_a_raise_site_that_publishes_details_hint_keeps_publishing_it():
    """``details["hint"]`` is a contract this change does not get to break.

    The HTTPError and URLError sites in ``_load_from_target`` have set
    ``details["hint"]`` ("run ``comfy cloud login``") for as long as they have
    existed, and `nodes`/`validate` render ``details`` into the envelope. Had
    the lift stripped the key, those two envelopes would have lost
    ``error.details.hint`` -- a different error class than the one this PR is
    about, silently changed.
    """
    hint, details = engine.load_error_envelope_fields(
        engine.LoadError(
            "HTTP 401 from http://127.0.0.1:8188/object_info: ",
            details={"url": "http://127.0.0.1:8188/object_info", "status": 401, "hint": "run `comfy cloud login`"},
        ),
        "pass --input <path>, or start the server with `comfy launch`",
    )

    assert hint == "run `comfy cloud login`"
    assert details["hint"] == "run `comfy cloud login`"


def test_the_oversize_hint_is_credentialed_on_cloud_and_never_tells_a_cloud_user_to_launch():
    """The fetch that hit the cap was authenticated; the suggested one must be too.

    An anonymous ``curl`` against Comfy Cloud returns 401, and ``-s`` without
    ``-f`` writes that body to ``object_info.json``, so ``--input`` would load
    a bogus catalog without complaint.
    """
    cloud = engine._oversize_catalog_hint("https://api.comfy.org/api/object_info", is_cloud=True)

    assert "X-API-Key: $COMFY_API_KEY" in cloud
    assert "comfy launch" not in cloud
    assert "-fsS" in cloud and "--max-filesize" in cloud


def test_the_oversize_hint_is_bounded_so_it_does_not_route_around_the_guard():
    """The cap stops resource exhaustion; a hint past it must carry its own bound."""
    local = engine._oversize_catalog_hint("http://127.0.0.1:8188/object_info", is_cloud=False)

    assert "--max-filesize" in local
    assert "curl -fsS" in local
    assert "`comfy launch` is not the fix" in local


@pytest.mark.parametrize(
    ("url", "unwanted"),
    [
        ("http://user:tok@10.0.0.5:8188/object_info", "tok"),
        ("http://127.0.0.1:8188/object_info?token=tok", "tok"),
    ],
)
def test_a_credential_in_a_configured_url_never_reaches_the_hint(url: str, unwanted: str):
    """``workflow._get_graph`` renders ``hint`` and drops ``details`` entirely."""
    assert unwanted not in engine._oversize_catalog_hint(url, is_cloud=False)
    assert unwanted not in engine._hint_url(url)


@pytest.mark.parametrize(
    "url",
    [
        "http://127.0.0.1:8188/object_info;id",
        "http://127.0.0.1:8188/object_info$(id)",
        "http://[::1]:8188/object_info",
    ],
)
def test_a_shell_metacharacter_in_a_configured_url_is_quoted_not_executed(url: str):
    """The hint is copy-paste shell text, so an unquoted URL is a command."""
    quoted = engine._hint_url(url)

    assert quoted.startswith("'") and quoted.endswith("'")
    assert quoted in engine._oversize_catalog_hint(url, is_cloud=False)


def test_a_causeless_load_error_still_gets_the_command_fallback():
    """An error with no hint of its own keeps the old text — this is additive."""
    hint, details = engine.load_error_envelope_fields(
        engine.LoadError("boom", details={"url": "http://127.0.0.1:8188/object_info"}),
        "pass --input <path>, or start the server with `comfy launch`",
    )

    assert hint == "pass --input <path>, or start the server with `comfy launch`"
    assert details == {"url": "http://127.0.0.1:8188/object_info"}
