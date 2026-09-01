"""Usage errors are refusals with an envelope, not a hole in the stream.

Click validates argv BEFORE any command body runs, so an unknown option, command
or missing option value never reached a `renderer.error(...)` call site: click
printed its own Rich panel to stderr and exited 2 with zero bytes on stdout,
which a `--json` consumer cannot tell from a transport failure.

These drive the real CLI, because the fix lives at the click entrypoint
(`_RootGroup.make_context` / `.invoke`) rather than in any command body.
"""

from __future__ import annotations

import json
import sys

import pytest
from typer.testing import CliRunner

from comfy_cli.cmdline import app


@pytest.mark.parametrize(
    ("argv", "message_fragment", "detail_command"),
    [
        pytest.param(
            ["--json", "build", "release", "show", ".", "--release", "abc"],
            "No such option: --release",
            "comfy build release show",
            id="unknown-option-on-a-nested-command",
        ),
        pytest.param(
            ["--json", "nosuchgroup"],
            "No such command 'nosuchgroup'",
            "comfy",
            id="unknown-command",
        ),
    ],
)
def test_a_usage_error_ends_the_json_stream_with_an_envelope(argv, message_fragment, detail_command):
    """Given a bad invocation, When --json is on, Then stdout still ends in an envelope."""
    # Given / When
    result = CliRunner().invoke(app, argv, env={"NO_COLOR": "1", "COLUMNS": "400"})

    # Then
    assert result.exit_code == 2, "a usage error is exit 2, not 1"
    envelope = json.loads([line for line in result.stdout.splitlines() if line.strip()][-1])
    assert envelope["ok"] is False
    assert envelope["error"]["code"] == "usage_error"
    assert message_fragment in envelope["error"]["message"]
    assert envelope["error"]["details"]["command"] == detail_command
    assert envelope["error"]["details"]["exit_code"] == 2
    assert "--help" in envelope["error"]["hint"]


def test_a_near_miss_option_carries_clicks_suggestion():
    """Given a typo, When it is a near miss, Then `did_you_mean` names the real option."""
    # Given / When
    result = CliRunner().invoke(
        app, ["--json", "build", "status", "--models-dirr", "/tmp"], env={"NO_COLOR": "1", "COLUMNS": "400"}
    )

    # Then
    envelope = json.loads([line for line in result.stdout.splitlines() if line.strip()][-1])
    assert envelope["error"]["details"]["option"] == "--models-dirr"
    assert "--models-dir" in envelope["error"]["details"]["did_you_mean"]


def test_a_usage_error_stays_a_single_rich_panel_in_pretty_mode():
    """Given pretty mode, When a usage error happens, Then no envelope doubles it."""
    # Given / When
    result = CliRunner().invoke(
        app, ["--no-json", "nosuchgroup"], env={"NO_COLOR": "1", "COLUMNS": "400"}
    )

    # Then
    assert result.exit_code == 2
    assert result.stdout.strip() == ""
    assert "No such command" in result.stderr


def test_an_unknown_root_option_is_answered_about_this_invocation(monkeypatch):
    """Given --json and a bad ROOT option, When invoked in-process, Then an envelope.

    A root parse failure happens before click binds a single parameter, so
    `ctx.params` is empty. Recovering the flags from `sys.argv` answers with the
    HOST process's command line — under an embedding caller, or a test runner,
    that is somebody else's arguments entirely, and an ambient `COMFY_OUTPUT`
    then decides instead of the `--json` that was actually passed.
    """
    # Given: a host command line that says nothing about JSON, and an ambient
    # setting that would otherwise win.
    monkeypatch.setattr(sys, "argv", ["pytest", "--some-unrelated-runner-flag"])

    # When
    result = CliRunner().invoke(
        app, ["--json", "--bogusroot"], env={"NO_COLOR": "1", "COLUMNS": "400", "COMFY_OUTPUT": "pretty"}
    )

    # Then
    assert result.exit_code == 2
    envelope = json.loads([line for line in result.stdout.splitlines() if line.strip()][-1])
    assert envelope["error"]["code"] == "usage_error"
    assert "--bogusroot" in envelope["error"]["message"]
