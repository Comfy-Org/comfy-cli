"""CLI envelope tests for ``comfy feedback`` and ``comfy agent-review``.

These pin the agent-facing contract: JSON mode emits a single ``{sent, ...}``
envelope, a missing inline message in JSON mode is a structured error, and the
underlying ``tracking`` helpers receive exactly what the user typed. The
delivery/consent logic itself lives in test_tracking.py — here we only assert
the command layer wires through correctly.
"""

from __future__ import annotations

import json

import pytest
import typer

from comfy_cli import cmdline, tracking
from comfy_cli.caller import Caller
from comfy_cli.output.renderer import OutputMode, Renderer, set_renderer


def _force_renderer(mode: OutputMode) -> Renderer:
    r = Renderer.resolve(
        is_stdout_tty=False,
        env={},
        caller=Caller(kind="user", agentic=False, source_env=None),
        json_flag=(mode is OutputMode.JSON),
    )
    r.mode = mode
    set_renderer(r)
    return r


def _last_envelope(capsys) -> dict:
    out = capsys.readouterr().out.strip()
    assert out, "expected an envelope on stdout"
    return json.loads(out.splitlines()[-1])


class TestFeedbackJson:
    def test_oneshot_emits_sent_true(self, monkeypatch, capsys):
        monkeypatch.setattr(tracking, "submit_feedback", lambda *a, **k: True)
        _force_renderer(OutputMode.JSON)
        cmdline.feedback(message="run is great")
        env = _last_envelope(capsys)
        assert env["ok"] is True
        assert env["command"] == "feedback"
        assert env["data"] == {"sent": True, "message": "run is great"}

    def test_oneshot_relays_message_verbatim(self, monkeypatch, capsys):
        captured: dict = {}
        monkeypatch.setattr(tracking, "submit_feedback", lambda msg, **k: captured.update(msg=msg) or True)
        _force_renderer(OutputMode.JSON)
        cmdline.feedback(message="jobs watch needs an ETA")
        assert captured["msg"] == "jobs watch needs an ETA"

    def test_oneshot_sent_false_when_opted_out(self, monkeypatch, capsys):
        monkeypatch.setattr(tracking, "submit_feedback", lambda *a, **k: False)
        _force_renderer(OutputMode.JSON)
        cmdline.feedback(message="anything")
        env = _last_envelope(capsys)
        assert env["data"]["sent"] is False

    def test_missing_message_in_json_is_structured_error(self, monkeypatch, capsys):
        monkeypatch.setattr(tracking, "submit_feedback", _fail_if_called("submit_feedback"))
        _force_renderer(OutputMode.JSON)
        with pytest.raises(typer.Exit):
            cmdline.feedback(message=None)
        env = _last_envelope(capsys)
        assert env["ok"] is False
        assert env["error"]["code"] == "feedback_message_required"


class TestFeedbackPretty:
    def test_oneshot_pretty_thanks(self, monkeypatch, capsys):
        monkeypatch.setattr(tracking, "submit_feedback", lambda *a, **k: True)
        _force_renderer(OutputMode.PRETTY)
        cmdline.feedback(message="great tool")
        captured = capsys.readouterr()
        assert "Thank you" in (captured.out + captured.err)

    def test_oneshot_pretty_disabled_notice(self, monkeypatch, capsys):
        monkeypatch.setattr(tracking, "submit_feedback", lambda *a, **k: False)
        _force_renderer(OutputMode.PRETTY)
        cmdline.feedback(message="great tool")
        captured = capsys.readouterr()
        assert "not sent" in (captured.out + captured.err).lower()


class TestAgentReviewJson:
    def test_emits_sent_true(self, monkeypatch, capsys):
        monkeypatch.setattr(tracking, "submit_agent_review", lambda *a, **k: True)
        _force_renderer(OutputMode.JSON)
        cmdline.agent_review(summary="user shipped a clip after one retry")
        env = _last_envelope(capsys)
        assert env["ok"] is True
        assert env["command"] == "agent-review"
        assert env["data"] == {"sent": True, "summary": "user shipped a clip after one retry"}

    def test_relays_summary_verbatim(self, monkeypatch, capsys):
        captured: dict = {}
        monkeypatch.setattr(tracking, "submit_agent_review", lambda s, **k: captured.update(s=s) or True)
        _force_renderer(OutputMode.JSON)
        cmdline.agent_review(summary="hit a missing-model error then succeeded")
        assert captured["s"] == "hit a missing-model error then succeeded"

    def test_sent_false_when_consent_gated_off(self, monkeypatch, capsys):
        monkeypatch.setattr(tracking, "submit_agent_review", lambda *a, **k: False)
        _force_renderer(OutputMode.JSON)
        cmdline.agent_review(summary="anything")
        env = _last_envelope(capsys)
        assert env["data"]["sent"] is False


def _fail_if_called(name: str):
    def _boom(*_a, **_k):
        raise AssertionError(f"{name} must not be called on the error path")

    return _boom
