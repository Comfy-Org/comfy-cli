"""`comfy run --client-id` — submitting as an already-connected client.

ComfyUI addresses a prompt's execution events to the socket whose clientId
submitted it. Borrowing a live client's id is how an out-of-band submitter
(the in-app agent) gets that client's canvas to light up. The same mechanism
is a footgun in reverse: ComfyUI's `/ws` handler pops any socket already
registered under an incoming clientId, so anything here that reconnects as a
borrowed id would silence the client the flag exists to feed. These tests pin
both halves.
"""

from __future__ import annotations

import json
import os
import tempfile
import uuid
from unittest.mock import MagicMock, patch

import pytest
import typer

from comfy_cli import jobs_state
from comfy_cli.cmdline import run as run_command
from comfy_cli.command import jobs
from comfy_cli.command.jobs import _client_id_withheld, _resolve_watch_client_id
from comfy_cli.command.run import WorkflowExecution, execute
from comfy_cli.output import Renderer, set_renderer
from comfy_cli.output.renderer import OutputMode, reset_renderer_for_testing

BORROWED = "32ef0f50a7dc41dc8ab9e0107eedb88e"


@pytest.fixture(autouse=True)
def ndjson_renderer():
    renderer = Renderer(mode=OutputMode.NDJSON, command="run")
    set_renderer(renderer)
    yield renderer
    reset_renderer_for_testing()


@pytest.fixture
def simple_workflow():
    return {
        "1": {
            "class_type": "EmptyLatentImage",
            "inputs": {"width": 64, "height": 64, "batch_size": 1},
        },
        "2": {
            "class_type": "SaveImage",
            "inputs": {"filename_prefix": "x", "images": ["1", 0]},
        },
    }


@pytest.fixture
def workflow_file(simple_workflow):
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(simple_workflow, f)
        path = f.name
    yield path
    os.unlink(path)


def _envelope(out: str) -> dict:
    lines = [json.loads(ln) for ln in out.splitlines() if ln.strip()]
    assert lines, "expected at least one NDJSON line"
    assert lines[-1]["type"] == "envelope", lines[-1]
    return lines[-1]


class TestWorkflowExecutionClientId:
    def test_borrowed_id_is_used_verbatim(self, simple_workflow):
        ex = WorkflowExecution(simple_workflow, "127.0.0.1", 8188, False, None, False, 30, client_id=BORROWED)
        assert ex.client_id == BORROWED
        assert ex.borrowed_client_id is True

    def test_absent_id_is_minted(self, simple_workflow):
        ex = WorkflowExecution(simple_workflow, "127.0.0.1", 8188, False, None, False, 30)
        assert uuid.UUID(ex.client_id)
        assert ex.borrowed_client_id is False

    def test_borrowed_id_rides_the_prompt_post(self, simple_workflow):
        ex = WorkflowExecution(simple_workflow, "127.0.0.1", 8188, False, None, False, 30, client_id=BORROWED)
        with patch("comfy_cli.command.run.execution.no_redirect_urlopen") as mock_open:
            mock_open.return_value.__enter__.return_value.read.return_value = json.dumps({"prompt_id": "p"}).encode()
            ex.queue()
        body = json.loads(mock_open.call_args[0][0].data.decode())
        assert body["client_id"] == BORROWED

    def test_connect_refuses_to_evict_the_borrowed_client(self, simple_workflow):
        ex = WorkflowExecution(simple_workflow, "127.0.0.1", 8188, False, None, False, 30, client_id=BORROWED)
        with pytest.raises(RuntimeError, match="borrowed clientId"):
            ex.connect()


class TestExecuteRecordsBorrowedId:
    def test_async_run_persists_the_borrowed_flag(self, workflow_file, capsys, tmp_path):
        written: list[jobs_state.JobState] = []

        def capture(state):
            written.append(state)
            return tmp_path / f"{state.prompt_id}.json"

        with (
            patch("comfy_cli.command.run.check_comfy_server_running", return_value=True),
            patch("comfy_cli.command.run._fetch_object_info", return_value={}),
            patch("comfy_cli.command.run._preflight_validate"),
            patch("comfy_cli.command.run._journal_run"),
            patch("comfy_cli.command.run._spawn_watcher", return_value=True),
            patch("comfy_cli.command.run._tail_state_file"),
            patch("comfy_cli.jobs_state.write", side_effect=capture),
            patch("comfy_cli.http._AUTHED_OPENER.open") as mock_open,
        ):
            mock_open.return_value.__enter__.return_value.read.return_value = json.dumps({"prompt_id": "p"}).encode()
            execute(
                workflow_file,
                "127.0.0.1",
                8188,
                wait=False,
                timeout=30,
                client_id=BORROWED,
            )

        assert written, "expected a job state file write"
        state = written[0]
        assert state.client_id == BORROWED
        assert state.client_id_borrowed is True
        assert _envelope(capsys.readouterr().out)["data"]["client_id"] == BORROWED


class TestCliRejections:
    @pytest.mark.parametrize(
        ("kwargs", "reason"),
        [
            ({"wait": True, "where": "local"}, "wait"),
            ({"wait": False, "where": "cloud"}, "cloud"),
        ],
    )
    def test_conflicting_flags_are_refused(self, workflow_file, capsys, kwargs, reason):
        with (
            patch("comfy_cli.cmdline.tracking", MagicMock()),
            pytest.raises(typer.Exit) as exc,
        ):
            run_command(workflow=workflow_file, client_id=BORROWED, **kwargs)
        assert exc.value.exit_code == 1
        env = _envelope(capsys.readouterr().out)
        assert env["ok"] is False
        assert env["error"]["code"] == "client_id_rejected"
        assert env["error"]["details"]["reason"] == reason


class TestTelemetryDoesNotLeakTheSocketID:
    # A clientId is capability-bearing in ComfyUI's model: knowing it is enough
    # to receive another client's execution events and to evict its socket.
    def test_client_id_is_redacted_but_still_counted(self):
        from comfy_cli import tracking

        props = tracking.filter_command_kwargs({"client_id": BORROWED, "wait": False})
        assert BORROWED not in str(props)
        assert "client_id" in props, "the key must survive so flag usage stays measurable"


class TestBorrowedMarkerRidesThePrompt:
    # The state file is NOT a shared channel: the in-app agent runs the CLI
    # under a sandboxed HOME it deletes at turn end, so the user's own
    # `jobs watch` can never see the state file of the very runs that borrow an
    # id. The marker has to travel on the prompt or the guard is inert for
    # every run it exists to protect.
    @pytest.mark.parametrize("borrowed", [True, False])
    def test_marker_is_submitted_with_the_prompt(self, simple_workflow, borrowed):
        ex = WorkflowExecution(
            simple_workflow,
            "127.0.0.1",
            8188,
            False,
            None,
            False,
            30,
            client_id=BORROWED if borrowed else None,
        )
        with patch("comfy_cli.command.run.execution.no_redirect_urlopen") as mock_open:
            mock_open.return_value.__enter__.return_value.read.return_value = json.dumps({"prompt_id": "p"}).encode()
            ex.queue()
        extra = json.loads(mock_open.call_args[0][0].data.decode())["extra_data"]
        assert extra.get(jobs_state.BORROWED_CLIENT_ID_KEY) is (True if borrowed else None)


class TestWatchDoesNotStealTheSocket:
    @pytest.fixture(autouse=True)
    def no_server(self, monkeypatch):
        """Default: the server knows nothing. Tests that care override it."""
        monkeypatch.setattr(jobs, "_submitted_extra_data", lambda *_: None)

    def test_borrowed_record_resolves_to_no_client_id(self, monkeypatch):
        state = jobs_state.new(
            prompt_id="p",
            client_id=BORROWED,
            workflow="w.json",
            where="local",
            host="127.0.0.1",
            port=8188,
            client_id_borrowed=True,
        )
        monkeypatch.setattr(jobs_state, "read", lambda _: state)
        assert _resolve_watch_client_id("127.0.0.1", 8188, "p") is None

    def test_minted_record_still_resolves(self, monkeypatch):
        state = jobs_state.new(
            prompt_id="p",
            client_id="cli-minted",
            workflow="w.json",
            where="local",
            host="127.0.0.1",
            port=8188,
        )
        monkeypatch.setattr(jobs_state, "read", lambda _: state)
        assert _resolve_watch_client_id("127.0.0.1", 8188, "p") == "cli-minted"

    # Both reasons `_resolve_watch_client_id` answers None want OPPOSITE advice:
    # an unresolvable id is worth passing --client-id for, a withheld one must
    # never be, because doing so performs the eviction the guard exists to stop.
    @pytest.mark.parametrize("borrowed", [True, False])
    def test_withheld_is_distinguishable_from_unresolvable(self, monkeypatch, borrowed):
        state = jobs_state.new(
            prompt_id="p",
            client_id=BORROWED,
            workflow="w.json",
            where="local",
            client_id_borrowed=borrowed,
        )
        monkeypatch.setattr(jobs_state, "read", lambda _: state)
        assert _client_id_withheld("127.0.0.1", 8188, "p") is borrowed

    def test_no_state_file_is_not_withheld(self, monkeypatch):
        monkeypatch.setattr(jobs_state, "read", lambda _: None)
        assert _client_id_withheld("127.0.0.1", 8188, "p") is False

    # The case the state-file-only guard missed entirely: a watcher that cannot
    # see the submitting run's state still has to refuse the id.
    def test_server_marker_is_honoured_without_a_state_file(self, monkeypatch):
        monkeypatch.setattr(jobs_state, "read", lambda _: None)
        monkeypatch.setattr(
            jobs,
            "_submitted_extra_data",
            lambda *_: {"client_id": BORROWED, jobs_state.BORROWED_CLIENT_ID_KEY: True},
        )
        assert _resolve_watch_client_id("127.0.0.1", 8188, "p") is None
        assert _client_id_withheld("127.0.0.1", 8188, "p") is True

    def test_an_unmarked_server_record_still_resolves(self, monkeypatch):
        monkeypatch.setattr(jobs_state, "read", lambda _: None)
        monkeypatch.setattr(jobs, "_submitted_extra_data", lambda *_: {"client_id": "cli-minted"})
        assert _resolve_watch_client_id("127.0.0.1", 8188, "p") == "cli-minted"
        assert _client_id_withheld("127.0.0.1", 8188, "p") is False
