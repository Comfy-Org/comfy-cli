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
from typer.testing import CliRunner

from comfy_cli import jobs_state
from comfy_cli.cmdline import run as run_command
from comfy_cli.command import jobs
from comfy_cli.command.jobs import _ClientIdRejected, _select_watch_client_id, _submitted_extra_data
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
            patch("comfy_cli.command.run.WebSocket") as mock_ws,
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
        body = json.loads(mock_open.call_args[0][0].data.decode())
        assert body["client_id"] == BORROWED
        assert body["extra_data"][jobs_state.BORROWED_CLIENT_ID_KEY] is True
        mock_ws.assert_not_called()
        assert _envelope(capsys.readouterr().out)["data"]["client_id"] == BORROWED


class TestCliRejections:
    @pytest.mark.parametrize(
        ("kwargs", "reason"),
        [
            ({"wait": True, "where": "local"}, "wait"),
            ({"wait": False, "where": "cloud"}, "cloud"),
            ({"wait": False, "where": "local", "client_id": "   "}, "empty"),
        ],
    )
    def test_conflicting_flags_are_refused(self, workflow_file, capsys, kwargs, reason):
        with (
            patch("comfy_cli.cmdline.tracking", MagicMock()),
            pytest.raises(typer.Exit) as exc,
        ):
            run_command(workflow=workflow_file, client_id=kwargs.pop("client_id", BORROWED), **kwargs)
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


def _marked(**extra):
    """A server record for a prompt submitted by `comfy run --client-id`."""
    return {"client_id": BORROWED, jobs_state.BORROWED_CLIENT_ID_KEY: True, **extra}


class TestWatchDoesNotStealTheSocket:
    @pytest.fixture(autouse=True)
    def answered_empty_server(self, monkeypatch):
        """Default: the server ANSWERED and holds no record of this prompt.

        Conclusive, which is a different thing from a read that failed — the
        class below pins that difference.
        """
        monkeypatch.setattr(jobs, "_submitted_extra_data", lambda *_: (None, True))

    def test_borrowed_record_yields_no_client_id(self, monkeypatch):
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
        assert _select_watch_client_id("127.0.0.1", 8188, "p", None) == (None, "borrowed")

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
        assert _select_watch_client_id("127.0.0.1", 8188, "p", None) == ("cli-minted", None)

    # The two reasons a watch ends up with no id want OPPOSITE advice: an
    # unresolvable id is worth passing --client-id for, a borrowed one must
    # never be, because doing so performs the eviction the guard exists to stop.
    @pytest.mark.parametrize("borrowed", [True, False])
    def test_borrowed_is_distinguishable_from_unresolvable(self, monkeypatch, borrowed):
        state = jobs_state.new(
            prompt_id="p",
            client_id=BORROWED,
            workflow="w.json",
            where="local",
            client_id_borrowed=borrowed,
        )
        monkeypatch.setattr(jobs_state, "read", lambda _: state)
        _cid, reason = _select_watch_client_id("127.0.0.1", 8188, "p", None)
        assert (reason == "borrowed") is borrowed

    def test_an_answered_empty_server_is_attachable(self, monkeypatch):
        monkeypatch.setattr(jobs_state, "read", lambda _: None)
        assert _select_watch_client_id("127.0.0.1", 8188, "p", None) == (None, None)

    # The case the state-file-only guard missed entirely: a watcher that cannot
    # see the submitting run's state still has to refuse the id.
    def test_server_marker_is_honoured_without_a_state_file(self, monkeypatch):
        monkeypatch.setattr(jobs_state, "read", lambda _: None)
        monkeypatch.setattr(jobs, "_submitted_extra_data", lambda *_: (_marked(), True))
        assert _select_watch_client_id("127.0.0.1", 8188, "p", None) == (None, "borrowed")

    def test_explicit_override_cannot_bypass_borrowed_marker(self, monkeypatch):
        monkeypatch.setattr(jobs_state, "read", lambda _: None)
        monkeypatch.setattr(jobs, "_submitted_extra_data", lambda *_: (_marked(), True))
        with pytest.raises(_ClientIdRejected) as excinfo:
            _select_watch_client_id("127.0.0.1", 8188, "p", BORROWED)
        assert excinfo.value.reason == "borrowed"

    def test_selection_reads_server_marker_once(self, monkeypatch):
        monkeypatch.setattr(jobs_state, "read", lambda _: None)
        reads = 0

        def server_extra(*_args):
            nonlocal reads
            reads += 1
            if reads > 1:
                raise AssertionError("selection split marker and client-id reads")
            return _marked(), True

        monkeypatch.setattr(jobs, "_submitted_extra_data", server_extra)
        with pytest.raises(_ClientIdRejected):
            _select_watch_client_id("127.0.0.1", 8188, "p", BORROWED)
        assert reads == 1

    def test_watch_rejects_explicit_override_before_opening_a_socket(self, monkeypatch):
        monkeypatch.setattr(jobs, "_server_or_error", lambda *_args, **_kwargs: True)
        monkeypatch.setattr(
            jobs,
            "_snapshot",
            lambda *_: {"prompt_id": "p", "status": "running", "outputs": []},
        )
        monkeypatch.setattr(jobs_state, "read", lambda _: None)
        monkeypatch.setattr(jobs, "_submitted_extra_data", lambda *_: (_marked(), True))

        def fail_if_constructed(*_args, **_kwargs):
            raise AssertionError("borrowed watch constructed a WebSocket")

        monkeypatch.setattr(jobs, "WebSocket", fail_if_constructed)
        result = CliRunner().invoke(
            jobs.app,
            ["watch", "p", "--where", "local", "--client-id", BORROWED],
        )

        assert result.exit_code == 1, result.output
        env = _envelope(result.output)
        assert env["error"]["code"] == "client_id_rejected"
        assert env["error"]["details"]["reason"] == "borrowed"

    def test_watch_polls_borrowed_run_without_opening_a_socket(self, monkeypatch):
        snapshots = iter(
            [
                {"prompt_id": "p", "status": "running", "outputs": []},
                {"prompt_id": "p", "status": "completed", "outputs": ["result.png"]},
            ]
        )
        monkeypatch.setattr(jobs, "_server_or_error", lambda *_args, **_kwargs: True)
        monkeypatch.setattr(jobs, "_snapshot", lambda *_: next(snapshots))
        monkeypatch.setattr(jobs, "_history_completed_nodes", lambda *_: {"1"})
        monkeypatch.setattr(jobs, "_POLL_ONLY_WATCH_POLL_S", 0)
        monkeypatch.setattr(jobs_state, "read", lambda _: None)
        monkeypatch.setattr(jobs, "_submitted_extra_data", lambda *_: (_marked(), True))

        def fail_if_constructed(*_args, **_kwargs):
            raise AssertionError("borrowed watch constructed a WebSocket")

        monkeypatch.setattr(jobs, "WebSocket", fail_if_constructed)
        result = CliRunner().invoke(jobs.app, ["watch", "p", "--where", "local"])

        assert result.exit_code == 0, result.output
        env = _envelope(result.output)
        assert env["data"]["status"] == "completed"
        assert env["data"]["outputs"] == ["result.png"]
        assert env["data"]["completed_nodes"] == ["1"]
        assert env["data"]["client_id"] is None
        assert env["data"]["attached"] is False

    def test_borrowed_poll_preserves_a_failed_runs_error(self, monkeypatch):
        snapshots = iter(
            [
                {"prompt_id": "p", "status": "running", "outputs": []},
                {
                    "prompt_id": "p",
                    "status": "error",
                    "outputs": [],
                    "error": {"exception_message": "node 5 exploded", "node_id": "5"},
                },
            ]
        )
        monkeypatch.setattr(jobs, "_server_or_error", lambda *_args, **_kwargs: True)
        monkeypatch.setattr(jobs, "_snapshot", lambda *_: next(snapshots))
        monkeypatch.setattr(jobs, "_history_completed_nodes", lambda *_: set())
        monkeypatch.setattr(jobs, "_POLL_ONLY_WATCH_POLL_S", 0)
        monkeypatch.setattr(jobs_state, "read", lambda _: None)
        monkeypatch.setattr(jobs, "_submitted_extra_data", lambda *_: (_marked(), True))
        monkeypatch.setattr(
            jobs,
            "WebSocket",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("borrowed watch constructed a WebSocket")),
        )

        result = CliRunner().invoke(jobs.app, ["watch", "p", "--where", "local"])

        assert result.exit_code == 1, result.output
        env = _envelope(result.output)
        assert env["error"]["code"] == "execution_error"
        assert env["error"]["message"] == "node 5 exploded"
        assert "error" not in env["error"]["details"]

    def test_an_unmarked_server_record_still_resolves(self, monkeypatch):
        monkeypatch.setattr(jobs_state, "read", lambda _: None)
        monkeypatch.setattr(jobs, "_submitted_extra_data", lambda *_: ({"client_id": "cli-minted"}, True))
        assert _select_watch_client_id("127.0.0.1", 8188, "p", None) == ("cli-minted", None)


class TestUnverifiableOwnershipFailsSafe:
    """A failed read is not an unmarked record.

    The guard above reads the borrowed marker from two places, and BOTH can be
    unavailable at once: the state file is routinely invisible to the watcher
    (the in-app agent submits under a sandboxed HOME it deletes at turn end),
    and `/queue` + `/history` can blip. Treating that as "no marker" let an
    explicit --client-id attach to a run that may well be feeding a live tab —
    the exact eviction the marker exists to prevent. Absence of the marker is
    evidence only once somewhere it could have been recorded actually answered.
    """

    @pytest.fixture(autouse=True)
    def unverifiable(self, monkeypatch):
        monkeypatch.setattr(jobs_state, "read", lambda _: None)
        monkeypatch.setattr(jobs, "_submitted_extra_data", lambda *_: (None, False))

    def test_explicit_override_is_refused_rather_than_attached(self):
        with pytest.raises(_ClientIdRejected) as excinfo:
            _select_watch_client_id("127.0.0.1", 8188, "p", BORROWED)
        # NOT "borrowed": we never established that, and the advice differs —
        # a borrowed run says "drop --client-id", this one says "retry".
        assert excinfo.value.reason == "indeterminate"

    def test_no_override_polls_rather_than_attaching(self):
        assert _select_watch_client_id("127.0.0.1", 8188, "p", None) == (None, "indeterminate")

    def test_a_readable_state_file_still_settles_it(self, monkeypatch):
        """`comfy run` wrote that file and recorded that it did not borrow, so a
        server that cannot be reached does not make the run unverifiable."""
        state = jobs_state.new(prompt_id="p", client_id="cli-minted", workflow="w.json", where="local")
        monkeypatch.setattr(jobs_state, "read", lambda _: state)
        assert _select_watch_client_id("127.0.0.1", 8188, "p", None) == ("cli-minted", None)

    def test_watch_refuses_the_override_before_opening_a_socket(self, monkeypatch):
        monkeypatch.setattr(jobs, "_server_or_error", lambda *_args, **_kwargs: True)
        monkeypatch.setattr(jobs, "_snapshot", lambda *_: {"prompt_id": "p", "status": "running", "outputs": []})

        def fail_if_constructed(*_args, **_kwargs):
            raise AssertionError("unverifiable watch constructed a WebSocket")

        monkeypatch.setattr(jobs, "WebSocket", fail_if_constructed)
        result = CliRunner().invoke(jobs.app, ["watch", "p", "--where", "local", "--client-id", BORROWED])

        assert result.exit_code == 1, result.output
        env = _envelope(result.output)
        assert env["error"]["code"] == "client_id_rejected"
        assert env["error"]["details"]["reason"] == "indeterminate"
        assert "retry" in env["error"]["hint"]

    def test_watch_polls_without_opening_a_socket(self, monkeypatch):
        snapshots = iter(
            [
                {"prompt_id": "p", "status": "running", "outputs": []},
                {"prompt_id": "p", "status": "completed", "outputs": ["result.png"]},
            ]
        )
        monkeypatch.setattr(jobs, "_server_or_error", lambda *_args, **_kwargs: True)
        monkeypatch.setattr(jobs, "_snapshot", lambda *_: next(snapshots))
        monkeypatch.setattr(jobs, "_history_completed_nodes", lambda *_: {"1"})
        monkeypatch.setattr(jobs, "_POLL_ONLY_WATCH_POLL_S", 0)

        def fail_if_constructed(*_args, **_kwargs):
            raise AssertionError("unverifiable watch constructed a WebSocket")

        monkeypatch.setattr(jobs, "WebSocket", fail_if_constructed)
        result = CliRunner().invoke(jobs.app, ["watch", "p", "--where", "local"])

        assert result.exit_code == 0, result.output
        env = _envelope(result.output)
        assert env["data"]["status"] == "completed"
        assert env["data"]["client_id"] is None
        assert env["data"]["attached"] is False


class TestSubmittedExtraDataReportsConclusiveness:
    """`(extra_data, conclusive)` — the second element is the safety-critical one.

    A positive hit in either store IS the submitted record, so one endpoint is
    enough. A negative needs BOTH to have answered: a prompt absent from a
    `/history` we could read may still be live in a `/queue` we could not.
    """

    @staticmethod
    def _server(monkeypatch, *, queue, history):
        def fake_get(url, **_kw):
            answer = queue if url.endswith("/queue") else history
            if isinstance(answer, Exception):
                raise answer
            return answer

        monkeypatch.setattr(jobs, "_http_get_json", fake_get)

    BOOM = RuntimeError("failed to GET: connection refused")
    EMPTY_QUEUE = {"queue_running": [], "queue_pending": []}
    QUEUED = {"queue_running": [[0, "p", {}, {"client_id": "cid"}, {}]], "queue_pending": []}
    IN_HISTORY = {"p": {"prompt": [0, "p", {}, {"client_id": "cid"}, {}]}}

    def test_a_queue_hit_is_conclusive_even_if_history_is_unreachable(self, monkeypatch):
        self._server(monkeypatch, queue=self.QUEUED, history=self.BOOM)
        assert _submitted_extra_data("127.0.0.1", 8188, "p") == ({"client_id": "cid"}, True)

    def test_a_history_hit_is_conclusive_even_if_the_queue_was_unreachable(self, monkeypatch):
        self._server(monkeypatch, queue=self.BOOM, history=self.IN_HISTORY)
        assert _submitted_extra_data("127.0.0.1", 8188, "p") == ({"client_id": "cid"}, True)

    def test_both_answering_empty_is_a_conclusive_absence(self, monkeypatch):
        self._server(monkeypatch, queue=self.EMPTY_QUEUE, history={})
        assert _submitted_extra_data("127.0.0.1", 8188, "p") == (None, True)

    def test_an_unreadable_history_is_not_an_absence(self, monkeypatch):
        self._server(monkeypatch, queue=self.EMPTY_QUEUE, history=self.BOOM)
        assert _submitted_extra_data("127.0.0.1", 8188, "p") == (None, False)

    def test_an_unreadable_queue_is_not_an_absence(self, monkeypatch):
        """The prompt may be running right now in the queue we could not read."""
        self._server(monkeypatch, queue=self.BOOM, history={})
        assert _submitted_extra_data("127.0.0.1", 8188, "p") == (None, False)

    def test_neither_answering_is_not_an_absence(self, monkeypatch):
        self._server(monkeypatch, queue=self.BOOM, history=self.BOOM)
        assert _submitted_extra_data("127.0.0.1", 8188, "p") == (None, False)

    def test_a_wrongly_shaped_body_is_not_an_answer(self, monkeypatch):
        """A 200 carrying valid JSON of the wrong shape (a proxy error page, a
        captive portal) cannot be read as the server reporting an empty queue."""
        self._server(monkeypatch, queue=["not", "a", "dict"], history={})
        assert _submitted_extra_data("127.0.0.1", 8188, "p") == (None, False)

    def test_an_unwalkable_queue_section_is_not_an_answer(self, monkeypatch):
        self._server(monkeypatch, queue={"queue_running": "nonsense"}, history={})
        assert _submitted_extra_data("127.0.0.1", 8188, "p") == (None, False)

    def test_a_matched_entry_with_no_extra_data_slot_is_not_an_answer(self, monkeypatch):
        """Found the prompt but cannot read where the marker would live."""
        self._server(monkeypatch, queue={"queue_running": [[0, "p"]]}, history={})
        assert _submitted_extra_data("127.0.0.1", 8188, "p") == (None, False)

    def test_a_marked_record_round_trips(self, monkeypatch):
        self._server(monkeypatch, queue={"queue_running": [[0, "p", {}, _marked(), {}]]}, history={})
        extra, conclusive = _submitted_extra_data("127.0.0.1", 8188, "p")
        assert conclusive is True
        assert extra[jobs_state.BORROWED_CLIENT_ID_KEY] is True
