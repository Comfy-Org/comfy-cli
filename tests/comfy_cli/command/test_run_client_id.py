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
import pathlib
import tempfile
import uuid
from unittest.mock import MagicMock, patch

import pytest
import typer
from typer.testing import CliRunner

from comfy_cli import jobs_state
from comfy_cli.cmdline import run as run_command
from comfy_cli.command import jobs
from comfy_cli.command.jobs import (
    _ClientIdRejected,
    _select_watch_client_id,
    _submitted_extra_data,
    _SubmittedRecord,
)
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


def _validate_watch_envelope(data):
    """Check a watch envelope against the published schema.

    The `poll_reason` enum is only worth publishing if something exercises it
    with a non-null value.
    """
    import jsonschema

    schema_path = pathlib.Path(jobs.__file__).parent.parent / "schemas" / "jobs.json"
    jsonschema.Draft202012Validator(json.loads(schema_path.read_text())).validate(data)


def _record(extra=None, *, read=True, borrowed_ids=(), queue_read=True):
    """A `_submitted_extra_data` result, for stubbing it out."""
    return _SubmittedRecord(extra, read, frozenset(borrowed_ids), queue_read)


class TestWatchDoesNotStealTheSocket:
    @pytest.fixture(autouse=True)
    def record_read_but_empty(self, monkeypatch):
        """Default: this prompt's record WAS read, and carries no marker.

        Note `read=True` with `extra=None` is the shape for a record that was
        found and is unmarked — NOT for a server that holds no record at all,
        which reports `read=False` and is pinned in the classes below.
        """
        monkeypatch.setattr(jobs, "_submitted_extra_data", lambda *_: _record({}))

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

    def test_a_read_unmarked_record_is_attachable(self, monkeypatch):
        monkeypatch.setattr(jobs_state, "read", lambda _: None)
        assert _select_watch_client_id("127.0.0.1", 8188, "p", None) == (None, None)

    # The case the state-file-only guard missed entirely: a watcher that cannot
    # see the submitting run's state still has to refuse the id.
    def test_server_marker_is_honoured_without_a_state_file(self, monkeypatch):
        monkeypatch.setattr(jobs_state, "read", lambda _: None)
        monkeypatch.setattr(jobs, "_submitted_extra_data", lambda *_: _record(_marked()))
        assert _select_watch_client_id("127.0.0.1", 8188, "p", None) == (None, "borrowed")

    def test_explicit_override_cannot_bypass_borrowed_marker(self, monkeypatch):
        monkeypatch.setattr(jobs_state, "read", lambda _: None)
        monkeypatch.setattr(jobs, "_submitted_extra_data", lambda *_: _record(_marked()))
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
            return _record(_marked())

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
        monkeypatch.setattr(jobs, "_submitted_extra_data", lambda *_: _record(_marked()))

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
        monkeypatch.setattr(jobs, "_submitted_extra_data", lambda *_: _record(_marked()))

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
        # The reason rides the envelope too: `attached: false` alone cannot
        # tell a --json consumer to drop --client-id rather than retry.
        assert env["data"]["poll_reason"] == "borrowed"
        _validate_watch_envelope(env["data"])

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
        monkeypatch.setattr(jobs, "_submitted_extra_data", lambda *_: _record(_marked()))
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
        # The failure detail is lifted out of the snapshot it arrived in and
        # published under its own key. Asserting on `details` alone would pass
        # vacuously: that mapping is the whole payload, which never held `error`.
        assert env["error"]["details"]["execution_error"] == {"exception_message": "node 5 exploded", "node_id": "5"}
        assert "error" not in env["error"]["details"]["details"]

    def test_an_unmarked_server_record_still_resolves(self, monkeypatch):
        monkeypatch.setattr(jobs_state, "read", lambda _: None)
        monkeypatch.setattr(jobs, "_submitted_extra_data", lambda *_: _record({"client_id": "cli-minted"}))
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
        monkeypatch.setattr(jobs, "_submitted_extra_data", lambda *_: _record(read=False))

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
        assert env["data"]["poll_reason"] == "indeterminate"
        _validate_watch_envelope(env["data"])


class TestSubmittedExtraDataReportsWhetherTheRecordWasRead:
    """`(extra_data, record_read)` — the second element is the safety-critical one.

    It answers exactly one question: did we get this prompt's own record in our
    hands? A hit in either store is enough, because that IS the submitted
    record. Everything else is False, including a server that answered
    perfectly well and simply never mentioned the prompt — "not in the body I
    read" is not "was never marked".
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

    def test_a_queue_hit_counts_even_if_history_is_unreachable(self, monkeypatch):
        self._server(monkeypatch, queue=self.QUEUED, history=self.BOOM)
        assert _submitted_extra_data("127.0.0.1", 8188, "p")[:2] == ({"client_id": "cid"}, True)

    def test_a_history_hit_counts_even_if_the_queue_was_unreachable(self, monkeypatch):
        self._server(monkeypatch, queue=self.BOOM, history=self.IN_HISTORY)
        assert _submitted_extra_data("127.0.0.1", 8188, "p")[:2] == ({"client_id": "cid"}, True)

    @pytest.mark.parametrize(
        ("queue", "history", "why"),
        [
            (EMPTY_QUEUE, {}, "answered, prompt in neither store"),
            (EMPTY_QUEUE, BOOM, "history unreadable"),
            (BOOM, {}, "queue unreadable, so the prompt may be running right now"),
            (BOOM, BOOM, "neither store readable"),
            (["not", "a", "dict"], {}, "a 200 of the wrong shape is not a queue"),
            ({"error": "bad gateway"}, {}, "a proxy error page mentions no prompt"),
            ({}, {}, "an object with no queue sections places nothing"),
            ({"queue_running": "nonsense"}, {}, "an unwalkable section"),
            ({"queue_running": [[0, "p"]]}, {}, "matched, but too short to hold extra_data"),
            ({"queue_running": [[0, "p", {}, None, {}]]}, {}, "extra_data slot present but null"),
            ({"queue_running": [[0, "p", {}, [], {}]]}, {}, "extra_data slot is a list"),
            ({"queue_running": [[0, "p", {}, "x", {}]]}, {}, "extra_data slot is a string"),
            (EMPTY_QUEUE, {"p": {"prompt": [0, "p", {}, 7, {}]}}, "history slot is a number"),
            (EMPTY_QUEUE, {"p": {}}, "in history but no prompt tuple"),
        ],
        ids=[
            "absent-from-both",
            "history-unreadable",
            "queue-unreadable",
            "both-unreadable",
            "wrong-shaped-body",
            "proxy-error-page",
            "no-queue-sections",
            "unwalkable-section",
            "entry-missing-extra-slot",
            "history-entry-malformed",
            "slot-null",
            "slot-list",
            "slot-string",
            "history-slot-number",
        ],
    )
    def test_everything_else_is_not_a_record(self, monkeypatch, queue, history, why):
        self._server(monkeypatch, queue=queue, history=history)
        assert _submitted_extra_data("127.0.0.1", 8188, "p")[:2] == (None, False), why

    def test_a_marked_record_round_trips(self, monkeypatch):
        self._server(monkeypatch, queue={"queue_running": [[0, "p", {}, _marked(), {}]]}, history={})
        extra, record_read, _ids, _q = _submitted_extra_data("127.0.0.1", 8188, "p")
        assert record_read is True
        assert extra[jobs_state.BORROWED_CLIENT_ID_KEY] is True


class TestABodyThatMentionsNothingIsNotAnAllClear:
    """A server can answer and still tell us nothing about THIS prompt.

    The first fix taught the lookup to distrust a failed fetch. These are the
    neighbouring flavour: `/queue` comes back 200 with a body that simply does
    not place the prompt -- an object missing the queue sections, a proxy error
    page, or a genuinely empty queue once the record has been pruned. Reading
    any of those as "no marker, therefore safe" hands the caller's id straight
    to the socket while the borrowed run may still be live.
    """

    @pytest.fixture(autouse=True)
    def no_state_file(self, monkeypatch):
        monkeypatch.setattr(jobs_state, "read", lambda _: None)

    @pytest.mark.parametrize(
        "queue_body",
        [{}, {"error": "bad gateway"}, {"queue_running": [], "queue_pending": []}],
        ids=["no-queue-sections", "proxy-error-page", "empty-queue-record-pruned"],
    )
    def test_an_explicit_override_is_refused(self, monkeypatch, queue_body):
        monkeypatch.setattr(
            jobs,
            "_http_get_json",
            lambda url, **_kw: queue_body if url.endswith("/queue") else {},
        )
        with pytest.raises(_ClientIdRejected) as excinfo:
            _select_watch_client_id("127.0.0.1", 8188, "p", BORROWED)
        assert excinfo.value.reason == "indeterminate"

    @pytest.mark.parametrize(
        "queue_body",
        [{}, {"error": "bad gateway"}, {"queue_running": [], "queue_pending": []}],
        ids=["no-queue-sections", "proxy-error-page", "empty-queue-record-pruned"],
    )
    def test_without_an_override_it_polls(self, monkeypatch, queue_body):
        monkeypatch.setattr(
            jobs,
            "_http_get_json",
            lambda url, **_kw: queue_body if url.endswith("/queue") else {},
        )
        assert _select_watch_client_id("127.0.0.1", 8188, "p", None) == (None, "indeterminate")


class TestTheIdIsCheckedToNotJustThePrompt:
    """The marker rides a prompt; what it protects is a socket.

    Watching an ordinary, fully vouched-for prompt while naming a live
    client's id as `--client-id` evicts that client just as thoroughly as
    attaching to the borrowed run would have. Checking only the watched
    prompt's marker misses it entirely, so the id is checked against every
    borrowed run the queue is still carrying.
    """

    QUEUE = {
        "queue_running": [
            [0, "P_BORROWED", {}, {"client_id": BORROWED, jobs_state.BORROWED_CLIENT_ID_KEY: True}, {}],
            [1, "P_OTHER", {}, {"client_id": "ordinary"}, {}],
        ],
        "queue_pending": [],
    }

    @pytest.fixture(autouse=True)
    def live_queue(self, monkeypatch):
        monkeypatch.setattr(jobs_state, "read", lambda _: None)
        monkeypatch.setattr(
            jobs,
            "_http_get_json",
            lambda url, **_kw: self.QUEUE if url.endswith("/queue") else {},
        )

    def test_another_runs_borrowed_id_is_refused(self):
        with pytest.raises(_ClientIdRejected) as excinfo:
            _select_watch_client_id("127.0.0.1", 8188, "P_OTHER", BORROWED)
        assert excinfo.value.reason == "borrowed"

    def test_the_borrowed_run_itself_is_still_refused(self):
        with pytest.raises(_ClientIdRejected) as excinfo:
            _select_watch_client_id("127.0.0.1", 8188, "P_BORROWED", BORROWED)
        assert excinfo.value.reason == "borrowed"

    def test_an_unrelated_override_is_still_honoured(self):
        """The check must not turn --client-id into a no-op."""
        assert _select_watch_client_id("127.0.0.1", 8188, "P_OTHER", "my-own-id") == ("my-own-id", None)

    def test_the_ordinary_watch_is_untouched(self):
        assert _select_watch_client_id("127.0.0.1", 8188, "P_OTHER", None) == ("ordinary", None)

    def test_an_unread_queue_refuses_the_override_rather_than_waving_it_through(self, monkeypatch):
        """An empty census means "nobody is borrowing" only once the queue was
        walked. A readable state file vouches for THIS run and says nothing
        about whose socket the caller just named, so it must not unlock it."""
        state = jobs_state.new(prompt_id="P_OTHER", client_id="ordinary", workflow="w.json", where="local")
        monkeypatch.setattr(jobs_state, "read", lambda _: state)

        def boom(url, **_kw):
            if url.endswith("/queue"):
                raise RuntimeError("connection refused")
            return {}

        monkeypatch.setattr(jobs, "_http_get_json", boom)
        with pytest.raises(_ClientIdRejected) as excinfo:
            _select_watch_client_id("127.0.0.1", 8188, "P_OTHER", BORROWED)
        assert excinfo.value.reason == "indeterminate"
        # The no-flag path names no foreign id, so it still attaches.
        assert _select_watch_client_id("127.0.0.1", 8188, "P_OTHER", None) == ("ordinary", None)

    @pytest.mark.parametrize(
        "queue_body",
        [{"queue_running": []}, {"queue_pending": []}, {}, {"queue_running": [], "queue_pending": "x"}],
        ids=["only-running", "only-pending", "neither", "pending-unwalkable"],
    )
    def test_a_half_walkable_queue_is_not_a_census(self, monkeypatch, queue_body):
        """A borrowed run in the section we could not walk would be missing
        from the census without the census knowing it."""
        monkeypatch.setattr(jobs_state, "read", lambda _: None)
        monkeypatch.setattr(
            jobs,
            "_http_get_json",
            lambda url, **_kw: queue_body if url.endswith("/queue") else {},
        )
        with pytest.raises(_ClientIdRejected) as excinfo:
            _select_watch_client_id("127.0.0.1", 8188, "P_OTHER", BORROWED)
        assert excinfo.value.reason == "indeterminate"

    def test_the_borrowed_set_comes_from_the_same_read(self, monkeypatch):
        """One `/queue` body answers both questions, so they cannot disagree."""
        reads = []
        monkeypatch.setattr(
            jobs,
            "_http_get_json",
            lambda url, **_kw: (reads.append(url), self.QUEUE if url.endswith("/queue") else {})[1],
        )
        with pytest.raises(_ClientIdRejected):
            _select_watch_client_id("127.0.0.1", 8188, "P_OTHER", BORROWED)
        assert [u for u in reads if u.endswith("/queue")] == ["http://127.0.0.1:8188/queue"]


class TestAnAlreadyTerminalFailureIsTrimmedToo:
    """The short-circuit exit is a failed `jobs watch` as much as the live one.

    `_emit_terminal` keys its trimming and redaction on `execution_error`, so a
    failure left where `_snapshot` puts it reached the envelope whole.
    """

    SNAP = {
        "prompt_id": "p",
        "status": "error",
        "outputs": [],
        "error": {
            "exception_message": "boom",
            "node_id": "5",
            "traceback": [f"frame{i}" for i in range(12)],
            "current_inputs": {"api_key": "sk-SECRET-123"},
        },
    }

    @pytest.fixture(autouse=True)
    def terminal_failure(self, monkeypatch):
        monkeypatch.setattr(jobs, "_server_or_error", lambda *_a, **_k: True)
        monkeypatch.setattr(jobs, "_snapshot", lambda *_a: dict(self.SNAP))
        monkeypatch.setattr(jobs, "_history_completed_nodes", lambda *_a: set())

    def _env(self):
        result = CliRunner().invoke(jobs.app, ["watch", "p", "--where", "local"])
        assert result.exit_code == 1, result.output
        return _envelope(result.output)

    def test_the_failure_is_published_where_the_live_path_publishes_it(self):
        env = self._env()
        assert env["error"]["code"] == "execution_error"
        assert env["error"]["details"]["execution_error"]["exception_message"] == "boom"
        assert "error" not in env["error"]["details"]

    def test_a_secret_in_current_inputs_does_not_reach_the_envelope(self):
        assert "sk-SECRET-123" not in json.dumps(self._env())

    def test_the_traceback_is_capped_to_its_tail(self):
        env = self._env()
        assert env["error"]["details"]["execution_error"]["traceback"] == ["frame10", "frame11"]
        assert "frame0" not in json.dumps(env)


class TestTheOrdinaryCaseStillAttaches:
    """The guard must not turn every watch into a poll.

    Drives the REAL selector -- the integration tests elsewhere monkeypatch it
    away, so without this nothing pins that an ordinary run still opens a
    socket at all.
    """

    @pytest.fixture(autouse=True)
    def no_state_file(self, monkeypatch):
        monkeypatch.setattr(jobs_state, "read", lambda _: None)

    def test_a_queued_unmarked_run_resolves_and_attaches(self, monkeypatch):
        monkeypatch.setattr(
            jobs,
            "_http_get_json",
            lambda url, **_kw: (
                {"queue_running": [[0, "p", {}, {"client_id": "submitter"}, {}]], "queue_pending": []}
                if url.endswith("/queue")
                else {}
            ),
        )
        assert _select_watch_client_id("127.0.0.1", 8188, "p", None) == ("submitter", None)

    def test_an_override_is_honoured_once_the_record_vouches(self, monkeypatch):
        monkeypatch.setattr(
            jobs,
            "_http_get_json",
            lambda url, **_kw: (
                {"queue_running": [[0, "p", {}, {"client_id": "submitter"}, {}]], "queue_pending": []}
                if url.endswith("/queue")
                else {}
            ),
        )
        assert _select_watch_client_id("127.0.0.1", 8188, "p", "my-own-id") == ("my-own-id", None)

    def test_the_watch_opens_a_socket_for_it(self, monkeypatch):
        """End to end through the real selector: a socket is actually opened,
        under the resolved submitting id."""
        opened = {}

        class _WS:
            def connect(self, url):
                opened["url"] = url

            def settimeout(self, _t):
                pass

            def recv(self):
                raise AssertionError("stop")

            def close(self):
                pass

        monkeypatch.setattr(jobs, "_server_or_error", lambda *_a, **_k: True)
        monkeypatch.setattr(jobs, "_snapshot", lambda *_: {"prompt_id": "p", "status": "running", "outputs": []})
        monkeypatch.setattr(jobs, "_history_completed_nodes", lambda *_: set())
        monkeypatch.setattr(
            jobs,
            "_http_get_json",
            lambda url, **_kw: (
                {"queue_running": [[0, "p", {}, {"client_id": "submitter"}, {}]], "queue_pending": []}
                if url.endswith("/queue")
                else {}
            ),
        )
        monkeypatch.setattr(jobs, "WebSocket", lambda *_a, **_k: _WS())

        CliRunner().invoke(jobs.app, ["watch", "p", "--where", "local", "--timeout", "1"])
        assert "clientId=submitter" in opened.get("url", "")


class TestABlankOverrideIsRejectedLikeRun:
    @pytest.mark.parametrize("value", ["", "   "])
    def test_blank_is_refused_rather_than_used_verbatim(self, monkeypatch, value):
        monkeypatch.setattr(jobs, "_server_or_error", lambda *_a, **_k: True)
        monkeypatch.setattr(jobs, "_snapshot", lambda *_: {"prompt_id": "p", "status": "running", "outputs": []})
        result = CliRunner().invoke(jobs.app, ["watch", "p", "--where", "local", "--client-id", value])
        assert result.exit_code == 1, result.output
        env = _envelope(result.output)
        assert env["error"]["code"] == "client_id_rejected"
        assert env["error"]["details"]["reason"] == "empty"

    # Validating it late would let the paths that return first swallow it: a
    # cloud watch never forwards the flag, and an unreachable local server
    # exits at the probe. `comfy run` rejects it just as early.
    def test_a_cloud_target_still_refuses_it(self, monkeypatch):
        monkeypatch.setattr(
            jobs,
            "_cloud_watch",
            lambda *_a, **_k: (_ for _ in ()).throw(AssertionError("cloud watch swallowed a blank --client-id")),
        )
        result = CliRunner().invoke(jobs.app, ["watch", "p", "--where", "cloud", "--client-id", "  "])
        assert result.exit_code == 1, result.output
        env = _envelope(result.output)
        assert env["error"]["code"] == "client_id_rejected"
        assert env["error"]["details"]["reason"] == "empty"

    def test_a_server_that_is_down_still_refuses_it(self, monkeypatch):
        monkeypatch.setattr(
            jobs,
            "_server_or_error",
            lambda *_a, **_k: (_ for _ in ()).throw(AssertionError("probed the server before validating the flag")),
        )
        result = CliRunner().invoke(jobs.app, ["watch", "p", "--where", "local", "--client-id", ""])
        assert result.exit_code == 1, result.output
        assert _envelope(result.output)["error"]["details"]["reason"] == "empty"

    def test_a_padded_id_is_stripped_not_attached_verbatim(self, monkeypatch):
        monkeypatch.setattr(jobs_state, "read", lambda _: None)
        monkeypatch.setattr(
            jobs,
            "_http_get_json",
            lambda url, **_kw: (
                {"queue_running": [[0, "p", {}, {"client_id": "submitter"}, {}]], "queue_pending": []}
                if url.endswith("/queue")
                else {}
            ),
        )
        monkeypatch.setattr(jobs, "_server_or_error", lambda *_a, **_k: True)
        monkeypatch.setattr(jobs, "_snapshot", lambda *_: {"prompt_id": "p", "status": "running", "outputs": []})
        monkeypatch.setattr(jobs, "_history_completed_nodes", lambda *_: set())
        opened = {}

        class _WS:
            def connect(self, url):
                opened["url"] = url

            def settimeout(self, _t):
                pass

            def recv(self):
                raise AssertionError("stop")

            def close(self):
                pass

        monkeypatch.setattr(jobs, "WebSocket", lambda *_a, **_k: _WS())
        CliRunner().invoke(jobs.app, ["watch", "p", "--where", "local", "--timeout", "1", "--client-id", "  pad  "])
        assert "clientId=pad" in opened.get("url", "")
