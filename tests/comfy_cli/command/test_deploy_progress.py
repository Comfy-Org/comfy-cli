"""`comfy deploy up --watch` and `comfy deploy status` show a deployment's progress."""

from __future__ import annotations

import copy
import errno
import importlib
import inspect
import json
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path

import jsonschema
import pytest
from deploy_up_support import FakeBuilder, FakeDeploy, deployment, write_spec
from typer.testing import CliRunner

from comfy_cli.cmdline import app
from comfy_cli.command.build_spec import JsonObject
from comfy_cli.command.deploy_progress import (
    EVENT_PROGRESS,
    STALE_SECONDS,
    DeployWatchReporter,
    describe,
    is_stale,
    progress_of,
)
from comfy_cli.command.deploy_runtime import poll_deployment
from comfy_cli.output.renderer import OutputMode, Renderer

GB = 1 << 30
NOW = datetime(2026, 9, 20, 2, 1, 20, tzinfo=timezone.utc)


def _staging(**over) -> JsonObject:
    progress: JsonObject = {
        "step": "staging_models",
        "modelsDone": 0,
        "modelsTotal": 2,
        "bytesDone": 3 * GB,
        "bytesTotal": 7 * GB,
        "currentModel": "models/checkpoints/sd_xl_base_1.0.safetensors",
        "bytesPerSecond": 44 * (1 << 20),
        "etaSeconds": 85,
        "attempt": 1,
        "startedAt": "2026-09-20T01:59:55Z",
        "updatedAt": "2026-09-20T02:01:15Z",
    }
    progress.update(over)
    return {key: value for key, value in progress.items() if value is not None}


def _step(step: str, **over) -> JsonObject:
    stamp = "2026-09-20T02:02:45Z"
    return {"step": step, "attempt": 1, "startedAt": stamp, "updatedAt": stamp, **over}


# ----- the wording -----


@pytest.mark.parametrize(
    ("progress", "now", "expected"),
    [
        pytest.param(
            _staging(),
            NOW,
            "Staging models: model 1 of 2 sd_xl_base_1.0.safetensors, 3.0 GB of 7.0 GB, 44.0 MB/s, 1m 25s left",
            id="a model in flight",
        ),
        pytest.param(
            _staging(bytesTotal=None, etaSeconds=None),
            NOW,
            "Staging models: model 1 of 2 sd_xl_base_1.0.safetensors, 3.0 GB copied, 44.0 MB/s",
            id="a release nobody measured has no total and no time left",
        ),
        pytest.param(
            _staging(bytesTotalIsFloor=True, etaSeconds=None),
            NOW,
            "Staging models: model 1 of 2 sd_xl_base_1.0.safetensors, 3.0 GB of at least 7.0 GB, 44.0 MB/s",
            id="a floor says at least",
        ),
        pytest.param(
            _staging(bytesDone=0, bytesTotal=0, currentModel=None, bytesPerSecond=None, etaSeconds=None, modelsDone=2),
            NOW,
            "Staging models: every model is already in place",
            id="nothing to copy",
        ),
        pytest.param(
            _staging(bytesPerSecond=None, etaSeconds=None, bytesDone=0),
            NOW,
            "Staging models: model 1 of 2 sd_xl_base_1.0.safetensors, 0 B of 7.0 GB",
            id="the first seconds have no rate",
        ),
        pytest.param(
            _step("creating_endpoint", updatedAt="2026-09-20T02:01:20Z"),
            NOW,
            "Creating the endpoint",
            id="a step with nothing to count, the moment it begins",
        ),
        pytest.param(
            _step("waiting_for_worker", updatedAt="2026-09-20T01:59:05Z"),
            NOW,
            "Waiting for the first worker: 2m 15s so far",
            id="a step with nothing to count says how long it has been waiting",
        ),
        pytest.param(
            _step("waiting_for_worker", updatedAt="2026-09-20T01:58:00Z"),
            NOW,
            "Waiting for the first worker: 3m 20s so far",
            id="a long wait is never called stale: the service writes this step once",
        ),
        pytest.param(
            _step("waiting_for_worker", attempt=2, updatedAt="2026-09-20T02:01:20Z"),
            NOW,
            "Waiting for the first worker (attempt 2, this step was restarted)",
            id="a retried step says so",
        ),
        pytest.param(
            _step("warming_cache", updatedAt="2026-09-20T02:01:20Z"),
            NOW,
            "warming cache",
            id="a step this version never heard of",
        ),
        pytest.param(
            _staging(),
            NOW + timedelta(seconds=STALE_SECONDS + 35),
            "Staging models: model 1 of 2 sd_xl_base_1.0.safetensors, 3.0 GB of 7.0 GB, 44.0 MB/s, 1m 25s left"
            " (last update 1m 40s ago, so these numbers may be stale)",
            id="an old sample is called old, and its numbers are not moved",
        ),
    ],
)
def test_describe(progress, now, expected) -> None:
    assert describe(progress, now=now) == expected


@pytest.mark.parametrize("value", [None, "staging", 3, [], {}, {"step": 7}])
def test_anything_that_is_not_a_progress_object_reads_as_none(value) -> None:
    """Progress narrates a deploy. A shape this build cannot read must cost the
    narration, never the command."""
    row = deployment("dep-1")
    row["progress"] = value
    assert progress_of(row) is None


def test_a_deployment_that_sends_no_progress_has_none() -> None:
    assert progress_of(deployment("dep-1")) is None


# ----- the reporter -----


class _Clock:
    def __init__(self) -> None:
        self.now = NOW

    def __call__(self) -> datetime:
        return self.now


def _renderer(mode: OutputMode, tmp_path: Path):
    out, err = (tmp_path / "out").open("w+"), (tmp_path / "err").open("w+")
    renderer = Renderer(mode=mode)
    renderer.pretty_stream = out if mode is OutputMode.PRETTY else err
    renderer.machine_stream = out
    return renderer, out, err


def _read(handle) -> str:
    handle.flush()
    handle.seek(0)
    return handle.read()


def _snapshot(status: str, progress: JsonObject | None) -> JsonObject:
    row = deployment("dep-1", status=status)
    if progress is not None:
        row["progress"] = progress
    return row


def test_a_sample_with_no_stamp_is_reported_when_its_numbers_move_and_not_when_they_repeat(tmp_path) -> None:
    """`updatedAt` is optional to the lenient parser; a sample without one must
    not be mistaken for the last one just because both have no stamp."""
    # Given samples the service never stamped
    renderer, out, _ = _renderer(OutputMode.NDJSON, tmp_path)
    reporter = DeployWatchReporter(renderer, "dep-1", now=_Clock())
    first = _staging(updatedAt=None)
    second = _staging(bytesDone=4 * GB, updatedAt=None)

    # When
    for progress in (first, first, second, second, first):
        reporter.snapshot(_snapshot("provisioning", progress))

    # Then: one event per change, none per repeat
    events = [json.loads(line) for line in _read(out).splitlines()]
    assert [event["progress"]["bytesDone"] for event in events] == [3 * GB, 4 * GB, 3 * GB]


class _LiveRenderer:
    """A pretty renderer on a terminal, so the reporter picks the live surface."""

    class _Console:
        is_terminal = True

    def __init__(self) -> None:
        self.said: list[str] = []

    def is_pretty(self) -> bool:
        return True

    def console(self) -> _LiveRenderer._Console:
        return self._Console()

    def info(self, message: str, *, hint: str | None = None) -> None:
        self.said.append(message)


class _DisplayThatRefusesTheTask:
    """Rich's Progress with a stream that fails on the first redraw after start."""

    stopped = 0

    def __init__(self, *columns: object, **options: object) -> None:
        pass

    def start(self) -> None:
        pass

    def add_task(self, description: str, **fields: object) -> int:
        raise OSError(errno.EIO, "broken terminal")

    def stop(self) -> None:
        type(self).stopped += 1


def test_a_display_refused_at_the_first_redraw_mutes_the_watch_instead_of_raising(monkeypatch) -> None:
    # Given a terminal whose stream refuses the redraw that adding the task makes
    import rich.progress

    monkeypatch.setattr(rich.progress, "Progress", _DisplayThatRefusesTheTask)
    _DisplayThatRefusesTheTask.stopped = 0
    renderer = _LiveRenderer()
    reporter = DeployWatchReporter(renderer, "dep-1", now=_Clock())

    # When: two samples arrive and nothing raises
    reporter.snapshot(_snapshot("provisioning", _staging()))
    reporter.snapshot(_snapshot("provisioning", _staging(bytesDone=4 * GB)))
    reporter.close()

    # Then: the half-opened display was closed once, never reopened, and the
    # reporter stayed quiet on the stream it could not write to
    assert reporter._muted is True
    assert reporter._live is None and reporter._live_task is None
    assert _DisplayThatRefusesTheTask.stopped == 1
    assert renderer.said == []


def test_an_ndjson_watch_emits_one_event_per_new_sample_and_carries_the_object_unchanged(tmp_path) -> None:
    # Given a poll five times as fast as the service writes
    renderer, out, _ = _renderer(OutputMode.NDJSON, tmp_path)
    reporter = DeployWatchReporter(renderer, "dep-1", now=_Clock())
    first, second = _staging(), _staging(bytesDone=4 * GB, updatedAt="2026-09-20T02:01:25Z")

    # When
    for progress in (first, first, first, second, second):
        reporter.snapshot(_snapshot("provisioning", progress))

    # Then
    events = [json.loads(line) for line in _read(out).splitlines()]
    assert [event["progress"] for event in events] == [first, second]
    assert events[0] == {
        "schema": "event/1",
        "type": EVENT_PROGRESS,
        "deployment_id": "dep-1",
        "status": "provisioning",
        "stale": False,
        "progress": first,
    }
    schema_path = Path(__file__).parents[3] / "comfy_cli" / "schemas" / "deploy_progress_event.json"
    validator = jsonschema.Draft202012Validator(json.loads(schema_path.read_text(encoding="utf-8")))
    for event in events:
        validator.validate(event)


def test_a_sample_that_goes_stale_is_reported_once_more_and_never_again(tmp_path) -> None:
    # Given
    renderer, out, _ = _renderer(OutputMode.NDJSON, tmp_path)
    clock = _Clock()
    reporter = DeployWatchReporter(renderer, "dep-1", now=clock)
    sample = _snapshot("provisioning", _staging())

    # When the service stops writing for two minutes
    reporter.snapshot(sample)
    clock.now = NOW + timedelta(seconds=120)
    reporter.snapshot(sample)
    clock.now = NOW + timedelta(seconds=122)
    reporter.snapshot(sample)

    # Then
    events = [json.loads(line) for line in _read(out).splitlines()]
    assert [event["stale"] for event in events] == [False, True]
    assert events[0]["progress"] == events[1]["progress"], "a stale sample's numbers are not moved"


def test_a_json_watch_puts_events_on_stderr_so_stdout_stays_one_envelope(tmp_path, capsys) -> None:
    # Given
    renderer = Renderer(mode=OutputMode.JSON)
    reporter = DeployWatchReporter(renderer, "dep-1", now=_Clock())

    # When
    reporter.snapshot(_snapshot("provisioning", _staging()))

    # Then
    captured = capsys.readouterr()
    assert captured.out == ""
    assert json.loads(captured.err)["type"] == EVENT_PROGRESS


def test_piped_pretty_output_is_plain_lines_with_no_carriage_returns(tmp_path) -> None:
    # Given
    renderer, out, _ = _renderer(OutputMode.PRETTY, tmp_path)
    reporter = DeployWatchReporter(renderer, "dep-1", now=_Clock())

    # When
    reporter.snapshot(_snapshot("provisioning", _staging()))
    reporter.snapshot(_snapshot("provisioning", _staging()))
    reporter.snapshot(_snapshot("starting", _step("waiting_for_worker")))
    reporter.snapshot(_snapshot("ready", None))
    reporter.close()

    # Then
    text = _read(out)
    assert "\r" not in text
    assert text.count("Staging models:") == 1
    assert text.count("Waiting for the first worker") == 1


def test_an_older_service_that_sends_no_progress_reports_nothing(tmp_path) -> None:
    # Given
    renderer, out, err = _renderer(OutputMode.NDJSON, tmp_path)
    reporter = DeployWatchReporter(renderer, "dep-1", now=_Clock())

    # When
    for status in ("queued", "provisioning", "starting", "ready"):
        reporter.snapshot(_snapshot(status, None))

    # Then
    assert _read(out) == "" and _read(err) == ""


def test_a_stream_that_refuses_a_write_mutes_the_reporter_instead_of_failing_the_watch(tmp_path) -> None:
    # Given
    class Refusing(Renderer):
        def progress_event(self, type: str, **fields) -> None:
            raise BrokenPipeError

    reporter = DeployWatchReporter(Refusing(mode=OutputMode.NDJSON), "dep-1", now=_Clock())

    # When / Then
    reporter.snapshot(_snapshot("provisioning", _staging()))
    reporter.snapshot(_snapshot("provisioning", _staging(updatedAt="2026-09-20T02:01:25Z")))


def test_poll_deployment_hands_every_read_to_the_watcher_the_terminal_one_included() -> None:
    # Given
    client = FakeDeploy([deployment("dep-1", status="queued")], get_statuses=["provisioning", "starting", "ready"])
    seen: list[str] = []

    # When
    final = poll_deployment(client, "dep-1", lambda _: None, lambda row: seen.append(str(row["status"])))

    # Then
    assert seen == ["provisioning", "starting", "ready"]
    assert final["status"] == "ready"


# ----- the commands -----


class ProgressDeploy(FakeDeploy):
    """Serves one (status, progress) pair per read, then repeats the last."""

    def __init__(self, rows: list[JsonObject], reads: list[tuple[str, JsonObject | None]]) -> None:
        super().__init__(rows)
        self.reads = list(reads)

    def get_deployment(self, deployment_id: str) -> JsonObject:
        with self._lock:
            row = self.rows[deployment_id]
            if self.reads:
                status, progress = self.reads.pop(0)
                row["status"] = status
                row.pop("progress", None)
                if progress is not None:
                    row["progress"] = copy.deepcopy(progress)
            return copy.deepcopy(row)


def _status_row(status: str) -> JsonObject:
    row = deployment("dep-status", release_id="release-5", status=status, maximum=2)
    row.update({"releaseId": "release-5", "endpointUrl": None, "error": None, "serving": None, "stopReason": None})
    return row


def _release(version: int) -> JsonObject:
    return {"id": f"release-{version}", "buildId": "build-1", "version": version, "deployable": True}


def _install_status(monkeypatch, client: FakeDeploy, sleep) -> None:
    module = importlib.import_module("comfy_cli.command.deploy_status")
    monkeypatch.setattr(module, "_command_clients", lambda: (FakeBuilder([_release(5)]), client))
    monkeypatch.setattr(module, "_sleep", sleep)


def _envelope(result) -> dict:
    return json.loads([line for line in result.stdout.splitlines() if line.strip()][-1])


def _schema(name: str) -> JsonObject:
    path = Path(__file__).parents[3] / "comfy_cli" / "schemas" / name
    return json.loads(path.read_text(encoding="utf-8"))


def test_status_carries_the_progress_object_while_the_deployment_comes_up(tmp_path, monkeypatch) -> None:
    # Given
    row = _status_row("provisioning")
    row["progress"] = _staging()
    _install_status(monkeypatch, FakeDeploy([row]), lambda _: None)

    # When
    result = CliRunner().invoke(app, ["--json", "deploy", "status", str(write_spec(tmp_path))])

    # Then
    assert result.exit_code == 0, result.stderr
    data = _envelope(result)["data"]
    assert data["progress"] == _staging()
    jsonschema.Draft202012Validator(_schema("deploy_status.json")).validate(data)


def test_status_omits_progress_once_the_deployment_is_ready(tmp_path, monkeypatch) -> None:
    # Given
    _install_status(monkeypatch, FakeDeploy([_status_row("ready")]), lambda _: None)

    # When
    result = CliRunner().invoke(app, ["--json", "deploy", "status", str(write_spec(tmp_path))])

    # Then
    assert result.exit_code == 0, result.stderr
    assert "progress" not in _envelope(result)["data"]


def test_status_prints_the_progress_line_for_a_person(tmp_path, monkeypatch) -> None:
    # Given
    row = _status_row("provisioning")
    row["progress"] = _staging(updatedAt=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"))
    _install_status(monkeypatch, FakeDeploy([row]), lambda _: None)

    # When
    result = CliRunner().invoke(
        app, ["--no-json", "deploy", "status", str(write_spec(tmp_path))], env={"COLUMNS": "400"}
    )

    # Then
    assert result.exit_code == 0
    assert "Deployment dep-status: provisioning" in result.stdout
    assert "Staging models: model 1 of 2 sd_xl_base_1.0.safetensors, 3.0 GB of 7.0 GB" in result.stdout
    assert "may be stale" not in result.stdout


def test_status_watch_streams_progress_then_settles_on_an_envelope_without_it(tmp_path, monkeypatch) -> None:
    # Given
    reads = [
        ("provisioning", _staging()),
        ("provisioning", _staging(bytesDone=6 * GB, updatedAt="2026-09-20T02:01:25Z")),
        ("starting", _step("waiting_for_worker")),
        ("ready", None),
    ]
    _install_status(monkeypatch, ProgressDeploy([_status_row("queued")], reads), lambda _: None)

    # When
    result = CliRunner().invoke(app, ["--json-stream", "deploy", "status", str(write_spec(tmp_path)), "--watch"])

    # Then
    assert result.exit_code == 0, result.stderr
    lines = [json.loads(line) for line in result.stdout.splitlines() if line.strip()]
    events = [line for line in lines if line.get("type") == EVENT_PROGRESS]
    assert [event["progress"]["step"] for event in events] == [
        "staging_models",
        "staging_models",
        "waiting_for_worker",
    ]
    assert lines[-1]["type"] == "envelope"
    assert lines[-1]["data"]["deployment"]["status"] == "ready"
    assert "progress" not in lines[-1]["data"]


def test_interrupting_a_status_watch_leaves_the_deployment_alone_and_says_how_to_reattach(
    tmp_path, monkeypatch
) -> None:
    # Given a watch whose first wait is interrupted
    client = ProgressDeploy([_status_row("queued")], [("provisioning", _staging())])

    def interrupt(_: float) -> None:
        raise KeyboardInterrupt

    _install_status(monkeypatch, client, interrupt)

    # When
    result = CliRunner().invoke(
        app, ["--json", "deploy", "status", str(write_spec(tmp_path)), "--watch"], env={"COLUMNS": "400"}
    )

    # Then
    assert result.exit_code == 130
    assert "keeps coming up on our side" in result.stderr
    assert "comfy deploy status --deployment dep-status --watch" in result.stderr
    data = _envelope(result)["data"]
    assert data["deployment"]["status"] == "provisioning"
    assert data["progress"]["step"] == "staging_models"
    assert client.update_calls == [] and client.start_calls == []


def test_up_watch_streams_progress_and_interrupting_it_stops_nothing(tmp_path, monkeypatch) -> None:
    # Given
    module = importlib.import_module("comfy_cli.command.deploy")

    class CreatedThenStaging(ProgressDeploy):
        def get_deployment(self, deployment_id: str) -> JsonObject:
            # The create path reads once before any watch; leave that read alone.
            if not self.confirmed:
                self.confirmed = True
                return FakeDeploy.get_deployment(self, deployment_id)
            return super().get_deployment(deployment_id)

    client = CreatedThenStaging([], [("provisioning", _staging())])
    client.confirmed = False
    monkeypatch.setattr(module, "_command_clients", lambda: (FakeBuilder(), client))

    def interrupt(_: float) -> None:
        raise KeyboardInterrupt

    monkeypatch.setattr(module, "_sleep", interrupt)

    # When
    result = CliRunner().invoke(
        app,
        ["--json", "deploy", "up", str(write_spec(tmp_path)), "--gpu", "l4", "--region", "US-MO-2", "--watch"],
        env={"COLUMNS": "400"},
    )

    # Then
    assert result.exit_code == 130
    events = [json.loads(line) for line in result.stderr.splitlines() if line.startswith("{")]
    assert [event["type"] for event in events] == [EVENT_PROGRESS]
    assert "comfy deploy status --deployment dep-1 --watch" in result.stderr
    data = _envelope(result)["data"]
    assert data["deployment"]["status"] == "provisioning"
    assert data["progress"]["bytesDone"] == 3 * GB
    jsonschema.Draft202012Validator(_schema("deploy_up.json")).validate(data)


def test_only_the_staging_step_can_go_stale() -> None:
    """A step the service writes once is not a doubtful read, it is a wait.

    Staging rewrites its numbers every few seconds, so silence there means the
    numbers on screen may have moved on. The other two steps carry nothing that
    moves and are written once when they begin, so their age is the length of
    the wait and calling it stale tells the reader to distrust a true statement.
    """
    # Given a sample from each step, all two minutes old
    old = NOW + timedelta(seconds=120)

    # Then
    assert is_stale(_staging(), old) is True
    assert is_stale(_step("creating_endpoint"), old) is False
    assert is_stale(_step("waiting_for_worker"), old) is False
    assert "may be stale" not in describe(_step("waiting_for_worker"), now=old)


def test_deploy_up_watches_without_being_asked(tmp_path) -> None:
    """`up` starts a wait of several minutes, so it follows it by default.

    Asserted through the help a person actually reads, not through Typer's
    parameter objects: what matters is that the pair is offered and that the
    default shown is to watch.
    """
    # Given
    from comfy_cli.command import deploy as deploy_module

    # When
    help_text = CliRunner().invoke(app, ["deploy", "up", "--help"]).stdout

    # Then
    # Rich draws the help in a box, puts the two spellings of the flag in
    # separate cells, and wraps at whatever width it detects (80 on CI), so a
    # flag can straddle a line break. Compare with every escape code, box
    # character and space removed.
    flat = re.sub(r"\s+|[\u2500-\u257f]", "", re.sub(r"\x1b\[[0-9;]*m", "", help_text))
    assert "--watch" in flat and "--no-watch" in flat
    assert "[default:watch]" in flat
    assert inspect.signature(deploy_module.up_cmd).parameters["watch"].default is True
    # `status` answers a question and exits, so there watching stays opt-in.
    assert inspect.signature(deploy_module.status_cmd).parameters["watch"].default is False
