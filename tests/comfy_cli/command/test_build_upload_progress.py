"""What `comfy build push` says while a blob uploads.

A model upload is one streamed PUT that can run for an hour. These tests pin the
three things a reader gets out of it: the plan before the first byte, numbers
that keep arriving while the file moves (a stall included), and one completion
per file, in the shape each output mode promises.
"""

from __future__ import annotations

import io
import json
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

import jsonschema
import pytest
import requests

from comfy_cli.builder_api import BuilderClient, _CountingReader
from comfy_cli.caller import Caller
from comfy_cli.command.build_push import PushPreparation, PushUpload, already_held_count, upload_assets
from comfy_cli.command.build_upload_progress import (
    SlidingRate,
    UploadProgressReporter,
    human_bytes,
    human_seconds,
    plan_line,
)
from comfy_cli.output.renderer import OutputMode, Renderer, reset_renderer_for_testing

SCHEMA = Path(__file__).parent.parent.parent.parent / "comfy_cli" / "schemas" / "build_push_event.json"
GB = 1024**3


@pytest.fixture(autouse=True)
def reset_singleton():
    reset_renderer_for_testing()
    yield
    reset_renderer_for_testing()


class FakeClock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


@dataclass(frozen=True)
class Item:
    kind: str
    filename: str
    size_bytes: int


def _renderer(mode: OutputMode) -> Renderer:
    renderer = Renderer.resolve(
        is_stdout_tty=False, env={}, caller=Caller(kind="user", agentic=False, source_env=None), json_flag=True
    )
    renderer.mode = mode
    return renderer


def _lines(text: str) -> list[dict]:
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def _validate(events: list[dict]) -> None:
    validator = jsonschema.Draft202012Validator(json.loads(SCHEMA.read_text(encoding="utf-8")))
    for event in events:
        validator.validate(event)


# ----- the byte-counting reader -----


def test_the_counting_reader_reports_every_block_it_hands_over(tmp_path: Path) -> None:
    # Given
    path = tmp_path / "m.safetensors"
    path.write_bytes(b"x" * 100)
    seen: list[int] = []

    # When
    with path.open("rb") as raw:
        reader = _CountingReader(raw, seen.append)
        blocks = [reader.read(40), reader.read(40), reader.read(40), reader.read(40)]

    # Then: an empty read at EOF reports nothing
    assert [len(block) for block in blocks] == [40, 40, 20, 0]
    assert seen == [40, 40, 20]


def test_a_counted_upload_still_declares_its_length(tmp_path: Path) -> None:
    """A presigned GCS PUT refuses chunked transfer, so the wrapper must leave
    `requests` able to size the body exactly as it sizes the bare file."""
    # Given
    path = tmp_path / "m.safetensors"
    path.write_bytes(b"x" * 4096)

    # When
    with path.open("rb") as raw:
        body = _CountingReader(raw, lambda n: None)
        prepared = requests.Request("PUT", "https://storage.example/put?sig=1", data=body).prepare()

    # Then
    assert prepared.headers["Content-Length"] == "4096"
    assert "Transfer-Encoding" not in prepared.headers


def test_upload_blob_streams_through_the_counter_and_keeps_its_request(monkeypatch, tmp_path: Path) -> None:
    # Given
    captured: dict = {}

    class _Resp:
        status_code = 200

        def raise_for_status(self) -> None:
            pass

    def fake_put(url, data=None, headers=None, timeout=None, allow_redirects=None):
        captured.update(headers=headers, timeout=timeout, allow_redirects=allow_redirects)
        captured["body"] = b"".join(iter(lambda: data.read(16384), b""))
        return _Resp()

    monkeypatch.setattr("comfy_cli.builder_api.requests.put", fake_put)
    path = tmp_path / "m.safetensors"
    path.write_bytes(b"y" * 40000)
    seen: list[int] = []

    # When
    BuilderClient("https://builder.test/", "jwt").upload_blob("https://storage.example/put?sig=1", path, seen.append)

    # Then
    assert captured["body"] == b"y" * 40000
    assert sum(seen) == 40000
    assert captured["headers"] == {"x-goog-if-generation-match": "0"}
    assert captured["timeout"] == (10, 600)
    assert captured["allow_redirects"] is False


def test_upload_blob_without_a_callback_sends_the_bare_file(monkeypatch, tmp_path: Path) -> None:
    # Given
    captured: dict = {}

    class _Resp:
        status_code = 200

        def raise_for_status(self) -> None:
            pass

    def fake_put(url, data=None, **kwargs):
        captured["data"] = data
        return _Resp()

    monkeypatch.setattr("comfy_cli.builder_api.requests.put", fake_put)
    path = tmp_path / "m.safetensors"
    path.write_bytes(b"z")

    # When
    BuilderClient("https://builder.test/", "jwt").upload_blob("https://storage.example/put?sig=1", path)

    # Then
    assert isinstance(captured["data"], io.BufferedReader)


# ----- the sliding rate -----


def test_the_rate_is_unknown_until_time_has_passed() -> None:
    assert SlidingRate(FakeClock()).bytes_per_second() is None


def test_a_stall_reads_as_a_falling_rate_and_then_zero() -> None:
    # Given ten seconds at 100 bytes a second
    clock = FakeClock()
    rate = SlidingRate(clock, window=10.0)
    done = 0
    for _ in range(10):
        clock.advance(1.0)
        done += 100
        rate.sample(done)
    assert rate.bytes_per_second() == pytest.approx(100.0)

    # When the socket stalls: time passes, the byte count does not
    readings = []
    for _ in range(10):
        clock.advance(1.0)
        rate.sample(done)
        readings.append(rate.bytes_per_second())

    # Then a lifetime average would still say 50; the window says it stopped
    assert readings == sorted(readings, reverse=True)
    assert readings[4] == pytest.approx(50.0)
    assert readings[-1] == 0


# ----- formatting -----


@pytest.mark.parametrize(
    "files, total, held, expected",
    [
        (3, 53 * GB, 2, "3 files, 53.0 GB to upload, 2 already held"),
        (1, 1536, 0, "1 file, 1.5 KB to upload, 0 already held"),
        (0, 0, 4, "0 files, 0 B to upload, 4 already held"),
    ],
)
def test_the_plan_line(files: int, total: int, held: int, expected: str) -> None:
    assert plan_line(files, total, held) == expected


def test_sizes_and_durations_read_the_way_a_person_says_them() -> None:
    assert human_bytes(45.3 * 1024 * 1024) == "45.3 MB"
    assert human_bytes(3 * 1024**4) == "3.0 TB"
    assert human_seconds(42) == "42s"
    assert human_seconds(7 * 60 + 5) == "7m 05s"
    assert human_seconds(2 * 3600 + 3 * 60) == "2h 03m"


# ----- events for an agent -----


def _upload_one_file(reporter: UploadProgressReporter, clock: FakeClock, item: Item, steps: list[tuple[float, int]]):
    with reporter.uploading(item, 1, 1) as progress:
        for seconds, sent in steps:
            clock.advance(seconds)
            progress(sent)
            reporter.tick()


def test_json_stream_mode_puts_plan_progress_and_completion_on_stdout() -> None:
    # Given
    stdout = io.StringIO()
    renderer = _renderer(OutputMode.NDJSON)
    renderer.machine_stream = stdout
    clock = FakeClock()
    reporter = UploadProgressReporter(renderer, clock=clock, ticker=False)
    item = Item("model", "m.safetensors", 1000)

    # When: ten one-second ticks, 100 bytes each
    reporter.plan([item], already_held=2)
    _upload_one_file(reporter, clock, item, [(1.0, 100)] * 10)

    # Then
    events = _lines(stdout.getvalue())
    _validate(events)
    assert all(event["schema"] == "event/1" for event in events)
    assert events[0] == {
        "schema": "event/1",
        "type": "upload_plan",
        "files": 1,
        "bytes_total": 1000,
        "already_held": 2,
    }
    progress = [event for event in events if event["type"] == "upload_progress"]
    # One when the file starts, then one every two seconds: throttled, not per tick.
    assert [event["bytes_done"] for event in progress] == [0, 200, 400, 600, 800, 1000]
    assert progress[0]["bytes_per_second"] is None and progress[0]["eta_seconds"] is None
    assert progress[2] == {
        "schema": "event/1",
        "type": "upload_progress",
        "file": "m.safetensors",
        "kind": "model",
        "index": 1,
        "of": 1,
        "bytes_done": 400,
        "bytes_total": 1000,
        "bytes_per_second": 100,
        "eta_seconds": 6,
        "overall_bytes_done": 400,
        "overall_bytes_total": 1000,
        "overall_eta_seconds": 6,
    }
    assert events[-1] == {
        "schema": "event/1",
        "type": "upload_complete",
        "file": "m.safetensors",
        "kind": "model",
        "index": 1,
        "of": 1,
        "bytes_total": 1000,
        "seconds": 10.0,
        "bytes_per_second": 100,
        "deduplicated": False,
        "overall_bytes_done": 1000,
        "overall_bytes_total": 1000,
    }


def test_json_mode_keeps_stdout_for_the_envelope_and_reports_on_stderr(capsys) -> None:
    # Given
    renderer = _renderer(OutputMode.JSON)
    clock = FakeClock()
    reporter = UploadProgressReporter(renderer, clock=clock, ticker=False)
    item = Item("node_zip", "local-node.zip", 300)

    # When
    reporter.plan([item], already_held=0)
    _upload_one_file(reporter, clock, item, [(3.0, 300)])
    renderer.emit({"uploaded": 1}, command="build push")

    # Then
    captured = capsys.readouterr()
    assert [line["type"] for line in _lines(captured.out)] == ["envelope"]
    events = _lines(captured.err)
    _validate(events)
    assert [event["type"] for event in events] == [
        "upload_plan",
        "upload_progress",
        "upload_progress",
        "upload_complete",
    ]


def test_a_file_the_builder_already_held_completes_without_progress() -> None:
    # Given
    stdout = io.StringIO()
    renderer = _renderer(OutputMode.NDJSON)
    renderer.machine_stream = stdout
    reporter = UploadProgressReporter(renderer, clock=FakeClock(), ticker=False)
    item = Item("model", "m.safetensors", 1000)

    # When
    reporter.plan([item], already_held=0)
    reporter.deduplicated(item, 1, 1)

    # Then
    events = _lines(stdout.getvalue())
    _validate(events)
    assert [event["type"] for event in events] == ["upload_plan", "upload_complete"]
    assert events[-1]["deduplicated"] is True
    assert events[-1]["seconds"] == 0
    assert events[-1]["bytes_per_second"] is None
    assert events[-1]["overall_bytes_done"] == 1000


def test_a_failed_upload_reports_no_completion() -> None:
    # Given
    stdout = io.StringIO()
    renderer = _renderer(OutputMode.NDJSON)
    renderer.machine_stream = stdout
    reporter = UploadProgressReporter(renderer, clock=FakeClock(), ticker=False)
    item = Item("model", "m.safetensors", 1000)

    # When
    with pytest.raises(requests.ConnectionError):
        with reporter.uploading(item, 1, 1) as progress:
            progress(10)
            raise requests.ConnectionError("connection reset mid-upload")

    # Then
    assert [event["type"] for event in _lines(stdout.getvalue())] == ["upload_progress"]


def test_a_reader_that_hangs_up_does_not_fail_the_upload() -> None:
    # Given a stream that refuses every write
    class _Closed(io.StringIO):
        def write(self, text: str) -> int:
            raise BrokenPipeError("reader went away")

    renderer = _renderer(OutputMode.NDJSON)
    renderer.machine_stream = _Closed()
    clock = FakeClock()
    reporter = UploadProgressReporter(renderer, clock=clock, ticker=False)
    item = Item("model", "m.safetensors", 100)

    # When / Then: nothing raises
    reporter.plan([item], already_held=0)
    _upload_one_file(reporter, clock, item, [(3.0, 100)])


def test_the_ticker_thread_reports_without_being_driven() -> None:
    """The byte callback goes quiet when a socket stalls, so the numbers have to
    come from somewhere that does not: a started ticker samples on its own."""
    # Given
    stdout = io.StringIO()
    renderer = _renderer(OutputMode.NDJSON)
    renderer.machine_stream = stdout
    reporter = UploadProgressReporter(renderer)
    item = Item("model", "m.safetensors", 100)

    # When
    with reporter.uploading(item, 1, 1) as progress:
        progress(100)

    # Then: the start and the completion, and the thread is gone
    assert [event["type"] for event in _lines(stdout.getvalue())] == ["upload_progress", "upload_complete"]


# ----- lines for a person whose output is piped -----


def test_piped_pretty_output_is_plain_lines_with_no_redraws(monkeypatch: pytest.MonkeyPatch) -> None:
    # Given pretty mode writing to something that is not a terminal
    for forced in ("FORCE_COLOR", "TTY_COMPATIBLE"):
        monkeypatch.delenv(forced, raising=False)
    out = io.StringIO()
    renderer = _renderer(OutputMode.PRETTY)
    renderer.pretty_stream = out
    clock = FakeClock()
    reporter = UploadProgressReporter(renderer, clock=clock, ticker=False)
    item = Item("model", "m.safetensors", 20 * 1024 * 1024)

    # When: twelve seconds at 1 MB/s, then the rest
    reporter.plan([item], already_held=1)
    _upload_one_file(reporter, clock, item, [(1.0, 1024 * 1024)] * 12 + [(1.0, 8 * 1024 * 1024)])

    # Then
    text = out.getvalue()
    assert "\r" not in text
    lines = text.splitlines()
    assert lines[0] == "1 file, 20.0 MB to upload, 1 already held"
    assert lines[1] == "1/1 m.safetensors: 0 B of 20.0 MB"
    # A line every five seconds, not every tick.
    assert lines[2] == "1/1 m.safetensors: 5.0 MB of 20.0 MB, 1.0 MB/s, 15s left"
    assert lines[3].startswith("1/1 m.safetensors: 10.0 MB of 20.0 MB")
    assert lines[-1].startswith("1/1 m.safetensors: uploaded 20.0 MB in 13s")
    assert len(lines) == 5


# ----- upload_assets drives the reporter -----


class _Reporter:
    def __init__(self) -> None:
        self.calls: list[tuple] = []

    def uploading(self, item, index, of):
        @contextmanager
        def block():
            self.calls.append(("start", item.filename, index, of))
            yield lambda n: self.calls.append(("bytes", item.filename, n))
            self.calls.append(("done", item.filename))

        return block()

    def deduplicated(self, item, index, of) -> None:
        self.calls.append(("deduplicated", item.filename, index, of))


class _Client:
    def __init__(self, held: set[str]) -> None:
        self.held = held

    def create_blob(self, kind: str, filename: str, sha256: str, size_bytes: int) -> tuple[str, str | None]:
        return f"blob-{filename}", None if filename in self.held else "https://blobs.test/put"

    def upload_blob(self, upload_url: str, path: Path, progress=None) -> None:
        assert progress is not None
        progress(path.stat().st_size)


def test_upload_assets_reports_each_file_in_order(tmp_path: Path) -> None:
    # Given two files to send, one of which the builder turns out to hold
    model = tmp_path / "m.safetensors"
    model.write_bytes(b"MODEL")
    archive = tmp_path / "node-0.zip"
    archive.write_bytes(b"ZIP")
    uploads = (
        PushUpload("model", "models", 0, "m.safetensors", "a" * 64, 5, model),
        PushUpload("node_zip", "customNodes", 0, "n.zip", "d" * 64, 3, archive),
    )
    spec = {"definition": {"models": [{"source": "local"}], "customNodes": [{"source": "local"}]}}
    reporter = _Reporter()

    # When
    transferred = upload_assets(PushPreparation(spec, uploads, ()), _Client(held={"n.zip"}), reporter=reporter)

    # Then
    assert transferred == 1
    assert reporter.calls == [
        ("start", "m.safetensors", 1, 2),
        ("bytes", "m.safetensors", 5),
        ("done", "m.safetensors"),
        ("deduplicated", "n.zip", 2, 2),
    ]


def test_already_held_counts_local_entries_that_need_no_upload(tmp_path: Path) -> None:
    # Given three local models: one pushed before, one with a public link, one new
    model = tmp_path / "new.safetensors"
    model.write_bytes(b"MODEL")
    uploads = (PushUpload("model", "models", 2, "new.safetensors", "a" * 64, 5, model),)
    spec = {
        "definition": {
            "models": [
                {"source": "local", "blobId": "blob-1"},
                {"source": "local", "sourceUri": "https://example.test/m"},
                {"source": "local"},
                {"source": "huggingface", "sourceUri": "https://example.test/public"},
            ],
            "customNodes": [],
        }
    }

    # When / Then
    assert already_held_count(PushPreparation(spec, uploads, ())) == 2
