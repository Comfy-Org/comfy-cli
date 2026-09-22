"""Progress for the blob uploads of ``comfy build push``.

A model upload is one streamed PUT that can run for an hour, and until it ends
the transport says nothing. This module turns the byte count the PUT exposes
into something a reader can act on: a plan line before the first byte, then
bytes sent, rate and time left while a file moves, then one line when it lands.

Three audiences, one set of numbers:

- a person at a terminal gets a single redrawn Rich progress line;
- a person whose output is piped gets a plain line every few seconds, with no
  carriage-return redraws to litter the file;
- an agent gets ``upload_plan`` / ``upload_progress`` / ``upload_complete``
  events through ``Renderer.progress_event``: on stdout under ``--json-stream``,
  on stderr under ``--json``, where stdout stays the single envelope.

The numbers are sampled by a ticker, not by the byte callback. A stalled socket
stops the callback, so a reporter driven by it would go quiet exactly when the
reader most needs to hear that the rate is falling. The ticker keeps sampling,
and the rate is measured over a sliding window, so a stall reads as a number
heading to zero rather than a frozen average.
"""

from __future__ import annotations

import threading
import time
from collections import deque
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from typing import Any, Final, Protocol

from comfy_cli.output.progress import Surface, human_bytes, human_seconds, surface_for
from comfy_cli.output.sanitize import sanitize_markup

_WINDOW_SECONDS: Final = 10.0
_TICK_SECONDS: Final = 1.0
# "Every few seconds": often enough that a reader sees movement within five
# seconds of the first byte, rare enough that an hour-long upload stays a
# readable transcript.
_EVENT_SECONDS: Final = 2.0
_LINE_SECONDS: Final = 5.0

EVENT_PLAN: Final = "upload_plan"
EVENT_PROGRESS: Final = "upload_progress"
EVENT_COMPLETE: Final = "upload_complete"


class Clock(Protocol):
    def __call__(self) -> float: ...


class UploadItem(Protocol):
    """The three facts about a planned upload this module reads."""

    @property
    def kind(self) -> str: ...

    @property
    def filename(self) -> str: ...

    @property
    def size_bytes(self) -> int: ...


def plan_line(files: int, total_bytes: int, already_held: int) -> str:
    """The one line printed before the first byte moves."""
    noun = "file" if files == 1 else "files"
    return f"{files} {noun}, {human_bytes(total_bytes)} to upload, {already_held} already held"


class SlidingRate:
    """Bytes per second over the last ``window`` seconds of samples."""

    def __init__(self, clock: Clock, window: float = _WINDOW_SECONDS) -> None:
        self._clock = clock
        self._window = window
        self._samples: deque[tuple[float, int]] = deque([(clock(), 0)])

    def sample(self, done: int) -> None:
        now = self._clock()
        self._samples.append((now, done))
        # Keep one sample at or beyond the window's far edge, so the span
        # measured is the window and not whatever happens to be left of it.
        while len(self._samples) > 2 and self._samples[1][0] <= now - self._window:
            self._samples.popleft()

    def bytes_per_second(self) -> float | None:
        first_at, first_done = self._samples[0]
        last_at, last_done = self._samples[-1]
        elapsed = last_at - first_at
        if elapsed <= 0:
            return None
        return (last_done - first_done) / elapsed


def _eta(remaining: int, rate: float | None) -> float | None:
    if remaining <= 0:
        return 0.0
    if rate is None or rate <= 0:
        return None
    return remaining / rate


class _Transfer:
    """One file in flight: what the byte callback adds to and the ticker reads."""

    def __init__(self, item: UploadItem, index: int, of: int, clock: Clock) -> None:
        self.item = item
        self.index = index
        self.of = of
        self.started = clock()
        # Only ever incremented by the uploading thread and read by the ticker.
        # An int rebinding is atomic under the GIL, so the hot path takes no lock.
        self.done = 0
        self.rate = SlidingRate(clock)
        self.last_report = self.started

    def add(self, n: int) -> None:
        self.done += n


class UploadProgressReporter:
    """Reports the plan, each file's progress and each file's completion.

    ``ticker=False`` leaves sampling to the caller's own ``tick()`` calls; tests
    use it with a fake ``clock`` to step time without a thread.
    """

    def __init__(self, renderer: Any, *, clock: Clock = time.monotonic, ticker: bool = True) -> None:
        self._renderer = renderer
        self._clock = clock
        self._ticker = ticker
        self._overall_total = 0
        self._overall_done = 0
        self._transfer: _Transfer | None = None
        self._live: Any = None
        self._live_task: Any = None
        self._lock = threading.Lock()
        # Progress is never worth failing an upload over: the first write the
        # stream refuses turns reporting off for the rest of the push.
        self._muted = False
        self._surface: Surface = surface_for(renderer)

    # ----- the plan -----

    def plan(self, uploads: tuple[UploadItem, ...] | list[UploadItem], *, already_held: int) -> None:
        self._overall_total = sum(upload.size_bytes for upload in uploads)
        self._overall_done = 0
        if self._surface == "events":
            self._event(EVENT_PLAN, files=len(uploads), bytes_total=self._overall_total, already_held=already_held)
        else:
            self._say(plan_line(len(uploads), self._overall_total, already_held))

    # ----- one file -----

    @contextmanager
    def uploading(self, item: UploadItem, index: int, of: int) -> Iterator[Callable[[int], None]]:
        """Report one file's transfer. Yields the callback to feed byte counts to.

        Leaving the block normally reports the file complete. Leaving it on an
        exception reports nothing: the command's error envelope is the next
        thing the reader sees, and a "complete" line before it would be a lie.
        """
        transfer = _Transfer(item, index, of, self._clock)
        with self._lock:
            self._transfer = transfer
            self._open_live(transfer)
            self._report(transfer)
        stop = threading.Event()
        thread = self._start_ticker(stop)
        completed = False
        try:
            yield transfer.add
            completed = True
        finally:
            stop.set()
            if thread is not None:
                thread.join()
            with self._lock:
                self._transfer = None
                self._close_live()
                if completed:
                    self._overall_done += item.size_bytes
                    self._complete(transfer, deduplicated=False)

    def deduplicated(self, item: UploadItem, index: int, of: int) -> None:
        """The builder already held these bytes, so nothing was sent."""
        with self._lock:
            self._overall_done += item.size_bytes
            self._complete(_Transfer(item, index, of, self._clock), deduplicated=True)

    def tick(self) -> None:
        """Sample the file in flight and report it if its interval has passed."""
        with self._lock:
            transfer = self._transfer
            if transfer is None:
                return
            transfer.rate.sample(transfer.done)
            now = self._clock()
            interval = {"events": _EVENT_SECONDS, "lines": _LINE_SECONDS, "live": 0.0}[self._surface]
            if now - transfer.last_report >= interval:
                transfer.last_report = now
                self._report(transfer)

    # ----- internals -----

    def _start_ticker(self, stop: threading.Event) -> threading.Thread | None:
        if not self._ticker:
            return None

        def run() -> None:
            while not stop.wait(_TICK_SECONDS):
                self.tick()

        thread = threading.Thread(target=run, name="comfy-upload-progress", daemon=True)
        thread.start()
        return thread

    def _numbers(self, transfer: _Transfer) -> dict[str, Any]:
        item = transfer.item
        done = min(transfer.done, item.size_bytes)
        rate = transfer.rate.bytes_per_second()
        overall_done = self._overall_done + done
        return {
            "file": item.filename,
            "kind": item.kind,
            "index": transfer.index,
            "of": transfer.of,
            "bytes_done": done,
            "bytes_total": item.size_bytes,
            "bytes_per_second": None if rate is None else round(rate),
            "eta_seconds": _rounded(_eta(item.size_bytes - done, rate)),
            "overall_bytes_done": overall_done,
            "overall_bytes_total": self._overall_total,
            "overall_eta_seconds": _rounded(_eta(self._overall_total - overall_done, rate)),
        }

    def _report(self, transfer: _Transfer) -> None:
        numbers = self._numbers(transfer)
        if self._surface == "events":
            self._event(EVENT_PROGRESS, **numbers)
        elif self._surface == "live":
            self._update_live(numbers)
        else:
            self._say(_progress_line(numbers))

    def _complete(self, transfer: _Transfer, *, deduplicated: bool) -> None:
        item = transfer.item
        seconds = 0.0 if deduplicated else max(self._clock() - transfer.started, 0.0)
        rate = round(item.size_bytes / seconds) if seconds > 0 else None
        if self._surface == "events":
            self._event(
                EVENT_COMPLETE,
                file=item.filename,
                kind=item.kind,
                index=transfer.index,
                of=transfer.of,
                bytes_total=item.size_bytes,
                seconds=round(seconds, 1),
                bytes_per_second=rate,
                deduplicated=deduplicated,
                overall_bytes_done=self._overall_done,
                overall_bytes_total=self._overall_total,
            )
            return
        label = f"{transfer.index}/{transfer.of} {item.filename}"
        if deduplicated:
            self._say(f"{label}: already held, nothing sent")
            return
        speed = f", {human_bytes(rate)}/s" if rate is not None else ""
        self._say(f"{label}: uploaded {human_bytes(item.size_bytes)} in {human_seconds(seconds)}{speed}")

    def _event(self, type: str, **fields: Any) -> None:
        if self._muted:
            return
        try:
            self._renderer.progress_event(type, **fields)
        except OSError:
            self._muted = True

    def _say(self, message: str) -> None:
        if self._muted:
            return
        try:
            self._renderer.info(message)
        except OSError:
            self._muted = True

    def _open_live(self, transfer: _Transfer) -> None:
        if self._surface != "live" or self._muted:
            return
        from rich.progress import BarColumn, DownloadColumn, Progress, TextColumn

        item = transfer.item
        # Rate and time left are text fields fed from this module's own sliding
        # window, not Rich's speed columns: those only recompute when the byte
        # count advances, so a stalled upload would freeze at its last good rate.
        live = Progress(
            TextColumn("{task.description}"),
            BarColumn(),
            DownloadColumn(binary_units=True),
            TextColumn("{task.fields[rate]}"),
            TextColumn("{task.fields[left]}"),
            console=self._renderer.console(),
            transient=True,
        )
        # Adding the task redraws the display, so it can refuse the stream just
        # as starting it can; both stay inside one boundary and nothing is
        # published until both have landed.
        try:
            live.start()
            task = live.add_task(
                sanitize_markup(f"{transfer.index}/{transfer.of} {item.filename}"),
                total=item.size_bytes,
                rate="",
                left="",
            )
        except OSError:
            self._muted = True
            try:
                live.stop()
            except OSError:
                pass
            return
        self._live = live
        self._live_task = task

    def _update_live(self, numbers: dict[str, Any]) -> None:
        if self._live is None:
            return
        rate = numbers["bytes_per_second"]
        left = numbers["overall_eta_seconds"]
        try:
            self._live.update(
                self._live_task,
                completed=numbers["bytes_done"],
                rate="" if rate is None else f"{human_bytes(rate)}/s",
                left="" if left is None else f"{human_seconds(left)} left",
            )
        except OSError:
            self._muted = True

    def _close_live(self) -> None:
        live, self._live, self._live_task = self._live, None, None
        if live is None:
            return
        try:
            live.stop()
        except OSError:
            self._muted = True


def _rounded(value: float | None) -> int | None:
    return None if value is None else round(value)


def _progress_line(numbers: dict[str, Any]) -> str:
    rate = numbers["bytes_per_second"]
    left = numbers["overall_eta_seconds"]
    parts = [
        f"{numbers['index']}/{numbers['of']} {numbers['file']}:",
        f"{human_bytes(numbers['bytes_done'])} of {human_bytes(numbers['bytes_total'])}",
    ]
    if rate is not None:
        parts.append(f"{human_bytes(rate)}/s")
    if left is not None:
        parts.append(f"{human_seconds(left)} left")
    return " ".join(parts[:2]) + "".join(f", {part}" for part in parts[2:])
