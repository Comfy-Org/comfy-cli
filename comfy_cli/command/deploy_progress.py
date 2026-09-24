"""Progress for a deployment that is coming up.

While a deployment's status is ``provisioning`` or ``starting`` the deploy
service carries a ``progress`` object on the deployment read: the step it is in
and, while models are copied onto its storage, models and bytes done against the
total, the model in flight, the rate and the time left. The service computes the
rate and the time left itself, so this module only words what it is given: the
CLI, the portal and an agent reading the events cannot disagree about a number.

Three audiences, the same split ``comfy build push`` makes for an upload:

- a person at a terminal gets a single redrawn Rich progress line;
- a person whose output is piped gets a plain line each time the service writes
  a new sample, with no carriage-return redraws to litter the file;
- an agent gets ``deploy_progress`` events through ``Renderer.progress_event``:
  on stdout under ``--json-stream``, on stderr under ``--json``, where stdout
  stays the single envelope. The ``progress`` object rides through unchanged.

A server that sends no ``progress`` (an older one, or any status but the two
above) produces nothing here, so those commands print what they always printed.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from datetime import datetime, timezone
from typing import Any, Final

from comfy_cli.command.build_spec import JsonObject
from comfy_cli.output.progress import Surface, human_bytes, human_seconds, surface_for
from comfy_cli.output.sanitize import sanitize_markup
from comfy_cli.utils import parse_rfc3339

EVENT_PROGRESS: Final = "deploy_progress"

# The service rewrites the object every few seconds in every step while the
# deploy is alive, the two that count nothing included, and its writes are
# best-effort. A minute of silence is a sample worth doubting; one or two
# dropped writes are ordinary. How long a step has run is `startedAt`'s to say:
# `updatedAt` only says how fresh the sample is.
STALE_SECONDS: Final = 60.0

STAGING_STEP: Final = "staging_models"

_STEP_LABELS: Final = {
    STAGING_STEP: "Staging models",
    "creating_endpoint": "Creating the endpoint",
    "waiting_for_worker": "Waiting for the first worker",
}

Now = Callable[[], datetime]


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


COMING_UP = frozenset({"provisioning", "starting"})


def progress_of(deployment: JsonObject) -> JsonObject | None:
    """The deployment's progress object, or None where it sent none.

    Lenient where the rest of the deploy parsing is strict: progress narrates a
    deploy and must never be what fails a command, so anything that is not an
    object naming a step reads as no progress at all. Only a deployment coming
    up has any: a settled one still carrying an object is off the contract.
    """
    if deployment.get("status") not in COMING_UP:
        return None
    progress = deployment.get("progress")
    if not isinstance(progress, dict) or not isinstance(progress.get("step"), str):
        return None
    return progress


def _sample_key(progress: JsonObject, *, by_stamp: bool) -> str:
    """What makes one sample the same as the last.

    By its stamp for an agent, which is told about every write. By what it says
    for a person reading piped lines: a step with nothing to count is re-stamped
    every few seconds, and a line per re-stamp would bury the ones that matter.
    """
    updated_at = progress.get("updatedAt")
    if by_stamp and isinstance(updated_at, str):
        return updated_at
    content = {key: value for key, value in progress.items() if key != "updatedAt"}
    return json.dumps(content, sort_keys=True, separators=(",", ":"), default=str)


def _number(progress: JsonObject, key: str) -> int | None:
    value = progress.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return int(value)


def step_label(progress: JsonObject) -> str:
    step = str(progress.get("step"))
    # A step this version has never heard of still names itself.
    return _STEP_LABELS.get(step, step.replace("_", " "))


def _seconds_since(progress: JsonObject, key: str, now: datetime) -> float | None:
    stamp = progress.get(key)
    if not isinstance(stamp, str):
        return None
    try:
        then = parse_rfc3339(stamp)
    except ValueError:
        return None
    return max((now - then).total_seconds(), 0.0)


def seconds_since_update(progress: JsonObject, now: datetime) -> float | None:
    return _seconds_since(progress, "updatedAt", now)


def is_stale(progress: JsonObject, now: datetime) -> bool:
    age = seconds_since_update(progress, now)
    return age is not None and age > STALE_SECONDS


def _model_part(progress: JsonObject) -> str | None:
    model = progress.get("currentModel")
    if not isinstance(model, str) or not model:
        return None
    name = model.rsplit("/", 1)[-1]
    done, total = _number(progress, "modelsDone"), _number(progress, "modelsTotal")
    if done is None or not total:
        return name
    return f"model {min(done + 1, total)} of {total} {name}"


def _bytes_part(progress: JsonObject) -> str | None:
    done, total = _number(progress, "bytesDone"), _number(progress, "bytesTotal")
    if done is None:
        return None
    if total is None:
        return f"{human_bytes(done)} copied"
    of = "of at least" if progress.get("bytesTotalIsFloor") is True else "of"
    return f"{human_bytes(done)} {of} {human_bytes(total)}"


def _parts(progress: JsonObject, now: datetime) -> tuple[str, str | None, list[str]]:
    """The step's name, the model it is on, and the numbers, kept apart.

    The sentence and the live bar want the same facts in different shapes: one
    joins them with commas, the other spreads them across columns and has to
    know which piece may be truncated when the terminal is narrow.
    """
    label = step_label(progress)
    model: str | None = None
    parts: list[str] = []
    if progress.get("step") != STAGING_STEP:
        # Nothing here is measured, so the only honest number is how long the
        # step has been running. Without it the line never changes and a wait
        # that is working reads exactly like one that has died.
        waited = _seconds_since(progress, "startedAt", now)
        if waited is not None and waited >= 1:
            parts.append(f"{human_seconds(waited)} so far")
    else:
        if _number(progress, "bytesTotal") == 0:
            parts.append("every model is already in place")
        else:
            model = _model_part(progress)
            rate, left = _number(progress, "bytesPerSecond"), _number(progress, "etaSeconds")
            parts.extend(
                part
                for part in (
                    _bytes_part(progress),
                    None if not rate else f"{human_bytes(rate)}/s",
                    None if left is None else f"{human_seconds(left)} left",
                )
                if part is not None
            )
    return label, model, parts


def _notes(progress: JsonObject, now: datetime, *, short: bool = False) -> list[str]:
    """What a reader must know about the sample itself: a restart, an old read."""
    notes = []
    attempt = _number(progress, "attempt")
    if attempt is not None and attempt > 1:
        notes.append(f"attempt {attempt}" if short else f"attempt {attempt}, this step was restarted")
    # The same test the events' `stale` flag uses, so a line and an event
    # read at the same moment never disagree about the sample.
    age = seconds_since_update(progress, now)
    if age is not None and is_stale(progress, now):
        waited = human_seconds(age)
        notes.append(f"no update for {waited}" if short else f"last update {waited} ago, so these numbers may be stale")
    return notes


def describe(progress: JsonObject, *, now: datetime) -> str:
    """One line saying where the deployment is, from one progress object."""
    label, model, parts = _parts(progress, now)
    if model is not None:
        parts.insert(0, model)
    line = label if not parts else f"{label}: {', '.join(parts)}"
    return line + "".join(f" ({note})" for note in _notes(progress, now))


def live_line(progress: JsonObject, *, now: datetime) -> str:
    """The same facts for the redrawn line, ordered by what may be cut.

    The terminal crops from the right. The notes ride on the label in their short
    form, because a restart or a silent service changes how every number after
    them reads. The model's name goes last: it is the one piece long enough to
    need cutting and the only one a reader can lose without losing a number.
    """
    label, model, parts = _parts(progress, now)
    notes = _notes(progress, now, short=True)
    if notes:
        label = f"{label} ({', '.join(notes)})"
    if model is not None:
        parts.append(model)
    return label if not parts else f"{label}: {', '.join(parts)}"


def reattach_hint(deployment_id: str) -> str:
    return f"comfy deploy status --deployment {deployment_id} --watch"


class DeployWatchReporter:
    """Reports each new sample of a watched deployment's progress.

    Fed every snapshot the poll reads. The poll runs every two seconds and the
    service writes about every three, so a sample is reported once, when it
    changes, and once more if it then goes stale: a reader is never shown
    movement that did not happen.
    """

    def __init__(self, renderer: Any, deployment_id: str, *, now: Now = _utcnow) -> None:
        self._renderer = renderer
        self._deployment_id = deployment_id
        self._now = now
        self._reported: tuple[str, str, bool] | None = None
        self._live: Any = None
        self._live_task: Any = None
        # The last snapshot read, for the envelope an interrupted watch still owes.
        self.last: JsonObject | None = None
        # Progress is never worth failing a deploy command over: the first write
        # the stream refuses turns reporting off for the rest of the watch.
        self._muted = False
        self._surface: Surface = surface_for(renderer)

    def snapshot(self, deployment: JsonObject) -> None:
        self.last = deployment
        progress = progress_of(deployment)
        if progress is None:
            self.close()
            return
        now = self._now()
        stale = is_stale(progress, now)
        if self._surface == "live":
            self._update_live(progress, now)
            return
        # Keyed on the sample, not on the wording: the stale suffix and the
        # time so far count seconds, and a line per poll is what this exists to
        # avoid. A sample with no stamp is keyed on its content, so a number that
        # moved is still reported and a repeat still is not.
        by_stamp = self._surface == "events"
        key = (str(deployment.get("status")), _sample_key(progress, by_stamp=by_stamp), stale)
        if key == self._reported:
            return
        self._reported = key
        if self._surface == "events":
            self._event(deployment, progress, stale)
        else:
            self._say(describe(progress, now=now))

    def close(self) -> None:
        live, self._live, self._live_task = self._live, None, None
        if live is None:
            return
        try:
            live.stop()
        except OSError:
            self._muted = True

    def interrupted(self) -> None:
        """Say what Ctrl-C did not do: the deploy runs on the service's side."""
        self.close()
        self._say(
            f"Stopped watching. Deployment {self._deployment_id} keeps coming up on our side.",
            hint=f"run `{reattach_hint(self._deployment_id)}` to watch it again",
        )

    # ----- internals -----

    def _event(self, deployment: JsonObject, progress: JsonObject, stale: bool) -> None:
        if self._muted:
            return
        try:
            self._renderer.progress_event(
                EVENT_PROGRESS,
                deployment_id=self._deployment_id,
                status=deployment.get("status"),
                stale=stale,
                progress=progress,
            )
        except OSError:
            self._muted = True

    def _say(self, message: str, *, hint: str | None = None) -> None:
        if self._muted:
            return
        try:
            self._renderer.info(message, hint=hint)
        except OSError:
            self._muted = True

    def _open_live(self) -> None:
        from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn
        from rich.table import Column

        # The spinner turns on Rich's own clock, not on the service's writes, so
        # a step that counts nothing still shows the command is alive.
        # One text column holding the whole sentence, after a short fixed bar.
        # Spread over several columns, Rich's table gave the narrow terminal's
        # shortfall to whichever column it chose: at 80 columns it cut the label
        # and the time left and wrapped the bar onto a line of its own. A single
        # column that never wraps is cropped from its right end, and
        # `live_line` puts the model's name there.
        live = Progress(
            SpinnerColumn(),
            BarColumn(bar_width=10),
            TextColumn("{task.fields[line]}", table_column=Column(no_wrap=True, overflow="ellipsis", ratio=1)),
            console=self._renderer.console(),
            transient=True,
            expand=True,
        )
        # Adding the task redraws the display, so it can refuse the stream just
        # as starting it can; both stay inside one boundary and nothing is
        # published until both have landed.
        try:
            live.start()
            task = live.add_task("", total=None, line="")
        except OSError:
            self._muted = True
            try:
                live.stop()
            except OSError:
                pass
            return
        self._live = live
        self._live_task = task

    def _update_live(self, progress: JsonObject, now: datetime) -> None:
        if self._muted:
            return
        if self._live is None:
            self._open_live()
            if self._live is None:
                return
        total = _number(progress, "bytesTotal") if progress.get("step") == STAGING_STEP else None
        done = _number(progress, "bytesDone") or 0
        try:
            # No total means no fraction to draw, and Rich pulses the bar instead:
            # the step is moving, and how far along it is was never measured.
            self._live.update(
                self._live_task,
                total=total if total else None,
                completed=min(done, total) if total else 0,
                line=sanitize_markup(live_line(progress, now=now)),
            )
        except OSError:
            self._muted = True
