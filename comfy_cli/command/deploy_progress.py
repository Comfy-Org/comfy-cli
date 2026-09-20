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

from collections.abc import Callable
from datetime import datetime, timezone
from typing import Any, Final, Literal

from comfy_cli.command.build_spec import JsonObject
from comfy_cli.output.sanitize import sanitize_markup

EVENT_PROGRESS: Final = "deploy_progress"

# The service rewrites the object every ten seconds while models stage, and its
# writes are best-effort. Six missed in a row is a sample worth doubting; one or
# two is an ordinary dropped write.
STALE_SECONDS: Final = 60.0

_STEP_LABELS: Final = {
    "staging_models": "Staging models",
    "creating_endpoint": "Creating the endpoint",
    "waiting_for_worker": "Waiting for the first worker",
}

Surface = Literal["events", "live", "lines"]
Now = Callable[[], datetime]


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def progress_of(deployment: JsonObject) -> JsonObject | None:
    """The deployment's progress object, or None where it sent none.

    Lenient where the rest of the deploy parsing is strict: progress narrates a
    deploy and must never be what fails a command, so anything that is not an
    object naming a step reads as no progress at all.
    """
    progress = deployment.get("progress")
    if not isinstance(progress, dict) or not isinstance(progress.get("step"), str):
        return None
    return progress


def _number(progress: JsonObject, key: str) -> int | None:
    value = progress.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return int(value)


def human_bytes(n: float) -> str:
    size = float(n)
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024:
            return f"{size:.0f} {unit}" if unit == "B" else f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} TB"


def human_seconds(seconds: float) -> str:
    whole = int(seconds + 0.5)
    if whole < 60:
        return f"{whole}s"
    if whole < 3600:
        return f"{whole // 60}m {whole % 60:02d}s"
    hours, rest = divmod(whole, 3600)
    return f"{hours}h {rest // 60:02d}m"


def step_label(progress: JsonObject) -> str:
    step = str(progress.get("step"))
    # A step this version has never heard of still names itself.
    return _STEP_LABELS.get(step, step.replace("_", " "))


def seconds_since_update(progress: JsonObject, now: datetime) -> float | None:
    stamp = progress.get("updatedAt")
    if not isinstance(stamp, str):
        return None
    try:
        updated = datetime.fromisoformat(stamp.replace("Z", "+00:00"))
    except ValueError:
        return None
    if updated.tzinfo is None:
        updated = updated.replace(tzinfo=timezone.utc)
    return max((now - updated).total_seconds(), 0.0)


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


def describe(progress: JsonObject, *, now: datetime) -> str:
    """One line saying where the deployment is, from one progress object."""
    label = step_label(progress)
    parts: list[str] = []
    if progress.get("step") == "staging_models":
        if _number(progress, "bytesTotal") == 0:
            parts.append("every model is already in place")
        else:
            rate, left = _number(progress, "bytesPerSecond"), _number(progress, "etaSeconds")
            parts.extend(
                part
                for part in (
                    _model_part(progress),
                    _bytes_part(progress),
                    None if not rate else f"{human_bytes(rate)}/s",
                    None if left is None else f"{human_seconds(left)} left",
                )
                if part is not None
            )
    line = label if not parts else f"{label}: {', '.join(parts)}"
    attempt = _number(progress, "attempt")
    if attempt is not None and attempt > 1:
        line += f" (attempt {attempt}, this step was restarted)"
    age = seconds_since_update(progress, now)
    if age is not None and age > STALE_SECONDS:
        line += f" (last update {human_seconds(age)} ago, so these numbers may be stale)"
    return line


def reattach_hint(deployment_id: str) -> str:
    return f"comfy deploy status --deployment {deployment_id} --watch"


class DeployWatchReporter:
    """Reports each new sample of a watched deployment's progress.

    Fed every snapshot the poll reads. The poll runs every two seconds and the
    service writes every ten, so a sample is reported once, when it changes, and
    once more if it then goes stale: a reader is never shown movement that did
    not happen.
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
        if not renderer.is_pretty():
            self._surface: Surface = "events"
        elif renderer.console().is_terminal:
            self._surface = "live"
        else:
            self._surface = "lines"

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
        # Keyed on the service's own stamp, not on the wording: the stale suffix
        # counts seconds, and a line per poll is what this exists to avoid.
        key = (str(deployment.get("status")), str(progress.get("updatedAt")), stale)
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
        from rich.progress import BarColumn, Progress, TextColumn

        live = Progress(
            TextColumn("{task.description}"),
            BarColumn(),
            TextColumn("{task.fields[detail]}"),
            console=self._renderer.console(),
            transient=True,
        )
        try:
            live.start()
        except OSError:
            self._muted = True
            return
        self._live = live
        self._live_task = live.add_task("", total=None, detail="")

    def _update_live(self, progress: JsonObject, now: datetime) -> None:
        if self._muted:
            return
        if self._live is None:
            self._open_live()
            if self._live is None:
                return
        line = describe(progress, now=now)
        label, _, detail = line.partition(": ")
        total = _number(progress, "bytesTotal") if progress.get("step") == "staging_models" else None
        done = _number(progress, "bytesDone") or 0
        try:
            # No total means no fraction to draw, and Rich pulses the bar instead:
            # the step is moving, and how far along it is was never measured.
            self._live.update(
                self._live_task,
                description=sanitize_markup(label),
                total=total if total else None,
                completed=min(done, total) if total else 0,
                detail=sanitize_markup(detail),
            )
        except OSError:
            self._muted = True
