"""The ``<os>/<gpu>`` build-target vocabulary shared by every release-cutting command.

A target belongs to a *release*, never to a push (build design lines 253-279):
an implicit target spends build minutes the caller never asked for, so nothing
here ever substitutes a default. ``--target os/gpu`` is the single spelling, and
a value that is not exactly two non-empty segments is refused rather than
guessed at.

The *vocabulary* is deliberately not enumerated here. ``GET /v1/build-targets``
is the builder's own allow-list — the same one it validates a cut against — and
its OpenAPI note is explicit that the wire enums are wider than the buildable
list, so a client that hardcoded either would offer targets the builder rejects
at cut time. This module therefore owns the *shape* and lets the builder own the
values; ``catalog_choices`` turns that endpoint's response into picker choices.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Final

__all__ = [
    "TARGET_FORM",
    "BuildTarget",
    "BuildTargetInvalidError",
    "catalog_choices",
    "parse_build_targets",
]

#: How the option is spelled in help text, hints and errors, in one place.
TARGET_FORM: Final = "<os>/<gpu>"

#: A concrete pair, so an error never leaves the reader guessing at the shape.
TARGET_EXAMPLE: Final = "linux/nvidia"


class BuildTargetInvalidError(ValueError):
    """One or more ``--target`` values that are not exactly ``<os>/<gpu>``.

    Carries every rejected value rather than only the first, so a caller fixes
    them all in one retry — the same reasoning as ``require_option``'s
    ``missing`` list.
    """

    def __init__(self, values: Sequence[str]) -> None:
        self.values: Final = tuple(values)
        subject = "value" if len(self.values) == 1 else "values"
        joined = ", ".join(repr(value) for value in self.values)
        super().__init__(f"--target {subject} {joined} must be of the form {TARGET_FORM}, for example {TARGET_EXAMPLE}")


@dataclass(frozen=True, slots=True)
class BuildTarget:
    """One parsed target pair."""

    os: str
    gpu: str

    def as_wire(self) -> dict[str, str]:
        """The builder's ``Target`` shape (``accelVariant`` is server-derived)."""
        return {"os": self.os, "gpu": self.gpu}

    def spelling(self) -> str:
        """The value a caller would type back into ``--target``."""
        return f"{self.os}/{self.gpu}"


def _parse_one(value: str) -> BuildTarget | None:
    segments = [segment.strip() for segment in value.split("/")]
    if len(segments) != 2 or not all(segments):
        return None
    os_name, gpu = segments
    return BuildTarget(os_name, gpu)


def parse_build_targets(values: Sequence[str]) -> list[BuildTarget]:
    """Parse every ``--target`` value, in call order.

    Raises:
        BuildTargetInvalidError: naming *every* value that is not exactly two
            non-empty ``/``-separated segments.
    """
    targets: list[BuildTarget] = []
    invalid: list[str] = []
    for value in values:
        parsed = _parse_one(value)
        if parsed is None:
            invalid.append(value)
        else:
            targets.append(parsed)
    if invalid:
        raise BuildTargetInvalidError(invalid)
    return targets


def catalog_choices(options: Sequence[Mapping[str, object]]) -> list[str]:
    """``<os>/<gpu>`` spellings from a ``GET /v1/build-targets`` response, in menu order.

    Each option is a ``BuildTargetOption`` — ``{target: {os, gpu}, label, ...}``
    — and only the pair is a choice; the rest is display metadata the builder
    ignores on a request. Anything not shaped like a pair is skipped rather than
    raised on: the picker is an aid, and a catalog the CLI cannot read fully
    should still offer what it can.
    """
    choices: list[str] = []
    for option in options:
        target = option.get("target")
        if not isinstance(target, Mapping):
            continue
        os_name = target.get("os")
        gpu = target.get("gpu")
        if not isinstance(os_name, str) or not isinstance(gpu, str) or not os_name or not gpu:
            continue
        spelling = f"{os_name}/{gpu}"
        if spelling not in choices:
            choices.append(spelling)
    return choices
