"""The wording every long wait in the CLI shares.

`build push` and `deploy up` narrate different things — bytes this process is
writing, and bytes a deployment is pulling — but a person meets both in the same
terminal minutes apart, so a size, a duration and the choice of surface have to
read the same. These are the tests that stop the two drifting.
"""

from __future__ import annotations

import pytest

from comfy_cli.output.progress import human_bytes, human_seconds, surface_for


@pytest.mark.parametrize(
    ("n", "expected"),
    [
        (0, "0 B"),
        (999, "999 B"),
        # 1024s, not 1000s: the portal's formatBytes counts the same way, and the
        # same deployment is read on both surfaces.
        (1024, "1.0 KB"),
        (7_272_719_498, "6.8 GB"),
        (2 << 40, "2.0 TB"),
    ],
)
def test_human_bytes(n, expected) -> None:
    assert human_bytes(n) == expected


@pytest.mark.parametrize(
    ("seconds", "expected"),
    [
        (0, "0s"),
        # Rounded, not truncated, so the last whole second of a wait is not "0s".
        (0.6, "1s"),
        (59, "59s"),
        (60, "1m 00s"),
        (200, "3m 20s"),
        (3900, "1h 05m"),
    ],
)
def test_human_seconds(seconds, expected) -> None:
    assert human_seconds(seconds) == expected


class _Console:
    def __init__(self, is_terminal: bool) -> None:
        self.is_terminal = is_terminal


class _Renderer:
    def __init__(self, *, pretty: bool, terminal: bool) -> None:
        self._pretty, self._console = pretty, _Console(terminal)

    def is_pretty(self) -> bool:
        return self._pretty

    def console(self) -> _Console:
        return self._console


def test_a_machine_mode_beats_a_terminal() -> None:
    """An agent reading captured output must never be handed cursor moves.

    The machine check comes first on purpose: a renderer can be in JSON mode
    while stdout is still a tty (`comfy --json` typed by hand), and drawing a
    redrawn line there would put escape sequences in the stream being parsed.
    """
    assert surface_for(_Renderer(pretty=False, terminal=True)) == "events"
    assert surface_for(_Renderer(pretty=False, terminal=False)) == "events"


def test_a_person_gets_the_redrawn_line_only_on_a_real_terminal() -> None:
    assert surface_for(_Renderer(pretty=True, terminal=True)) == "live"
    # Captured: plain lines, so nothing writes carriage returns into a log file.
    assert surface_for(_Renderer(pretty=True, terminal=False)) == "lines"
