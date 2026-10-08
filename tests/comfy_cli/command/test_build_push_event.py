"""The one event a push sends about its uploads: how much, how long, how it
ended. Nothing measured an upload before it, in the CLI or the portal."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from comfy_cli.command import build
from comfy_cli.command.build_push import PushUpload


def _upload(name: str, size: int) -> PushUpload:
    return PushUpload(
        kind="model", filename=name, sha256="0" * 64, size_bytes=size, path=Path(name), collection="models", index=0
    )


@pytest.fixture
def sent():
    with patch.object(build.tracking, "track_event") as track:
        yield track


def test_a_finished_push_reports_bytes_seconds_and_what_was_already_stored(sent) -> None:
    build._track_push_upload([_upload("a.safetensors", 10), _upload("b.safetensors", 5)], uploaded=1, seconds=2.5)

    sent.assert_called_once()
    (event,) = sent.call_args.args
    props = sent.call_args.kwargs["properties"]
    assert event == "build:push_upload"
    assert props == {
        "upload_count": 2,
        "upload_bytes": 15,
        "seconds": 2.5,
        "outcome": "ok",
        "uploaded": 1,
        "deduped": 1,
    }
    assert "a.safetensors" not in str(props), "a private model's name is the customer's"


def test_a_push_that_died_mid_upload_still_reports_and_does_not_guess_the_split(sent) -> None:
    build._track_push_upload([_upload("a.safetensors", 10)], uploaded=None, seconds=0.25)

    props = sent.call_args.kwargs["properties"]
    assert props["outcome"] == "error"
    assert "uploaded" not in props and "deduped" not in props


def test_a_push_with_nothing_to_upload_sends_nothing(sent) -> None:
    build._track_push_upload([], uploaded=0, seconds=0.0)
    sent.assert_not_called()
