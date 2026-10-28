"""`comfy download` on a job with nothing to fetch must say whether the job is
still running or already finished. A finished job with no outputs used to get
"has no outputs yet — wait for the job to complete", which sent the cloud agent
into re-runs and led it to tell a user the generation service was down."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest
import typer

from comfy_cli import jobs_state
from comfy_cli.command import transfer
from comfy_cli.output import Renderer, set_renderer
from comfy_cli.output.renderer import OutputMode, reset_renderer_for_testing
from comfy_cli.target import Target

PROMPT_ID = "prompt-empty-6d34c8d4"


@pytest.fixture(autouse=True)
def reset_renderer():
    reset_renderer_for_testing()
    yield
    reset_renderer_for_testing()


@pytest.fixture
def fake_target():
    return Target(
        kind="cloud",
        base_url="https://cloud.example.com",
        path_prefix="/api",
        history_path="history_v2",
        jobs_path="jobs",
        api_key="test-api-key",
    )


def _write_state(target: Target, status: str) -> None:
    state = jobs_state.JobState(
        prompt_id=PROMPT_ID,
        client_id=None,
        workflow="/abs/voice.json",
        where="cloud",
        base_url=target.base_url,
        status=status,
        outputs=[],
        record=None,
        item_map=None,
    )
    assert jobs_state.write(state) is not None


def _download_error(fake_target, capsys, history_record) -> dict:
    set_renderer(Renderer(mode=OutputMode.JSON, command="download"))
    client = MagicMock()
    client.get_history.return_value = history_record
    client.extract_output_urls.return_value = []
    with (
        patch("comfy_cli.command.transfer.resolve_target", return_value=fake_target),
        patch("comfy_cli.command.transfer.Client", return_value=client),
        pytest.raises(typer.Exit) as excinfo,
    ):
        transfer.execute_download(PROMPT_ID)
    assert excinfo.value.exit_code == 1
    lines = [ln for ln in capsys.readouterr().out.splitlines() if ln.strip()]
    envelope = json.loads(lines[-1])
    assert envelope["ok"] is False
    return envelope["error"]


def test_completed_job_with_no_outputs_does_not_say_wait(fake_target, capsys):
    _write_state(fake_target, "completed")
    err = _download_error(fake_target, capsys, {"status": {"completed": True, "status_str": "success"}, "outputs": {}})
    assert err["code"] == "download_no_outputs"
    assert "completed but produced no output files" in err["message"]
    assert "wait for the job" not in err["hint"]
    assert err["hint"].startswith("waiting will not produce files")
    assert err["details"]["job_status"] == "completed"


def test_history_record_alone_marks_the_job_finished(fake_target, capsys):
    err = _download_error(fake_target, capsys, {"execution_status": "success", "outputs": {}})
    assert "completed but produced no output files" in err["message"]


def test_fetched_record_wins_over_a_stale_state_file(fake_target, capsys):
    _write_state(fake_target, "completed")
    err = _download_error(fake_target, capsys, {"status": {"status_str": "error"}, "outputs": {}})
    assert "ended without completing" in err["message"]
    assert err["details"]["job_status"] == "failed"


def test_failed_job_does_not_say_wait(fake_target, capsys):
    _write_state(fake_target, "error")
    err = _download_error(fake_target, capsys, None)
    assert "ended without completing" in err["message"]
    assert "wait" not in err["hint"]


def test_text_only_job_says_there_is_nothing_to_download(fake_target, capsys):
    _write_state(fake_target, "completed")
    record = {"status": {"completed": True, "status_str": "success"}, "outputs": {"7": {"text": ["a caption"]}}}
    err = _download_error(fake_target, capsys, record)
    assert "text output only" in err["message"]
    assert "comfy jobs status" in err["hint"]


def test_running_job_still_says_wait(fake_target, capsys):
    _write_state(fake_target, "running")
    err = _download_error(fake_target, capsys, {"status": {"completed": False}, "outputs": {}})
    assert err["code"] == "download_no_outputs"
    assert err["message"].endswith("has no outputs yet")
    assert err["hint"] == "wait for the job to complete before downloading"
    assert "job_status" not in err["details"]
