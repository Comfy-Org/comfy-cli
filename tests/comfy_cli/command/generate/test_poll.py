"""Tests for the BFL polling adapter."""

from unittest.mock import patch

import httpx

from comfy_cli.command.generate import poll


def _resp(body):
    return httpx.Response(200, json=body)


def test_poll_bfl_extracts_sample_url():
    responses = iter(
        [
            _resp({"id": "abc", "status": "Pending", "progress": 0.2}),
            _resp(
                {
                    "id": "abc",
                    "status": "Ready",
                    "progress": 1.0,
                    "result": {"sample": "https://cdn.example/result.png"},
                }
            ),
        ]
    )

    progress_seen: list[float] = []

    with (
        patch("comfy_cli.command.generate.client.get", side_effect=lambda *a, **kw: next(responses)),
        patch("comfy_cli.command.generate.poll._sleep", lambda *_: None),
    ):
        result = poll.poll_bfl(
            {"polling_url": "https://api.comfy.org/proxy/bfl/get_result?id=abc"},
            api_key="sk-test",
            on_progress=progress_seen.append,
        )

    assert result.status == "succeeded"
    assert result.image_urls == ["https://cdn.example/result.png"]
    assert progress_seen == [0.2, 1.0]


def test_poll_bfl_reports_failure():
    responses = iter([_resp({"id": "abc", "status": "Content Moderated", "progress": 0.0})])
    with (
        patch("comfy_cli.command.generate.client.get", side_effect=lambda *a, **kw: next(responses)),
        patch("comfy_cli.command.generate.poll._sleep", lambda *_: None),
    ):
        result = poll.poll_bfl(
            {"polling_url": "https://x"},
            api_key="sk-test",
        )
    assert result.status == "failed"
    assert "Content Moderated" in (result.error or "")
