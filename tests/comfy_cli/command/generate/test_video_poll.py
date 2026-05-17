"""Tests for the generic config-driven poller and per-partner specs."""

import httpx
import pytest

from comfy_cli.command.generate import poll


def _resp(body):
    return httpx.Response(200, json=body)


@pytest.fixture
def no_sleep(monkeypatch):
    monkeypatch.setattr(poll, "_sleep", lambda *_: None)


def _make_runner(get_responses):
    """Patch client.get with an iterator over fake responses."""
    it = iter(get_responses)
    return lambda *_a, **_kw: next(it)


def test_kling_sibling_poll_path(no_sleep, monkeypatch):
    """Kling builds the poll URL from {create_path}/{id}."""
    captured = {}

    def fake_get(url, **kw):
        captured["url"] = url
        return _resp({"data": {"task_status": "succeed", "task_result": {"videos": [{"url": "https://cdn/v.mp4"}]}}})

    monkeypatch.setattr("comfy_cli.command.generate.client.get", fake_get)
    poller = poll.get_poller("kling")
    result = poller(
        {"data": {"task_id": "abc"}},
        api_key="comfyui-test",
        create_path="/proxy/kling/v1/videos/text2video",
    )
    assert captured["url"] == "/proxy/kling/v1/videos/text2video/abc"
    assert result.status == "succeeded"
    assert result.image_urls == ["https://cdn/v.mp4"]


def test_luma_succeeds(no_sleep, monkeypatch):
    monkeypatch.setattr(
        "comfy_cli.command.generate.client.get",
        _make_runner(
            [
                _resp({"id": "luma-1", "state": "dreaming"}),
                _resp({"id": "luma-1", "state": "completed", "assets": {"video": "https://cdn/x.mp4"}}),
            ]
        ),
    )
    result = poll.get_poller("luma")({"id": "luma-1", "state": "queued"}, api_key="k")
    assert result.status == "succeeded"
    assert "https://cdn/x.mp4" in result.image_urls


def test_runway_progress_normalized(no_sleep, monkeypatch):
    """Runway reports progress as 0–1 floats — the poller forwards as-is."""
    seen: list[float] = []
    monkeypatch.setattr(
        "comfy_cli.command.generate.client.get",
        _make_runner(
            [
                _resp({"id": "x", "status": "RUNNING", "progress": 0.5}),
                _resp({"id": "x", "status": "SUCCEEDED", "output": ["https://cdn/v.mp4"]}),
            ]
        ),
    )
    poll.get_poller("runway")({"id": "x"}, api_key="k", on_progress=seen.append)
    assert seen == [0.5]


def test_runway_failure_states(no_sleep, monkeypatch):
    monkeypatch.setattr(
        "comfy_cli.command.generate.client.get",
        _make_runner([_resp({"id": "x", "status": "CANCELLED"})]),
    )
    result = poll.get_poller("runway")({"id": "x"}, api_key="k")
    assert result.status == "failed"
    assert "CANCELLED" in (result.error or "")


def test_minimax_redeems_file_id(no_sleep, monkeypatch):
    """After Success, minimax needs a second GET to /files/retrieve to get the download URL."""
    monkeypatch.setattr(
        "comfy_cli.command.generate.client.get",
        _make_runner(
            [
                _resp({"status": "Processing", "task_id": "t1"}),
                _resp({"status": "Success", "task_id": "t1", "file_id": "f-42"}),
                _resp({"file": {"download_url": "https://cdn/minimax.mp4"}}),
            ]
        ),
    )
    result = poll.get_poller("minimax")({"task_id": "t1"}, api_key="k")
    assert result.status == "succeeded"
    assert "https://cdn/minimax.mp4" in result.image_urls


def test_pika_polls_videos_endpoint(no_sleep, monkeypatch):
    captured = {}

    def fake_get(url, **kw):
        captured["url"] = url
        return _resp({"id": "v1", "status": "finished", "url": "https://cdn/p.mp4"})

    monkeypatch.setattr("comfy_cli.command.generate.client.get", fake_get)
    result = poll.get_poller("pika")({"video_id": "v1"}, api_key="k")
    assert captured["url"] == "/proxy/pika/videos/v1"
    assert result.status == "succeeded"


def test_vidu_polls_creations_path(no_sleep, monkeypatch):
    captured = {}

    def fake_get(url, **kw):
        captured["url"] = url
        return _resp({"state": "success", "creations": [{"url": "https://cdn/vidu.mp4"}]})

    monkeypatch.setattr("comfy_cli.command.generate.client.get", fake_get)
    poll.get_poller("vidu")({"task_id": "t1"}, api_key="k")
    assert captured["url"] == "/proxy/vidu/tasks/t1/creations"


def test_xai_video_polls_request_id(no_sleep, monkeypatch):
    captured = {}

    def fake_get(url, **kw):
        captured["url"] = url
        return _resp({"status": "done", "video": {"url": "https://cdn/x.mp4"}})

    monkeypatch.setattr("comfy_cli.command.generate.client.get", fake_get)
    poll.get_poller("xai_video")({"request_id": "req-1"}, api_key="k")
    assert captured["url"] == "/proxy/xai/v1/videos/req-1"


def test_moonvalley_polls_prompts(no_sleep, monkeypatch):
    captured = {}

    def fake_get(url, **kw):
        captured["url"] = url
        return _resp({"id": "p-1", "status": "completed", "output_url": "https://cdn/m.mp4"})

    monkeypatch.setattr("comfy_cli.command.generate.client.get", fake_get)
    poll.get_poller("moonvalley")({"id": "p-1"}, api_key="k")
    assert captured["url"] == "/proxy/moonvalley/prompts/p-1"


def test_missing_id_raises(monkeypatch):
    monkeypatch.setattr("comfy_cli.command.generate.client.get", lambda *a, **kw: _resp({}))
    with pytest.raises(Exception, match="missing id"):
        poll.get_poller("kling")({}, api_key="k", create_path="/x")


def test_kling_without_create_path_raises():
    with pytest.raises(Exception, match="create path"):
        poll.get_poller("kling")({"data": {"task_id": "abc"}}, api_key="k")


def test_build_synthetic_initial_for_each_partner():
    """Sanity-check the resume helper for every registered partner."""
    for name in ("kling", "luma", "minimax", "runway", "moonvalley", "pika", "vidu", "xai_video"):
        body = poll.build_synthetic_initial(name, "abc")
        assert poll.extract_job_id(name, body) == "abc", name


def test_build_synthetic_initial_for_bfl():
    body = poll.build_synthetic_initial("bfl", "abc", base_url="https://api.comfy.org")
    assert "polling_url" in body
    assert "abc" in body["polling_url"]


def test_extract_urls_recognizes_video_extensions():
    found = poll._extract_urls({"video_url": "https://cdn/x.mp4", "ignore": "https://cdn/notmedia"})
    assert "https://cdn/x.mp4" in found
    assert "https://cdn/notmedia" not in found


def test_extract_urls_recognizes_query_strings():
    """Signed URLs with ?Expires=… shouldn't be excluded by their query string."""
    found = poll._extract_urls({"url": "https://cdn/v.mp4?Expires=123&Signature=abc"})
    assert found == ["https://cdn/v.mp4?Expires=123&Signature=abc"]


def test_extract_job_id_from_nested_paths():
    assert poll.extract_job_id("kling", {"data": {"task_id": "k1"}}) == "k1"
    assert poll.extract_job_id("luma", {"id": "l1"}) == "l1"
    assert poll.extract_job_id("minimax", {"task_id": "m1"}) == "m1"
    assert poll.extract_job_id("xai_video", {"request_id": "x1"}) == "x1"


def test_existing_bfl_poller_still_works(no_sleep, monkeypatch):
    """Regression: the original BFL adapter shouldn't be disturbed by the refactor."""
    monkeypatch.setattr(
        "comfy_cli.command.generate.client.get",
        _make_runner([_resp({"status": "Ready", "result": {"sample": "https://cdn/b.png"}})]),
    )
    result = poll.get_poller("bfl")({"polling_url": "https://x"}, api_key="k")
    assert result.status == "succeeded"
    assert "https://cdn/b.png" in result.image_urls
