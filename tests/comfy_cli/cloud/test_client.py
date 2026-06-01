"""Tests for the unified HTTP Client + Target abstraction.

Both local and cloud paths flow through the same ``Client``; the differences
are encoded as ``Target`` fields. These tests pin the contract.
"""

from __future__ import annotations

import io
import json
import urllib.error
from unittest.mock import patch

import pytest

from comfy_cli import comfy_client
from comfy_cli.target import Target


def _mock_response(payload):
    class _Resp:
        def __init__(self, body):
            self.body = body if isinstance(body, bytes) else json.dumps(body).encode()

        def read(self):
            return self.body

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

    return _Resp(payload)


def _http_error(status: int, body: bytes = b""):
    return urllib.error.HTTPError(
        url="https://cloud/x",
        code=status,
        msg=f"HTTP {status}",
        hdrs=None,
        fp=io.BytesIO(body),
    )


CLOUD = Target(
    kind="cloud",
    base_url="https://cloud.example.com",
    path_prefix="/api",
    history_path="history_v2",
    jobs_path="jobs",
    auth_token="tok-abc",
)

LOCAL = Target(
    kind="local",
    base_url="http://127.0.0.1:8188",
    path_prefix="",
    history_path="history",
    jobs_path=None,
    auth_token=None,
    host="127.0.0.1",
    port=8188,
)


class TestTargetURLs:
    def test_cloud_paths_get_api_prefix(self):
        assert CLOUD.url("prompt") == "https://cloud.example.com/api/prompt"
        assert CLOUD.url("history_v2", "abc") == "https://cloud.example.com/api/history_v2/abc"

    def test_local_paths_have_no_prefix(self):
        assert LOCAL.url("prompt") == "http://127.0.0.1:8188/prompt"
        assert LOCAL.url("history", "abc") == "http://127.0.0.1:8188/history/abc"


class TestSubmitPrompt:
    def test_posts_with_bearer_to_prefixed_url(self):
        with patch.object(
            comfy_client._OPENER,
            "open",
            return_value=_mock_response({"prompt_id": "pid-1", "number": 7, "node_errors": {}}),
        ) as urlopen:
            client = comfy_client.Client(CLOUD)
            result = client.submit_prompt({"1": {"class_type": "X", "inputs": {}}}, "cid")
        assert result.prompt_id == "pid-1"
        assert result.number == 7
        req = urlopen.call_args.args[0]
        assert req.full_url == "https://cloud.example.com/api/prompt"
        assert req.headers["Authorization"] == "Bearer tok-abc"
        body = json.loads(req.data)
        assert body == {
            "prompt": {"1": {"class_type": "X", "inputs": {}}},
            "client_id": "cid",
            "extra_data": {"auth_token_comfy_org": "tok-abc"},
        }

    def test_local_target_has_no_auth_header(self):
        with patch.object(
            comfy_client._OPENER,
            "open",
            return_value=_mock_response({"prompt_id": "pid-2", "number": 1, "node_errors": {}}),
        ) as urlopen:
            client = comfy_client.Client(LOCAL)
            client.submit_prompt({"1": {"class_type": "X", "inputs": {}}}, "cid")
        req = urlopen.call_args.args[0]
        assert req.full_url == "http://127.0.0.1:8188/prompt"
        assert "Authorization" not in req.headers
        body = json.loads(req.data)
        # Local submissions stay lean — no body-level token injection.
        assert body.keys() == {"prompt", "client_id"}

    def test_cloud_caller_extra_data_is_merged_not_overwritten(self):
        with patch.object(
            comfy_client._OPENER,
            "open",
            return_value=_mock_response({"prompt_id": "pid", "number": 1, "node_errors": {}}),
        ) as urlopen:
            client = comfy_client.Client(CLOUD)
            client.submit_prompt(
                {"1": {"class_type": "X", "inputs": {}}},
                "cid",
                extra_data={"pnginfo": {"workflow": "..."}},
            )
        body = json.loads(urlopen.call_args.args[0].data)
        # Caller-supplied keys preserved; cloud auth token added alongside.
        assert body["extra_data"]["pnginfo"] == {"workflow": "..."}
        assert body["extra_data"]["auth_token_comfy_org"] == "tok-abc"

    def test_cloud_caller_auth_token_is_not_clobbered(self):
        with patch.object(
            comfy_client._OPENER,
            "open",
            return_value=_mock_response({"prompt_id": "pid", "number": 1, "node_errors": {}}),
        ) as urlopen:
            client = comfy_client.Client(CLOUD)
            client.submit_prompt(
                {"1": {"class_type": "X", "inputs": {}}},
                "cid",
                extra_data={"auth_token_comfy_org": "caller-token"},
            )
        body = json.loads(urlopen.call_args.args[0].data)
        # setdefault — caller wins.
        assert body["extra_data"]["auth_token_comfy_org"] == "caller-token"

    def test_cloud_with_api_key_sends_x_api_key_header(self):
        cloud_apikey = Target(
            kind="cloud",
            base_url="https://cloud.example.com",
            path_prefix="/api",
            history_path="history_v2",
            jobs_path="jobs",
            api_key="sk-test-1234",
        )
        with patch.object(
            comfy_client._OPENER,
            "open",
            return_value=_mock_response({"prompt_id": "pid", "number": 1, "node_errors": {}}),
        ) as urlopen:
            client = comfy_client.Client(cloud_apikey)
            client.submit_prompt({"1": {"class_type": "X", "inputs": {}}}, "cid")
        req = urlopen.call_args.args[0]
        # X-API-Key header is set; Authorization Bearer is NOT.
        assert req.headers["X-api-key"] == "sk-test-1234"
        assert "Authorization" not in req.headers
        # Partner-API extra_data uses api_key_comfy_org for the key path.
        body = json.loads(req.data)
        assert body["extra_data"] == {"api_key_comfy_org": "sk-test-1234"}

    def test_cloud_api_key_wins_over_bearer_when_both_set(self):
        """If both are configured, API key wins (testing convenience)."""
        cloud_both = Target(
            kind="cloud",
            base_url="https://cloud.example.com",
            path_prefix="/api",
            history_path="history_v2",
            jobs_path="jobs",
            auth_token="bearer-token",
            api_key="api-key-1234",
        )
        with patch.object(
            comfy_client._OPENER,
            "open",
            return_value=_mock_response({"prompt_id": "pid", "number": 1, "node_errors": {}}),
        ) as urlopen:
            client = comfy_client.Client(cloud_both)
            client.submit_prompt({"1": {"class_type": "X", "inputs": {}}}, "cid")
        req = urlopen.call_args.args[0]
        assert req.headers["X-api-key"] == "api-key-1234"
        assert "Authorization" not in req.headers
        body = json.loads(req.data)
        assert "api_key_comfy_org" in body["extra_data"]
        assert "auth_token_comfy_org" not in body["extra_data"]

    def test_raises_http_error_on_4xx(self):
        with patch.object(comfy_client._OPENER, "open", side_effect=_http_error(400, b"bad workflow")):
            with pytest.raises(comfy_client.HTTPError) as exc:
                comfy_client.Client(CLOUD).submit_prompt({}, "cid")
        assert exc.value.status == 400
        assert "bad workflow" in exc.value.body

    def test_raises_on_missing_prompt_id_in_response(self):
        with patch.object(comfy_client._OPENER, "open", return_value=_mock_response({"number": 1})):
            with pytest.raises(comfy_client.HTTPError):
                comfy_client.Client(CLOUD).submit_prompt({}, "cid")


class TestUnauthenticated:
    def test_cloud_without_token_raises_at_construction(self):
        cloud_no_token = Target(
            kind="cloud",
            base_url="https://cloud.example.com",
            path_prefix="/api",
            history_path="history_v2",
            jobs_path="jobs",
            auth_token=None,
        )
        with pytest.raises(comfy_client.Unauthenticated):
            comfy_client.Client(cloud_no_token)


class TestGetHistory:
    def test_cloud_uses_history_v2_path(self):
        with patch.object(
            comfy_client._OPENER,
            "open",
            return_value=_mock_response({"pid-1": {"outputs": {"3": {"images": []}}, "status": {"completed": True}}}),
        ) as urlopen:
            rec = comfy_client.Client(CLOUD).get_history("pid-1")
        req = urlopen.call_args.args[0]
        assert req.full_url == "https://cloud.example.com/api/history_v2/pid-1"
        assert rec["status"]["completed"] is True

    def test_local_uses_history_path(self):
        with patch.object(
            comfy_client._OPENER,
            "open",
            return_value=_mock_response({"pid-1": {"outputs": {}, "status": {"completed": True}}}),
        ) as urlopen:
            comfy_client.Client(LOCAL).get_history("pid-1")
        req = urlopen.call_args.args[0]
        assert req.full_url == "http://127.0.0.1:8188/history/pid-1"

    def test_404_treated_as_transient_returns_none(self):
        with patch.object(comfy_client._OPENER, "open", side_effect=_http_error(404, b"not yet")):
            assert comfy_client.Client(CLOUD).get_history("pid") is None

    def test_returns_inner_when_flat(self):
        flat = {"outputs": {"3": {"images": []}}, "status": {"completed": False}}
        with patch.object(comfy_client._OPENER, "open", return_value=_mock_response(flat)):
            rec = comfy_client.Client(CLOUD).get_history("pid-1")
        assert rec["status"]["completed"] is False

    def test_returns_none_for_unrecognized_shape(self):
        with patch.object(comfy_client._OPENER, "open", return_value=_mock_response({"unrelated": 1})):
            assert comfy_client.Client(CLOUD).get_history("pid-1") is None


class TestListJobs:
    def test_cloud_hits_jobs_endpoint_with_limit(self):
        with patch.object(
            comfy_client._OPENER, "open", return_value=_mock_response({"jobs": [{"id": "a"}, {"id": "b"}]})
        ) as urlopen:
            jobs = comfy_client.Client(CLOUD).list_jobs(limit=5)
        req = urlopen.call_args.args[0]
        assert req.full_url == "https://cloud.example.com/api/jobs?limit=5"
        assert [j["id"] for j in jobs] == ["a", "b"]

    def test_local_raises_not_implemented(self):
        with pytest.raises(NotImplementedError):
            comfy_client.Client(LOCAL).list_jobs()


class TestGetJobStatus:
    def test_cloud_uses_job_status_endpoint(self):
        with patch.object(comfy_client._OPENER, "open", return_value=_mock_response({"status": "success"})) as urlopen:
            comfy_client.Client(CLOUD).get_job_status("pid-1")
        req = urlopen.call_args.args[0]
        assert req.full_url == "https://cloud.example.com/api/job/pid-1/status"

    def test_404_returns_none(self):
        with patch.object(comfy_client._OPENER, "open", side_effect=_http_error(404)):
            assert comfy_client.Client(CLOUD).get_job_status("pid") is None


class TestWaitForCompletion:
    def test_returns_record_when_status_completed_true(self):
        record = {"status": {"completed": True}, "outputs": {}}
        with patch.object(comfy_client.Client, "get_history", return_value=record):
            assert comfy_client.Client(CLOUD).wait_for_completion("pid", poll_interval=0) == record

    def test_treats_outputs_present_as_done(self):
        record = {"outputs": {"3": {"images": [{"filename": "out.png"}]}}}
        with patch.object(comfy_client.Client, "get_history", return_value=record):
            assert comfy_client.Client(CLOUD).wait_for_completion("pid", poll_interval=0) == record

    def test_raises_timeout(self):
        with patch.object(comfy_client.Client, "get_history", return_value=None):
            with pytest.raises(TimeoutError):
                comfy_client.Client(CLOUD).wait_for_completion("pid", poll_interval=0.01, timeout=0.05)


class TestTransientRetry:
    """A 429 / transient 5xx during polling must back off and retry, not abort
    the request — this is the bug that killed `comfy run --wait` mid-job."""

    def test_get_retries_on_429_then_succeeds(self):
        seq = [_http_error(429), _mock_response({"status": "success"})]
        with patch("comfy_cli.comfy_client.time.sleep"):
            with patch.object(comfy_client._OPENER, "open", side_effect=seq) as urlopen:
                result = comfy_client.Client(CLOUD).get_job_status("pid")
        assert result == {"status": "success"}
        assert urlopen.call_count == 2

    def test_persistent_429_eventually_raises(self):
        with patch("comfy_cli.comfy_client.time.sleep"):
            with patch.object(comfy_client._OPENER, "open", side_effect=_http_error(429)) as urlopen:
                with pytest.raises(comfy_client.HTTPError) as exc:
                    comfy_client.Client(CLOUD).get_job_status("pid")
        assert exc.value.status == 429
        assert urlopen.call_count == comfy_client._MAX_TRANSIENT_RETRIES + 1

    def test_submit_retries_on_429(self):
        # 429 means the request was rejected (not processed), so retrying a POST is safe.
        seq = [_http_error(429), _mock_response({"prompt_id": "pid-1", "node_errors": {}})]
        with patch("comfy_cli.comfy_client.time.sleep"):
            with patch.object(comfy_client._OPENER, "open", side_effect=seq) as urlopen:
                res = comfy_client.Client(CLOUD).submit_prompt({"1": {}}, "cid")
        assert res.prompt_id == "pid-1"
        assert urlopen.call_count == 2

    def test_5xx_retried_on_get(self):
        seq = [_http_error(503), _mock_response({"status": "success"})]
        with patch("comfy_cli.comfy_client.time.sleep"):
            with patch.object(comfy_client._OPENER, "open", side_effect=seq) as urlopen:
                assert comfy_client.Client(CLOUD).get_job_status("pid") == {"status": "success"}
        assert urlopen.call_count == 2

    def test_5xx_not_retried_on_post(self):
        # A 5xx on submit could have partially applied — must NOT auto-retry (double-execute risk).
        with patch("comfy_cli.comfy_client.time.sleep"):
            with patch.object(comfy_client._OPENER, "open", side_effect=_http_error(503)) as urlopen:
                with pytest.raises(comfy_client.HTTPError):
                    comfy_client.Client(CLOUD).submit_prompt({"1": {}}, "cid")
        assert urlopen.call_count == 1

    def test_honors_retry_after_header(self):
        from http.client import HTTPMessage

        hdrs = HTTPMessage()
        hdrs["Retry-After"] = "3"
        err = urllib.error.HTTPError(url="https://cloud/x", code=429, msg="429", hdrs=hdrs, fp=io.BytesIO(b""))
        seq = [err, _mock_response({"status": "success"})]
        with patch("comfy_cli.comfy_client.time.sleep") as sleep:
            with patch.object(comfy_client._OPENER, "open", side_effect=seq):
                comfy_client.Client(CLOUD).get_job_status("pid")
        assert sleep.call_args.args[0] == 3.0

    def test_wait_for_completion_survives_transient_429(self):
        done = {"status": {"completed": True}, "outputs": {}}
        seq = [_http_error(429), _mock_response(done)]
        with patch("comfy_cli.comfy_client.time.sleep"):
            with patch.object(comfy_client._OPENER, "open", side_effect=seq):
                assert comfy_client.Client(CLOUD).wait_for_completion("pid", poll_interval=0) == done


class TestOutputUrls:
    def test_view_url_uses_api_prefix_for_cloud(self):
        url = comfy_client.Client(CLOUD).view_url({"filename": "a.png", "subfolder": "", "type": "output"})
        assert url == "https://cloud.example.com/api/view?filename=a.png&subfolder=&type=output"

    def test_view_url_no_prefix_for_local(self):
        url = comfy_client.Client(LOCAL).view_url({"filename": "a.png", "subfolder": "", "type": "output"})
        assert url == "http://127.0.0.1:8188/view?filename=a.png&subfolder=&type=output"

    def test_extract_collects_image_urls(self):
        record = {
            "outputs": {
                "3": {
                    "images": [
                        {"filename": "a.png", "subfolder": "", "type": "output"},
                        {"filename": "b.png", "subfolder": "sub", "type": "temp"},
                    ]
                }
            }
        }
        urls = comfy_client.Client(CLOUD).extract_output_urls(record)
        assert urls == [
            "https://cloud.example.com/api/view?filename=a.png&subfolder=&type=output",
            "https://cloud.example.com/api/view?filename=b.png&subfolder=sub&type=temp",
        ]

    def test_extract_skips_malformed(self):
        assert comfy_client.Client(CLOUD).extract_output_urls({}) == []
        record = {"outputs": {"3": {"images": [{"no_filename": True}, "garbage"]}}}
        assert comfy_client.Client(CLOUD).extract_output_urls(record) == []


class TestRedirectRefusal:
    """The opener must refuse to follow redirects so the Bearer token can't
    be replayed at a different host."""

    def test_302_to_attacker_raises_http_error(self):
        # Build a 302 response that the redirect handler would normally follow.
        from http.client import HTTPMessage

        headers = HTTPMessage()
        headers["Location"] = "http://attacker.example/steal"
        err = urllib.error.HTTPError(
            url="https://cloud.example.com/api/prompt",
            code=302,
            msg="Found",
            hdrs=headers,
            fp=io.BytesIO(b""),
        )
        with patch.object(comfy_client._OPENER, "open", side_effect=err):
            with pytest.raises(comfy_client.HTTPError) as exc:
                comfy_client.Client(CLOUD).submit_prompt({}, "cid")
        assert exc.value.status == 302


class TestHttpUrlRejectedForCloud:
    def test_cloud_with_http_non_loopback_refused(self):
        bad = Target(
            kind="cloud",
            base_url="http://attacker.example",  # http, non-loopback, with token
            path_prefix="/api",
            history_path="history_v2",
            jobs_path="jobs",
            auth_token="tok",
        )
        client = comfy_client.Client(bad)
        with pytest.raises(ValueError, match="non-https"):
            client.submit_prompt({}, "cid")

    def test_cloud_with_http_loopback_allowed(self):
        local_cloud = Target(
            kind="cloud",
            base_url="http://127.0.0.1:8190",  # loopback exception
            path_prefix="/api",
            history_path="history_v2",
            jobs_path="jobs",
            auth_token="tok",
        )
        with patch.object(
            comfy_client._OPENER,
            "open",
            return_value=_mock_response({"prompt_id": "x", "number": 1, "node_errors": {}}),
        ):
            comfy_client.Client(local_cloud).submit_prompt({}, "cid")  # no raise


class TestTokenRedaction:
    def test_http_error_str_does_not_leak_bearer(self):
        body = b'{"error": "missing scope, header was Bearer abc123def456"}'
        err = urllib.error.HTTPError("https://x", 401, "Unauthorized", None, io.BytesIO(body))
        with patch.object(comfy_client._OPENER, "open", side_effect=err):
            with pytest.raises(comfy_client.HTTPError) as exc:
                comfy_client.Client(CLOUD).submit_prompt({}, "cid")
        assert "abc123def456" not in str(exc.value)
        assert "abc123def456" not in exc.value.body

    def test_target_repr_omits_token(self):
        # Bearer should never show in logger.debug("%r", target).
        assert "tok-abc" not in repr(CLOUD)
