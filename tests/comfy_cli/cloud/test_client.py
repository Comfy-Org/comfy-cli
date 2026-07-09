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
            "extra_data": {"auth_token_comfy_org": "tok-abc", "comfy_usage_source": "comfy-cli"},
        }
        # Usage-source attribution header on every request (#468).
        assert req.headers["Comfy-usage-source"] == "comfy-cli"

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
        # Local submissions get no body-level token injection — only the
        # usage-source attribution rides extra_data (#468).
        assert body.keys() == {"prompt", "client_id", "extra_data"}
        assert body["extra_data"] == {"comfy_usage_source": "comfy-cli"}

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

    def test_oauth_refresh_rebuilds_partner_auth_extra_data(self):
        cloud = Target(
            kind="cloud",
            base_url="https://cloud.example.com",
            path_prefix="/api",
            history_path="history_v2",
            jobs_path="jobs",
            auth_token="expired-token",
        )
        client = comfy_client.Client(cloud)

        def refresh():
            object.__setattr__(cloud, "auth_token", "fresh-token")
            return True

        seq = [_http_error(401), _mock_response({"prompt_id": "pid", "number": 1, "node_errors": {}})]
        with patch.object(client, "_try_refresh_token", side_effect=refresh):
            with patch.object(comfy_client._OPENER, "open", side_effect=seq) as urlopen:
                client.submit_prompt({"1": {"class_type": "X", "inputs": {}}}, "cid")

        first_req = urlopen.call_args_list[0].args[0]
        retry_req = urlopen.call_args_list[1].args[0]
        assert first_req.headers["Authorization"] == "Bearer expired-token"
        assert json.loads(first_req.data)["extra_data"]["auth_token_comfy_org"] == "expired-token"
        assert retry_req.headers["Authorization"] == "Bearer fresh-token"
        assert json.loads(retry_req.data)["extra_data"]["auth_token_comfy_org"] == "fresh-token"

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
        assert body["extra_data"] == {"api_key_comfy_org": "sk-test-1234", "comfy_usage_source": "comfy-cli"}

    def test_cloud_oauth_wins_over_api_key_when_both_set(self):
        """OAuth-first: if both are configured, the Bearer token wins."""
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
        assert req.headers["Authorization"] == "Bearer bearer-token"
        assert "X-api-key" not in req.headers
        body = json.loads(req.data)
        assert "auth_token_comfy_org" in body["extra_data"]
        assert "api_key_comfy_org" not in body["extra_data"]

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
    def test_cloud_uses_jobs_detail_endpoint(self):
        with patch.object(comfy_client._OPENER, "open", return_value=_mock_response({"status": "success"})) as urlopen:
            comfy_client.Client(CLOUD).get_job_status("pid-1")
        req = urlopen.call_args.args[0]
        assert req.full_url == "https://cloud.example.com/api/jobs/pid-1"

    def test_404_returns_none(self):
        with patch.object(comfy_client._OPENER, "open", side_effect=_http_error(404)):
            assert comfy_client.Client(CLOUD).get_job_status("pid") is None

    def test_local_raises_not_implemented(self):
        # A local target has no jobs_path — must fail loudly, not fall back to
        # GET {base}/<id> (mirrors list_jobs).
        with pytest.raises(NotImplementedError):
            comfy_client.Client(LOCAL).get_job_status("pid")

    @pytest.mark.parametrize("bad_id", ["", "   "])
    def test_empty_id_rejected(self, bad_id):
        # An empty id would collapse to the plural LIST endpoint and misclassify
        # the job — reject before issuing any request.
        with patch.object(comfy_client._OPENER, "open") as urlopen:
            with pytest.raises(ValueError):
                comfy_client.Client(CLOUD).get_job_status(bad_id)
        urlopen.assert_not_called()

    def test_id_is_percent_encoded(self):
        # A raw id with path/query metacharacters must not escape the
        # /api/jobs/<id> segment.
        with patch.object(comfy_client._OPENER, "open", return_value=_mock_response({"status": "success"})) as urlopen:
            comfy_client.Client(CLOUD).get_job_status("../admin?x=1")
        req = urlopen.call_args.args[0]
        assert req.full_url == "https://cloud.example.com/api/jobs/..%2Fadmin%3Fx%3D1"


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


class TestPollLevelBackoff:
    """In-request retries cover quick blips, but once a poll's request budget
    is spent — or on a plain 500, which the in-request layer never retries —
    the whole `run --wait` used to abort for a job that was still fine
    (fennec friction #2). wait_for_completion owns a poll-level budget:
    exponential backoff (base 2s, cap 60s, jitter, Retry-After honored) and
    only ~5 CONSECUTIVE poll failures surface the existing HTTPError path."""

    DONE = {"status": {"completed": True}, "outputs": {}}

    @staticmethod
    def _history_side_effect(events):
        """Each event is an Exception (raised) or a record (returned)."""
        it = iter(events)

        def _next(prompt_id, **kwargs):
            value = next(it)
            if isinstance(value, Exception):
                raise value
            return value

        return _next

    def test_429_twice_then_success_completes(self):
        client = comfy_client.Client(CLOUD)
        events = [
            comfy_client.HTTPError(429, "Too Many Requests"),
            comfy_client.HTTPError(429, "Too Many Requests"),
            self.DONE,
        ]
        with (
            patch("comfy_cli.comfy_client.time.sleep") as sleep,
            patch.object(client, "get_history", side_effect=self._history_side_effect(events)) as gh,
        ):
            assert client.wait_for_completion("pid", poll_interval=0) == self.DONE
        assert gh.call_count == 3
        assert sleep.call_count >= 2  # backed off between failed polls

    def test_500_twice_then_success_completes(self):
        # Plain 500 is NOT retried by the in-request layer; the poller must
        # absorb it instead of aborting the wait.
        client = comfy_client.Client(CLOUD)
        events = [
            comfy_client.HTTPError(500, "Internal Server Error"),
            comfy_client.HTTPError(500, "Internal Server Error"),
            self.DONE,
        ]
        with (
            patch("comfy_cli.comfy_client.time.sleep"),
            patch.object(client, "get_history", side_effect=self._history_side_effect(events)) as gh,
        ):
            assert client.wait_for_completion("pid", poll_interval=0) == self.DONE
        assert gh.call_count == 3

    def test_permanent_500_fails_after_retry_budget(self):
        client = comfy_client.Client(CLOUD)
        with (
            patch("comfy_cli.comfy_client.time.sleep"),
            patch.object(client, "get_history", side_effect=comfy_client.HTTPError(500, "boom")) as gh,
        ):
            with pytest.raises(comfy_client.HTTPError) as exc:
                client.wait_for_completion("pid", poll_interval=0)
        assert exc.value.status == 500
        assert gh.call_count == comfy_client._MAX_POLL_FAILURES

    def test_backoff_is_exponential_with_cap(self):
        client = comfy_client.Client(CLOUD)
        with (
            patch("comfy_cli.comfy_client.time.sleep") as sleep,
            patch("comfy_cli.comfy_client.random.uniform", return_value=0.0),
            patch.object(client, "get_history", side_effect=comfy_client.HTTPError(429, "rl")),
        ):
            with pytest.raises(comfy_client.HTTPError):
                client.wait_for_completion("pid", poll_interval=0)
        backoffs = [c.args[0] for c in sleep.call_args_list if c.args[0] > 0]
        assert backoffs == [2.0, 4.0, 8.0, 16.0]  # base 2s, doubling, jitter zeroed
        assert all(b <= 60.0 for b in backoffs)

    def test_retry_after_overrides_backoff(self):
        client = comfy_client.Client(CLOUD)
        events = [
            comfy_client.HTTPError(429, "rl", retry_after=7.0),
            self.DONE,
        ]
        with (
            patch("comfy_cli.comfy_client.time.sleep") as sleep,
            patch.object(client, "get_history", side_effect=self._history_side_effect(events)),
        ):
            assert client.wait_for_completion("pid", poll_interval=0) == self.DONE
        assert 7.0 in [c.args[0] for c in sleep.call_args_list]

    def test_failure_counter_resets_on_successful_poll(self):
        client = comfy_client.Client(CLOUD)
        budget = comfy_client._MAX_POLL_FAILURES
        events = (
            [comfy_client.HTTPError(429, "rl")] * (budget - 1)
            + [None]  # successful poll (job not done yet) resets the counter
            + [comfy_client.HTTPError(429, "rl")] * (budget - 1)
            + [self.DONE]
        )
        with (
            patch("comfy_cli.comfy_client.time.sleep"),
            patch.object(client, "get_history", side_effect=self._history_side_effect(events)) as gh,
        ):
            assert client.wait_for_completion("pid", poll_interval=0, timeout=600) == self.DONE
        assert gh.call_count == len(events)

    def test_non_transient_http_error_propagates_immediately(self):
        client = comfy_client.Client(CLOUD)
        with (
            patch("comfy_cli.comfy_client.time.sleep") as sleep,
            patch.object(client, "get_history", side_effect=comfy_client.HTTPError(403, "forbidden")) as gh,
        ):
            with pytest.raises(comfy_client.HTTPError) as exc:
                client.wait_for_completion("pid", poll_interval=0)
        assert exc.value.status == 403
        assert gh.call_count == 1
        assert sleep.call_count == 0

    def test_request_attaches_retry_after_to_http_error(self):
        from http.client import HTTPMessage

        hdrs = HTTPMessage()
        hdrs["Retry-After"] = "12"
        err = urllib.error.HTTPError(url="https://cloud/x", code=500, msg="500", hdrs=hdrs, fp=io.BytesIO(b""))
        with patch.object(comfy_client._OPENER, "open", side_effect=err):
            with pytest.raises(comfy_client.HTTPError) as exc:
                comfy_client.Client(CLOUD).submit_prompt({"1": {}}, "cid")
        assert exc.value.retry_after == 12.0


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

    def test_extract_outputs_keeps_node_association(self):
        """extract_outputs returns one dict per artifact with the producing
        node id — the node association that flat URL lists drop."""
        record = {
            "outputs": {
                "9": {
                    "images": [
                        {"filename": "a.png", "subfolder": "", "type": "output"},
                        {"filename": "b.png", "subfolder": "sub", "type": "temp"},
                    ]
                },
                "12": {"videos": [{"filename": "v.mp4", "subfolder": "", "type": "output"}]},
            }
        }
        out = comfy_client.Client(CLOUD).extract_outputs(record)
        assert out == [
            {
                "node_id": "9",
                "url": "https://cloud.example.com/api/view?filename=a.png&subfolder=&type=output",
                "filename": "a.png",
                "type": "output",
            },
            {
                "node_id": "9",
                "url": "https://cloud.example.com/api/view?filename=b.png&subfolder=sub&type=temp",
                "filename": "b.png",
                "type": "temp",
            },
            {
                "node_id": "12",
                "url": "https://cloud.example.com/api/view?filename=v.mp4&subfolder=&type=output",
                "filename": "v.mp4",
                "type": "output",
            },
        ]

    def test_extract_outputs_skips_non_dict_noise(self):
        record = {
            "outputs": {
                "3": "garbage-not-a-dict",
                "4": {"images": [{"no_filename": True}, "garbage", None]},
                "5": {"images": "not-a-list"},
                "6": {"images": [{"filename": "ok.png", "subfolder": "", "type": "output"}]},
            }
        }
        out = comfy_client.Client(CLOUD).extract_outputs(record)
        assert [o["node_id"] for o in out] == ["6"]
        assert comfy_client.Client(CLOUD).extract_outputs({}) == []
        assert comfy_client.Client(CLOUD).extract_outputs({"outputs": "nope"}) == []

    def test_extract_output_urls_delegates_to_extract_outputs(self):
        record = {
            "outputs": {
                "9": {"images": [{"filename": "a.png", "subfolder": "", "type": "output"}]},
                "12": {"videos": [{"filename": "v.mp4", "subfolder": "", "type": "output"}]},
            }
        }
        client = comfy_client.Client(CLOUD)
        assert client.extract_output_urls(record) == [o["url"] for o in client.extract_outputs(record)]


class TestGroupOutputs:
    """_group_outputs: pure grouping of extract_outputs entries by node and
    (via an item_map) by blueprint foreach item."""

    OUTPUTS = [
        {"node_id": "9", "url": "https://x/a.png", "filename": "a.png", "type": "output"},
        {"node_id": "9", "url": "https://x/b.png", "filename": "b.png", "type": "output"},
        {"node_id": "12", "url": "https://x/v.mp4", "filename": "v.mp4", "type": "output"},
    ]

    def test_groups_by_node(self):
        by_node, by_item = comfy_client._group_outputs(self.OUTPUTS, None)
        assert by_node == {"9": ["https://x/a.png", "https://x/b.png"], "12": ["https://x/v.mp4"]}
        assert by_item == {}

    def test_groups_by_item_via_item_map_nodes(self):
        item_map = {
            "s1": {"nodes": ["7", "9"], "save_node": "9", "prefix": "outputs/s1"},
            "s2": {"nodes": ["10", "12"], "save_node": "12", "prefix": "outputs/s2"},
        }
        _by_node, by_item = comfy_client._group_outputs(self.OUTPUTS, item_map)
        assert by_item == {
            "s1": ["https://x/a.png", "https://x/b.png"],
            "s2": ["https://x/v.mp4"],
        }

    def test_item_with_no_outputs_keeps_empty_list(self):
        # A pruned branch (item produced nothing) must still be visible.
        item_map = {
            "s1": {"nodes": ["9"], "save_node": "9", "prefix": "p"},
            "s2": {"nodes": ["99"], "save_node": "99", "prefix": "p"},
        }
        _by_node, by_item = comfy_client._group_outputs(self.OUTPUTS[:1], item_map)
        assert by_item == {"s1": ["https://x/a.png"], "s2": []}

    def test_save_node_membership_counts_even_outside_nodes_list(self):
        item_map = {"s1": {"nodes": ["7"], "save_node": "9", "prefix": "p"}}
        _by_node, by_item = comfy_client._group_outputs(self.OUTPUTS[:1], item_map)
        assert by_item == {"s1": ["https://x/a.png"]}

    def test_empty_inputs(self):
        assert comfy_client._group_outputs([], None) == ({}, {})
        assert comfy_client._group_outputs([], {}) == ({}, {})

    def test_skips_malformed_entries(self):
        outputs = [
            "garbage",
            {"url": "https://x/no-node.png"},
            {"node_id": "9"},
            {"node_id": "9", "url": "https://x/a.png"},
        ]
        item_map = {"s1": {"nodes": ["9"]}, "bad": "not-a-dict"}
        by_node, by_item = comfy_client._group_outputs(outputs, item_map)
        assert by_node == {"9": ["https://x/a.png"]}
        assert by_item == {"s1": ["https://x/a.png"], "bad": []}


class TestWaitForCompletionProgressProbe:
    def test_wait_for_completion_resets_idle_on_progress(self, monkeypatch):
        import comfy_cli.comfy_client as cc

        client = cc.Client(CLOUD)

        poll_interval = 0.04  # each poll sleeps ~40ms
        timeout = 0.06  # without reset, the idle timer trips after 60ms

        calls = {"n": 0}
        done = {"status": {"completed": True}, "outputs": {}}

        def history(pid):
            calls["n"] += 1
            return done if calls["n"] >= 5 else {"status": {"status_str": "running"}}

        counter = {"v": 0}

        def probe():
            counter["v"] += 1
            return ("running", counter["v"])  # advances every poll → resets last_change

        monkeypatch.setattr(client, "get_history", history)
        # ~5 polls × 40ms ≈ 200ms total elapsed, far past the 60ms timeout, but the
        # idle timer resets each poll because the probe value advances → must NOT raise.
        rec = client.wait_for_completion("pid", poll_interval=poll_interval, timeout=timeout, progress_probe=probe)
        assert rec is done

    def test_wait_for_completion_times_out_on_silence(self, monkeypatch):
        import comfy_cli.comfy_client as cc

        client = cc.Client(CLOUD)
        monkeypatch.setattr(client, "get_history", lambda pid: {"status": {"status_str": "running"}})
        with pytest.raises(TimeoutError):
            client.wait_for_completion("pid", poll_interval=0, timeout=0.05, progress_probe=lambda: ("running", 1))


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
