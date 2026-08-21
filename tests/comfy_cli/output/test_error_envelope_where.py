"""BE-6275: `where` on the ERROR envelopes of the routed commands.

BE-6274 fixed `Renderer.error()` and the `run` path. This module pins the same
property for every other command that resolves a local/cloud target: once the
routing decision is made, the failure envelope names the target it routed to
instead of `where: null`.

Shape of every test here: drive the real CLI (`comfy --json <cmd> --where …`)
with a forced downstream failure *after* target resolution, then assert
`envelope["where"]`. Driving the CLI rather than calling the inner function is
the point — the assignment under test lives at each command's routing decision,
and the top-level callback is what installs the per-invocation renderer.

The two negative tests at the bottom are the other half of the contract:
errors raised *before* a target is resolved must keep `where: null`, which is
why the field stays nullable in `comfy_cli/schemas/envelope.json`.
"""

from __future__ import annotations

import json
import urllib.error
from contextlib import contextmanager
from unittest.mock import patch

import pytest
from typer.testing import CliRunner

from comfy_cli import cmdline
from comfy_cli import where as where_module
from comfy_cli.cql.engine import LoadError


@pytest.fixture(autouse=True)
def _no_ambient_where(monkeypatch: pytest.MonkeyPatch):
    """`--where` must be the only routing input.

    `COMFY_WHERE` is set process-wide by the top-level `comfy --where` flag, so
    a value left behind by another test (or by the developer's shell) would
    silently decide the target these tests are asserting on.
    """
    monkeypatch.delenv("COMFY_WHERE", raising=False)


def _run(argv: list[str]) -> dict:
    """Invoke the CLI and return the terminating envelope."""
    result = CliRunner().invoke(cmdline.app, argv)
    lines = [ln for ln in (result.stdout or "").splitlines() if ln.strip()]
    assert lines, f"no envelope on stdout (rc={result.exit_code}, exc={result.exception!r})"
    envelope = json.loads(lines[-1])
    assert envelope.get("ok") is False, f"expected a failure envelope, got {envelope}"
    return envelope


@contextmanager
def _cloud_signed_out():
    """Fail the shared cloud preflight — the cheapest post-routing failure for
    any command whose cloud branch starts with `cloud_preflight_or_exit()`."""
    err = where_module.CloudError(
        code="cloud_not_configured",
        message="not signed in",
        hint="run: comfy cloud login",
        details={},
    )
    with patch("comfy_cli.where.cloud_preflight", return_value=err):
        yield


_NETWORK_DOWN = urllib.error.URLError("connection refused")
_LOAD_ERROR = LoadError("no object_info", details={"hint": "start the server"})


# ---------------------------------------------------------------------------
# jobs
# ---------------------------------------------------------------------------


class TestJobs:
    def test_status_cloud(self):
        with _cloud_signed_out():
            env = _run(["--json", "jobs", "status", "abc123", "--where", "cloud"])
        assert env["error"]["code"] == "cloud_not_configured"
        assert env["where"] == "cloud"

    def test_status_local(self):
        with (
            patch("comfy_cli.command.jobs._server_or_error", return_value=False),
            patch("comfy_cli.command.jobs._state_file_for_local_target", return_value=None),
        ):
            env = _run(["--json", "jobs", "status", "abc123", "--where", "local"])
        assert env["error"]["code"] == "server_not_running"
        assert env["where"] == "local"

    def test_cancel_cloud(self):
        with _cloud_signed_out():
            env = _run(["--json", "jobs", "cancel", "abc123", "--where", "cloud"])
        assert env["where"] == "cloud"

    def test_watch_cloud(self):
        with _cloud_signed_out():
            env = _run(["--json", "jobs", "watch", "abc123", "--where", "cloud"])
        assert env["where"] == "cloud"

    def test_wait_cloud(self):
        # `no_prompt_ids` is raised straight after the routing decision, so it
        # is the narrowest probe of the stamp on the `wait` path.
        env = _run(["--json", "jobs", "wait", "--where", "cloud"])
        assert env["error"]["code"] == "no_prompt_ids"
        assert env["where"] == "cloud"

    def test_wait_local(self):
        env = _run(["--json", "jobs", "wait", "--where", "local"])
        assert env["error"]["code"] == "no_prompt_ids"
        assert env["where"] == "local"

    def test_ls_rejects_watch_in_json_mode_with_target(self):
        env = _run(["--json", "jobs", "ls", "--where", "cloud", "--watch"])
        assert env["error"]["code"] == "json_incompatible"
        assert env["where"] == "cloud"


# ---------------------------------------------------------------------------
# workflow (saved-workflow verbs)
# ---------------------------------------------------------------------------


class TestWorkflow:
    def test_list_cloud(self):
        with patch("comfy_cli.command.workflow._http_request", side_effect=_NETWORK_DOWN):
            env = _run(["--json", "workflow", "list", "--where", "cloud"])
        assert env["where"] == "cloud"

    def test_list_local(self):
        with patch("comfy_cli.command.workflow._userdata_request", side_effect=_NETWORK_DOWN):
            env = _run(["--json", "workflow", "list", "--where", "local"])
        assert env["where"] == "local"

    def test_get_cloud(self):
        with patch("comfy_cli.command.workflow._http_request", side_effect=_NETWORK_DOWN):
            env = _run(["--json", "workflow", "get", "some-id", "--where", "cloud"])
        assert env["where"] == "cloud"

    def test_delete_local(self):
        with patch("comfy_cli.command.workflow._userdata_request", side_effect=_NETWORK_DOWN):
            env = _run(["--json", "workflow", "delete", "flux.json", "--where", "local"])
        assert env["where"] == "local"

    def test_slots_stamps_the_object_info_route(self, tmp_path, monkeypatch: pytest.MonkeyPatch):
        """`slots`/`set-slot`/`vary`/`notes` route through `workflow._get_graph`,
        a different resolver than the saved-workflow verbs above — it has no
        per-command `--where`, so routing arrives via `COMFY_WHERE`."""
        path = tmp_path / "wf.json"
        path.write_text(json.dumps({"nodes": [], "links": [], "last_node_id": 0, "last_link_id": 0}))
        monkeypatch.setenv("COMFY_WHERE", "cloud")
        with patch("comfy_cli.cql.loader.resilient_load_object_info", side_effect=_LOAD_ERROR):
            env = _run(["--json", "workflow", "slots", str(path)])
        assert env["error"]["code"] == "cql_no_graph"
        assert env["where"] == "cloud"

    def test_decompose_stamps_the_object_info_route(self, tmp_path, monkeypatch: pytest.MonkeyPatch):
        """`workflow decompose` loads object_info through its own resolver in
        `workflow_fragments`, reached only for a frontend-format workflow."""
        path = tmp_path / "ui.json"
        path.write_text(json.dumps({"nodes": [], "links": [], "last_node_id": 0, "last_link_id": 0}))
        monkeypatch.setenv("COMFY_WHERE", "cloud")
        with patch("comfy_cli.cql.loader.resilient_load_object_info", side_effect=_LOAD_ERROR):
            env = _run(["--json", "workflow", "decompose", str(path)])
        assert env["error"]["code"] == "object_info_unavailable"
        assert env["where"] == "cloud"


# ---------------------------------------------------------------------------
# transfer - upload / download
# ---------------------------------------------------------------------------


class TestTransfer:
    def test_upload_rejects_host_with_cloud_target(self):
        """The `--host/--port + cloud` rejection fires before `execute_upload`
        is ever called, so it can only carry `where` via the stamp in
        `cmdline.upload`."""
        env = _run(["--json", "upload", "x.png", "--where", "cloud", "--host", "1.2.3.4"])
        assert env["error"]["code"] == "host_flag_cloud"
        assert env["where"] == "cloud"

    def test_upload_local_missing_file(self, tmp_path):
        env = _run(["--json", "upload", str(tmp_path / "nope.png"), "--where", "local"])
        assert env["error"]["code"] == "upload_failed"
        assert env["where"] == "local"

    def test_download_cloud(self):
        with _cloud_signed_out():
            env = _run(["--json", "download", "abc123", "--where", "cloud"])
        assert env["error"]["code"] == "cloud_not_configured"
        assert env["where"] == "cloud"


# ---------------------------------------------------------------------------
# system - system-stats / free
# ---------------------------------------------------------------------------


class TestSystem:
    def test_system_stats_local(self):
        with patch("comfy_cli.comfy_client.Client.get_system_stats", side_effect=OSError("boom")):
            env = _run(["--json", "system-stats", "--where", "local"])
        assert env["error"]["code"] == "server_not_running"
        assert env["where"] == "local"

    def test_system_stats_cloud(self, monkeypatch: pytest.MonkeyPatch):
        # A credential is required for `Client(target)` to construct at all on
        # the cloud branch — without one it raises `Unauthenticated` before any
        # envelope is written (pre-existing, unrelated to the stamp).
        monkeypatch.setenv("COMFY_CLOUD_API_KEY", "test-key")
        with patch("comfy_cli.comfy_client.Client.get_system_stats", side_effect=OSError("boom")):
            env = _run(["--json", "system-stats", "--where", "cloud"])
        assert env["error"]["code"] == "cloud_http_error"
        assert env["where"] == "cloud"

    def test_free_local(self):
        with patch("comfy_cli.comfy_client.Client.post_free", side_effect=OSError("boom")):
            env = _run(["--json", "free", "--where", "local"])
        assert env["error"]["code"] == "server_not_running"
        assert env["where"] == "local"


# ---------------------------------------------------------------------------
# nodes
# ---------------------------------------------------------------------------


class TestNodes:
    def test_ls_local(self):
        with patch("comfy_cli.cql.loader.resilient_load_object_info", side_effect=_LOAD_ERROR):
            env = _run(["--json", "nodes", "ls", "--where", "local"])
        assert env["error"]["code"] == "cql_no_graph"
        assert env["where"] == "local"

    def test_ls_cloud(self):
        with patch("comfy_cli.cql.loader.resilient_load_object_info", side_effect=_LOAD_ERROR):
            env = _run(["--json", "nodes", "ls", "--where", "cloud"])
        assert env["error"]["code"] == "cql_no_graph"
        assert env["where"] == "cloud"

    def test_show_cloud(self):
        with patch("comfy_cli.cql.loader.resilient_load_object_info", side_effect=_LOAD_ERROR):
            env = _run(["--json", "nodes", "show", "KSampler", "--where", "cloud"])
        assert env["where"] == "cloud"


# ---------------------------------------------------------------------------
# models
# ---------------------------------------------------------------------------


class TestModels:
    def test_search_cloud(self):
        with patch("comfy_cli.command.models.search._cloud_search", side_effect=_NETWORK_DOWN):
            env = _run(["--json", "models", "search", "--where", "cloud"])
        assert env["error"]["code"] == "cloud_http_error"
        assert env["where"] == "cloud"

    def test_search_local(self):
        with patch("comfy_cli.command.models.search._local_search", side_effect=_NETWORK_DOWN):
            env = _run(["--json", "models", "search", "--where", "local"])
        assert env["error"]["code"] == "server_not_running"
        assert env["where"] == "local"

    def test_list_folders_local(self):
        with patch("comfy_cli.command.models.search._http_get_json", side_effect=_NETWORK_DOWN):
            env = _run(["--json", "models", "list-folders", "--where", "local"])
        assert env["where"] == "local"

    def test_show_local_is_unsupported_but_still_names_the_target(self):
        env = _run(["--json", "models", "show", "vae.safetensors", "--where", "local"])
        assert env["error"]["code"] == "models_show_local_unsupported"
        assert env["where"] == "local"


# ---------------------------------------------------------------------------
# assets push (project)
# ---------------------------------------------------------------------------


class TestAssetsPush:
    @staticmethod
    def _project(tmp_path, monkeypatch):
        root = tmp_path / "proj"
        root.mkdir()
        (root / "comfy.yaml").write_text("schema: project/1\nname: t\n")
        for d in ("assets", ".comfy"):
            (root / d).mkdir()
        (root / "assets" / "a.png").write_bytes(b"png")
        monkeypatch.chdir(root)

    def test_push_cloud(self, tmp_path, monkeypatch: pytest.MonkeyPatch):
        self._project(tmp_path, monkeypatch)
        err = urllib.error.HTTPError("http://x", 500, "boom", {}, None)
        with patch("comfy_cli.command.project._upload_file", side_effect=err):
            env = _run(["--json", "assets", "push", "--where", "cloud"])
        assert env["error"]["code"] == "upload_failed"
        assert env["where"] == "cloud"

    def test_push_local(self, tmp_path, monkeypatch: pytest.MonkeyPatch):
        self._project(tmp_path, monkeypatch)
        err = urllib.error.HTTPError("http://x", 500, "boom", {}, None)
        with patch("comfy_cli.command.project._upload_file", side_effect=err):
            env = _run(["--json", "assets", "push", "--where", "local"])
        assert env["error"]["code"] == "upload_failed"
        assert env["where"] == "local"


# ---------------------------------------------------------------------------
# validate
# ---------------------------------------------------------------------------


class TestValidate:
    @staticmethod
    def _workflow(tmp_path):
        path = tmp_path / "wf.json"
        path.write_text(json.dumps({"1": {"class_type": "KSampler", "inputs": {}}}))
        return str(path)

    def test_cloud(self, tmp_path):
        err = LoadError("no object_info", details={"hint": "sign in"})
        with patch("comfy_cli.cql.engine.Graph.load", side_effect=err):
            env = _run(["--json", "validate", "--workflow", self._workflow(tmp_path), "--where", "cloud"])
        assert env["error"]["code"] == "cql_no_graph"
        assert env["where"] == "cloud"

    def test_local(self, tmp_path):
        err = LoadError("no object_info", details={"hint": "comfy launch"})
        with patch("comfy_cli.cql.engine.Graph.load", side_effect=err):
            env = _run(["--json", "validate", "--workflow", self._workflow(tmp_path), "--where", "local"])
        assert env["error"]["code"] == "cql_no_graph"
        assert env["where"] == "local"


# ---------------------------------------------------------------------------
# logs (local-only routing)
# ---------------------------------------------------------------------------


class TestLogs:
    def test_no_log_file_local(self):
        with patch("comfy_cli.command.launch.resolve_background_log_path", return_value=None):
            env = _run(["--json", "logs", "--where", "local"])
        assert env["error"]["code"] == "no_log_file"
        assert env["where"] == "local"

    def test_cloud_routing_rejected_before_the_decision_lands(self):
        """`comfy logs` refuses cloud routing outright, so that rejection is a
        failed decision — it keeps `where: null` like any other `where_invalid`."""
        env = _run(["--json", "logs", "--where", "cloud"])
        assert env["error"]["code"] == "where_invalid"
        assert env["where"] is None


# ---------------------------------------------------------------------------
# Pre-decision errors keep `where: null`
# ---------------------------------------------------------------------------


class TestPreDecisionErrorsStayNull:
    def test_where_invalid_on_upload(self):
        env = _run(["--json", "upload", "x.png", "--where", "nowhere"])
        assert env["error"]["code"] == "where_invalid"
        assert env["where"] is None

    def test_where_invalid_on_validate(self, tmp_path):
        path = tmp_path / "wf.json"
        path.write_text("{}")
        env = _run(["--json", "validate", "--workflow", str(path), "--where", "nowhere"])
        assert env["error"]["code"] == "where_invalid"
        assert env["where"] is None

    def test_validate_file_error_precedes_routing(self, tmp_path):
        env = _run(["--json", "validate", "--workflow", str(tmp_path / "gone.json"), "--where", "cloud"])
        assert env["error"]["code"] == "workflow_not_found"
        assert env["where"] is None

    def test_models_list_folder_rejects_unsafe_segment_before_routing(self):
        env = _run(["--json", "models", "list-folder", "../etc", "--where", "cloud"])
        assert env["where"] is None
