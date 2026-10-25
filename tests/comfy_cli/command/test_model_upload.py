"""Cloud model uploads enforce target, bytes and explicit named-file approval."""

import hashlib
import inspect
import json
import urllib.error
import urllib.parse
from unittest.mock import Mock

import blake3
import pytest
import typer
from typer.testing import CliRunner

from comfy_cli.target import Target


def test_model_upload_is_a_discoverable_command():
    from comfy_cli.command.models.models import app

    result = CliRunner().invoke(app, ["upload", "--help"])
    assert result.exit_code == 0, result.output
    assert "--dry-run" in result.output
    assert "--expected-sha256" in result.output


def setup_upload(tmp_path, monkeypatch, *, flags=None, current="ws-approved", rows=None):
    from comfy_cli.command.models import upload

    path = tmp_path / "private.safetensors"
    path.write_bytes(b"synthetic-approved-model")
    renderer = Mock()
    renderer.is_pretty.return_value = False
    target = Target("cloud", "https://cloud.example.test", "/api", auth_token="test-secret")
    reads = []

    def read(url, target, *, max_bytes, **kwargs):
        reads.append(url)
        if url.endswith("/workspaces/current"):
            return 200, {"id": current, "name": "Test Workspace"}
        if url.endswith("/features"):
            return 200, flags if flags is not None else {"private_models_enabled": True}
        if url.endswith("/experiment/models"):
            return 200, [{"name": "loras"}, {"name": "checkpoints"}]
        return 200, {"assets": rows or [], "has_more": False}

    sent = []

    def send(request, **kwargs):
        chunks = b"".join(request.data)
        sent.append(chunks)
        body = {
            "id": "asset-test",
            "name": path.name,
            "tags": ["models", "loras"],
            "hash": "blake3:" + blake3.blake3(path.read_bytes()).hexdigest(),
            "loader_path": path.name,
            "created_new": True,
        }
        result = Mock()
        result.status = 201
        result.read.side_effect = [json.dumps(body).encode(), b""]
        result.__enter__ = Mock(return_value=result)
        result.__exit__ = Mock(return_value=False)
        return result

    monkeypatch.setattr(upload, "get_renderer", lambda: renderer)
    monkeypatch.setattr(upload, "cloud_target_or_local_error", lambda *a: target)
    monkeypatch.setattr(upload, "request_json", read)
    monkeypatch.setattr(upload, "no_redirect_urlopen", send)
    return upload, path, renderer, reads, sent


def run_case(module, path, **changes):
    options = dict(
        model_type="loras",
        workspace="ws-approved",
        dry_run=False,
        yes=True,
        expected_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        name=None,
        where="cloud",
    )
    options.update(changes)
    return inspect.unwrap(module.upload_cmd)(path, **options)


def test_plan_never_uploads_and_names_exact_bytes_and_target(tmp_path, monkeypatch):
    m, p, r, reads, sent = setup_upload(tmp_path, monkeypatch)
    run_case(m, p, dry_run=True, yes=False, expected_sha256=None)
    assert sent == []
    plan = r.emit.call_args.args[0]
    assert plan["sha256"] == hashlib.sha256(p.read_bytes()).hexdigest()
    assert plan["workspace_id"] == "ws-approved"
    assert plan["destination"].endswith("/api/assets")
    assert plan["size_bytes"] == p.stat().st_size
    assert plan["uploaded"] is False


@pytest.mark.parametrize(
    "changes",
    [
        {"yes": False},
        {"expected_sha256": None},
        {"expected_sha256": "0" * 64},
        {"workspace": "ws-other"},
        {"model_type": "unsupported"},
    ],
)
def test_missing_or_changed_approval_cannot_upload(tmp_path, monkeypatch, changes):
    m, p, r, reads, sent = setup_upload(tmp_path, monkeypatch)
    with pytest.raises(typer.Exit):
        run_case(m, p, **changes)
    assert sent == []
    r.error.assert_called_once()


@pytest.mark.parametrize("flags", [{}, {"private_models_enabled": False}, {"private_models_enabled": "true"}])
def test_capability_must_be_explicitly_enabled(tmp_path, monkeypatch, flags):
    m, p, r, reads, sent = setup_upload(tmp_path, monkeypatch, flags=flags)
    with pytest.raises(typer.Exit):
        run_case(m, p)
    assert sent == []


def test_approved_upload_streams_exact_file_and_model_tags(tmp_path, monkeypatch):
    m, p, r, reads, sent = setup_upload(tmp_path, monkeypatch)
    run_case(m, p)
    assert len(sent) == 1
    assert p.read_bytes() in sent[0]
    assert b'["models", "loras"]' in sent[0]
    assert r.emit.call_args.args[0]["loader_path"] == p.name
    assert r.emit.call_args.args[0]["uploaded"] is True


@pytest.mark.parametrize("hash_value", ["blake3:" + "0" * 64, None, "unverified"])
def test_same_name_different_or_unverified_cloud_bytes_requires_new_name(tmp_path, monkeypatch, hash_value):
    m, p, r, reads, sent = setup_upload(
        tmp_path, monkeypatch, rows=[{"name": "private.safetensors", "hash": hash_value}]
    )
    with pytest.raises(typer.Exit):
        run_case(m, p)
    assert sent == []


def test_mutation_after_snapshot_cannot_change_uploaded_bytes(tmp_path, monkeypatch):
    m, p, r, reads, sent = setup_upload(tmp_path, monkeypatch)
    original = p.read_bytes()
    original_send = m.no_redirect_urlopen

    def mutate_then_send(request, **kwargs):
        p.write_bytes(b"UNAPPROVED CHANGED CONTENT")
        # The mock response must describe the bytes actually sent, not the changed source.
        result = original_send(request, **kwargs)
        body = {
            "id": "asset-test",
            "name": p.name,
            "tags": ["models", "loras"],
            "hash": "blake3:" + blake3.blake3(original).hexdigest(),
            "loader_path": p.name,
            "created_new": True,
        }
        result.read.side_effect = [json.dumps(body).encode(), b""]
        return result

    monkeypatch.setattr(m, "no_redirect_urlopen", mutate_then_send)
    run_case(m, p, expected_sha256=hashlib.sha256(original).hexdigest())
    assert original in sent[0]
    assert p.read_bytes() not in sent[0]


def test_legacy_cloud_hash_filename_and_loader_name_are_verified(tmp_path, monkeypatch):
    m, p, r, reads, sent = setup_upload(tmp_path, monkeypatch)
    send = m.no_redirect_urlopen

    def legacy_response(request, **kwargs):
        response = send(request, **kwargs)
        body = json.loads(response.read())
        body["hash"] = body["hash"].removeprefix("blake3:") + ".safetensors"
        body.pop("loader_path")
        response.read.side_effect = [json.dumps(body).encode(), b""]
        return response

    monkeypatch.setattr(m, "no_redirect_urlopen", legacy_response)
    run_case(m, p)
    assert r.emit.call_args.args[0]["loader_path"] == p.name
    r.error.assert_not_called()


def test_different_model_folder_cannot_be_reported_as_requested_import(tmp_path, monkeypatch):
    m, p, r, reads, sent = setup_upload(tmp_path, monkeypatch)
    send = m.no_redirect_urlopen

    def wrong_folder(request, **kwargs):
        response = send(request, **kwargs)
        body = json.loads(response.read())
        body["tags"] = ["models", "checkpoints"]
        response.read.side_effect = [json.dumps(body).encode(), b""]
        return response

    monkeypatch.setattr(m, "no_redirect_urlopen", wrong_folder)
    with pytest.raises(typer.Exit):
        run_case(m, p)
    assert r.error.call_args.kwargs["details"]["uploaded"] is True


def test_identical_workspace_owned_asset_is_reused_without_uploading_bytes(tmp_path, monkeypatch):
    m, p, r, reads, sent = setup_upload(tmp_path, monkeypatch)
    digest = "blake3:" + blake3.blake3(p.read_bytes()).hexdigest()
    read = m.request_json
    operations = []

    def existing_asset(url, target, **kwargs):
        operations.append((url, kwargs.get("method", "GET")))
        if "/assets?" in url:
            return 200, {"assets": [{"id": "asset-existing", "name": p.name, "hash": digest}], "has_more": False}
        if url.endswith("/assets/from-hash"):
            assert kwargs["body"] == {"hash": digest, "tags": ["models", "loras"]}
            return 200, {
                "id": "asset-existing",
                "name": p.name,
                "hash": digest,
                "tags": ["models", "loras"],
                "loader_path": p.name,
                "created_new": False,
            }
        return read(url, target, **kwargs)

    monkeypatch.setattr(m, "request_json", existing_asset)
    run_case(m, p)
    assert sent == []
    assert r.emit.call_args.args[0]["deduped"] is True
    assert r.emit.call_args.args[0]["uploaded"] is False
    assert any(url.endswith("/assets/from-hash") and method == "POST" for url, method in operations)
    assert any("include_public=false" in url for url, _ in operations)


def test_public_only_identical_asset_uploads_approved_snapshot(tmp_path, monkeypatch):
    m, p, r, reads, sent = setup_upload(tmp_path, monkeypatch)
    approved = p.read_bytes()
    digest = "blake3:" + blake3.blake3(approved).hexdigest()
    read = m.request_json
    operations = []

    def public_only_asset(url, target, **kwargs):
        operations.append((url, kwargs.get("method", "GET")))
        if "/assets?" in url:
            query = urllib.parse.parse_qs(urllib.parse.urlsplit(url).query)
            rows = [{"id": "public-asset", "name": p.name, "hash": digest}]
            return 200, {"assets": rows if query["include_public"] == ["true"] else [], "has_more": False}
        if url.endswith("/assets/from-hash"):
            # Cloud's ownership preflight excludes public assets and refuses
            # an unowned digest even though the bytes are publicly accessible.
            raise urllib.error.HTTPError(url, 404, "ASSET_NOT_FOUND", {}, None)
        return read(url, target, **kwargs)

    monkeypatch.setattr(m, "request_json", public_only_asset)
    run_case(m, p)
    assert len(sent) == 1
    assert approved in sent[0]
    assert b'["models", "loras"]' in sent[0]
    result = r.emit.call_args.args[0]
    assert result["uploaded"] is True
    assert result["deduped"] is False
    assert result["sha256"] == hashlib.sha256(approved).hexdigest()
    assert result["workspace_id"] == "ws-approved"
    assert result["loader_path"] == p.name
    assert not any(url.endswith("/assets/from-hash") for url, _ in operations)
    r.error.assert_not_called()


@pytest.mark.parametrize(
    "owned_listing",
    [{}, {"assets": [], "has_more": True}, {"assets": [], "has_more": None}, {"assets": [None], "has_more": False}],
)
def test_unverified_workspace_ownership_cannot_dedup_or_upload(tmp_path, monkeypatch, owned_listing):
    m, p, r, reads, sent = setup_upload(tmp_path, monkeypatch)
    digest = "blake3:" + blake3.blake3(p.read_bytes()).hexdigest()
    read = m.request_json
    operations = []

    def unverified_ownership(url, target, **kwargs):
        operations.append((url, kwargs.get("method", "GET")))
        if "/assets?" in url:
            query = urllib.parse.parse_qs(urllib.parse.urlsplit(url).query)
            if query["include_public"] == ["false"]:
                return 200, owned_listing
            return 200, {"assets": [{"id": "visible-asset", "name": p.name, "hash": digest}], "has_more": False}
        if url.endswith("/assets/from-hash"):
            return 200, {"id": "unexpected-dedup", "hash": digest, "tags": ["models", "loras"], "loader_path": p.name}
        return read(url, target, **kwargs)

    monkeypatch.setattr(m, "request_json", unverified_ownership)
    with pytest.raises(typer.Exit):
        run_case(m, p)
    assert sent == []
    assert not any(method == "POST" for _, method in operations)
    r.emit.assert_not_called()
    assert r.error.call_args.kwargs["details"]["reason"] == "ownership_unverified"


@pytest.mark.parametrize("mode", ["bad_json", "oversize", "wrong_hash"])
def test_unverifiable_upload_response_never_reports_success(tmp_path, monkeypatch, mode):
    from comfy_cli.http import ResponseTooLarge

    m, p, r, reads, sent = setup_upload(tmp_path, monkeypatch)
    send = m.no_redirect_urlopen

    def bad_response(request, **kwargs):
        response = send(request, **kwargs)
        if mode == "bad_json":
            response.read.side_effect = [b"<html>error</html>", b""]
        elif mode == "oversize":
            response.read.side_effect = ResponseTooLarge("response too large")
        else:
            body = json.loads(response.read())
            body["hash"] = "blake3:" + "0" * 64
            response.read.side_effect = [json.dumps(body).encode(), b""]
        return response

    monkeypatch.setattr(m, "no_redirect_urlopen", bad_response)
    with pytest.raises(typer.Exit):
        run_case(m, p)
    r.emit.assert_not_called()
    r.error.assert_called_once()


@pytest.mark.parametrize("public_match", [False, True])
def test_actual_http_transport_streams_approved_snapshot_with_exact_length(tmp_path, monkeypatch, public_match):
    import http.server
    import threading

    from comfy_cli.command.models import upload

    source = tmp_path / "transport-model.safetensors"
    source.write_bytes(b"APPROVED-MODEL-BYTES" * 65536)
    approved = source.read_bytes()
    observed = []

    class Handler(http.server.BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def answer(self, payload):
            encoded = json.dumps(payload).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(encoded)))
            self.end_headers()
            self.wfile.write(encoded)

        def do_GET(self):
            if self.path.endswith("/workspaces/current"):
                self.answer({"id": "ws-approved", "name": "Test"})
            elif self.path.endswith("/features"):
                self.answer({"private_models_enabled": True})
            elif self.path.endswith("/experiment/models"):
                self.answer([{"name": "loras"}])
            else:
                query = urllib.parse.parse_qs(urllib.parse.urlsplit(self.path).query)
                rows = []
                if public_match and query.get("include_public") == ["true"]:
                    rows = [
                        {
                            "id": "public-only",
                            "name": source.name,
                            "hash": "blake3:" + blake3.blake3(approved).hexdigest(),
                        }
                    ]
                self.answer({"assets": rows, "has_more": False})

        def do_POST(self):
            assert self.path == "/api/assets"
            source.write_bytes(b"CHANGED-AFTER-CLIENT-SNAPSHOT")
            body = self.rfile.read(int(self.headers["Content-Length"]))
            observed.append((body, self.headers["Authorization"], self.headers["Content-Length"]))
            self.answer(
                {
                    "id": "asset-transport",
                    "name": source.name,
                    "tags": ["models", "loras"],
                    "hash": "blake3:" + blake3.blake3(approved).hexdigest(),
                    "loader_path": source.name,
                    "created_new": True,
                }
            )

    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    renderer = Mock()
    renderer.is_pretty.return_value = False
    target = Target("cloud", f"http://127.0.0.1:{server.server_port}", "/api", auth_token="test-token")
    monkeypatch.setattr(upload, "cloud_target_or_local_error", lambda *args: target)
    monkeypatch.setattr(upload, "get_renderer", lambda: renderer)
    try:
        run_case(upload, source, expected_sha256=hashlib.sha256(approved).hexdigest())
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
    assert len(observed) == 1
    body, credential, length = observed[0]
    assert credential == "Bearer test-token"
    assert len(body) == int(length)
    assert approved in body
    assert source.read_bytes() not in body
    renderer.error.assert_not_called()


@pytest.mark.parametrize("dry_run", [False, True], ids=["upload", "plan"])
@pytest.mark.parametrize(
    ("status", "rows"),
    [(200, [None]), (200, ["malformed"]), (201, []), (503, [])],
    ids=["null-row", "string-row", "non-ok-success", "non-ok-error"],
)
def test_unverified_public_collision_listing_never_plans_or_uploads(tmp_path, monkeypatch, dry_run, status, rows):
    m, p, r, reads, sent = setup_upload(tmp_path, monkeypatch)
    read = m.request_json
    operations = []

    def malformed_public_listing(url, target, **kwargs):
        operations.append((url, kwargs.get("method", "GET")))
        if "/assets?" in url:
            return status, {"assets": rows, "has_more": False}
        return read(url, target, **kwargs)

    monkeypatch.setattr(m, "request_json", malformed_public_listing)
    with pytest.raises(typer.Exit):
        run_case(m, p, dry_run=dry_run)
    assert sent == []
    assert not any(method == "POST" for _, method in operations)
    r.emit.assert_not_called()
    assert r.error.call_args.kwargs["details"]["reason"] == "collision_unverified"
