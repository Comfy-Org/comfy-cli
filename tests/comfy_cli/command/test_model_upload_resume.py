"""A fake bucket holds bytes across connection drops and process reruns."""

import json
import os
import time
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest
import requests

from comfy_cli.command import assets_library as uploads
from comfy_cli.command import model_upload_resume as resume
from comfy_cli.hashing import file_digests


@pytest.fixture
def session():
    return {
        "upload_id": "upload-1",
        "upload_url": "https://bucket.example/upload?secret=SESSION",
        "upload_mode": "resumable",
        "method": "PUT",
        "headers": {},
        "chunk_size_bytes": 256 * 1024,
        "expires_at": datetime.fromtimestamp(time.time() + resume.LIFETIME, timezone.utc).isoformat(),
    }


class Bucket:
    def __init__(self, *, held=b"", drop=None, lose_final=False):
        self.held, self.drop, self.lose_final = held, drop, lose_final
        self.calls = []

    def __call__(self, url, *, data, headers, timeout, allow_redirects):
        assert allow_redirects is False
        assert headers["Content-Length"] == str(len(data))
        content_range = headers["Content-Range"]
        self.calls.append(content_range)
        size = int(content_range.split("/")[1])
        if not content_range.startswith("bytes */"):
            offset = int(content_range.split()[1].split("-")[0])
            assert offset == len(self.held)
            if self.drop is not None:
                self.held += data[: self.drop]
                self.drop = None
                raise requests.ConnectionError(url)
            self.held += data
            if self.lose_final and len(self.held) == size:
                self.lose_final = False
                raise requests.ConnectionError(url)
        code = 200 if len(self.held) == size else 308
        response_headers = {"Range": f"bytes=0-{len(self.held) - 1}"} if self.held else {}
        return SimpleNamespace(status_code=code, headers=response_headers, close=lambda: None)


@pytest.fixture
def file(tmp_path):
    path = tmp_path / "checkpoint.safetensors"
    path.write_bytes(bytes(range(256)) * 2500)
    return path


@pytest.fixture
def clock():
    now = [0.0]
    return lambda: now[0], lambda s: now.__setitem__(0, now[0] + s)


def send(session, file, clock, *, resumed=False):
    digests = file_digests(file, block_size=4096)
    progress = []
    resume.transfer(
        session,
        digests.size,
        lambda o, n: uploads._read_range(file, digests, o, n),
        progress.append,
        resumed=resumed,
        clock=clock[0],
        sleep=clock[1],
    )
    return progress


def test_a_drop_mid_piece_queries_the_held_range_before_sending(session, file, clock, monkeypatch):
    bucket = Bucket(drop=12345)
    monkeypatch.setattr(requests, "put", bucket)
    assert sum(send(session, file, clock)) == file.stat().st_size
    assert bucket.held == file.read_bytes()
    assert bucket.calls[:3] == [
        f"bytes 0-262143/{file.stat().st_size}",
        f"bytes */{file.stat().st_size}",
        f"bytes 12345-274488/{file.stat().st_size}",
    ]


def test_a_rerun_queries_before_sending_and_checks_the_partial_hash_block(session, file, clock, monkeypatch):
    bucket = Bucket(held=file.read_bytes()[:12345])
    monkeypatch.setattr(requests, "put", bucket)
    assert sum(send(session, file, clock, resumed=True)) == file.stat().st_size
    assert bucket.held == file.read_bytes()
    assert bucket.calls[0] == f"bytes */{file.stat().st_size}"


@pytest.mark.parametrize("rerun", [False, True])
def test_last_piece_without_an_answer_goes_to_complete(session, file, clock, monkeypatch, rerun):
    bucket = Bucket(held=file.read_bytes() if rerun else b"", lose_final=not rerun)
    monkeypatch.setattr(requests, "put", bucket)
    send(session, file, clock, resumed=rerun)
    assert bucket.calls[-1] == f"bytes */{file.stat().st_size}"
    if rerun:
        assert len(bucket.calls) == 1


@pytest.mark.parametrize("status", [408, 429, 500, 503, None])
def test_transient_errors_back_off_and_query(session, file, clock, monkeypatch, status):
    bucket = Bucket()
    calls = []

    def put(*args, **kw):
        calls.append(kw["headers"]["Content-Range"])
        if len(calls) == 1:
            if status is None:
                raise requests.ConnectionError("SESSION")
            return SimpleNamespace(status_code=status, headers={}, close=lambda: None)
        return bucket(*args, **kw)

    monkeypatch.setattr(requests, "put", put)
    send(session, file, clock)
    assert calls[1].startswith("bytes */")
    assert clock[0]() == 1


@pytest.mark.parametrize(
    "code,exception",
    [
        (404, resume.SessionExpired),
        (410, resume.SessionExpired),
        (403, resume.TransferFailed),
        (307, resume.TransferFailed),
    ],
)
def test_expiry_or_refusal_never_follows_a_redirect(session, file, clock, monkeypatch, code, exception):
    monkeypatch.setattr(
        requests, "put", lambda *a, **k: SimpleNamespace(status_code=code, headers={}, close=lambda: None)
    )
    with pytest.raises(exception):
        send(session, file, clock)


def test_412_goes_to_complete(session, file, clock, monkeypatch):
    monkeypatch.setattr(
        requests, "put", lambda *a, **k: SimpleNamespace(status_code=412, headers={}, close=lambda: None)
    )
    assert sum(send(session, file, clock)) == file.stat().st_size


@pytest.mark.parametrize("held", ["bad", "bytes=1-20", "bytes=0-9999999"])
def test_bad_ranges_are_refused(session, file, clock, monkeypatch, held):
    monkeypatch.setattr(
        requests, "put", lambda *a, **k: SimpleNamespace(status_code=308, headers={"Range": held}, close=lambda: None)
    )
    with pytest.raises(resume.TransferFailed):
        send(session, file, clock)


def test_thirty_minutes_without_acknowledged_bytes_pauses_without_credentials(session, file, clock, monkeypatch):
    def lost(*a, **kw):
        raise requests.ConnectionError(session["upload_url"])

    monkeypatch.setattr(requests, "put", lost)
    with pytest.raises(resume.UploadPaused) as error:
        send(session, file, clock)
    assert error.value.args == (0,)
    assert "SESSION" not in str(error.value)
    assert clock[0]() == resume.STALL_SECONDS


def test_a_changed_partial_block_sends_none_of_its_bytes(file):
    digests = file_digests(file, block_size=4096)
    original = file.read_bytes()
    file.write_bytes(b"X" + original[1:])
    with pytest.raises(uploads._FileChanged):
        uploads._read_range(file, digests, 123, 10)


def test_shortened_file_is_refused(file):
    digests = file_digests(file, block_size=4096)
    file.write_bytes(b"x")
    with pytest.raises(uploads._FileChanged):
        uploads._read_range(file, digests, 123, 10)


def test_state_is_private_atomic_and_round_trips(tmp_path, session):
    store = resume.SessionStore(tmp_path / "sessions", {"path": "model", "hash": "hash"})
    identity = {"account": "user", "workspace": "ws"}
    assert store.save(session, identity)
    assert store.load(identity) == session
    if os.name != "nt":
        assert store.path.stat().st_mode & 0o777 == 0o600
        assert store.directory.stat().st_mode & 0o777 == 0o700
    assert not list(store.directory.glob(".upload-*"))
    store.clear()
    assert store.load(identity) is None


@pytest.mark.parametrize("field", ["server", "path", "size", "mtime_ns", "hash", "folder"])
def test_changed_file_or_destination_never_reuses_state(tmp_path, session, field):
    binding = {
        "server": "https://example/api",
        "path": "model",
        "size": 10,
        "mtime_ns": 123,
        "hash": "a",
        "folder": "loras",
    }
    identity = {"account": "user", "workspace": "ws"}
    store = resume.SessionStore(tmp_path / "sessions", binding)
    assert store.save(session, identity)
    changed = {**binding, field: "different"}
    assert resume.SessionStore(store.directory, changed).load(identity) is None


@pytest.mark.parametrize(
    "identity", [None, {"account": "other", "workspace": "ws"}, {"account": "user", "workspace": "other"}]
)
def test_unknown_or_different_identity_never_reuses_state(tmp_path, session, identity):
    store = resume.SessionStore(tmp_path / "sessions", {})
    assert store.save(session, {"account": "user", "workspace": "ws"})
    assert store.load(identity) is None


def test_expired_and_old_state_is_pruned(tmp_path, session, monkeypatch):
    store = resume.SessionStore(tmp_path / "sessions", {})
    identity = {"account": "user", "workspace": "ws"}
    assert store.save(session, identity)
    now = time.time()
    monkeypatch.setattr(resume.time, "time", lambda: now + resume.LIFETIME + 1)
    assert store.load(identity) is None
    assert list(store.directory.glob("*.json")) == []


def test_directory_is_bounded_and_invalid_json_is_ignored(tmp_path, session):
    identity = {"account": "user", "workspace": "ws"}
    for i in range(resume._MAX_ENTRIES + 5):
        store = resume.SessionStore(tmp_path / "sessions", {"path": str(i)})
        assert store.save(session, identity)
    assert len(list(store.directory.glob("*.json"))) == resume._MAX_ENTRIES
    store.path.write_text("{broken")
    assert store.load(identity) is None


def test_state_never_follows_a_symlink(tmp_path, session):
    actual = tmp_path / "actual"
    actual.mkdir()
    link = tmp_path / "link"
    try:
        link.symlink_to(actual, target_is_directory=True)
    except OSError:
        pytest.skip("symlink permission unavailable")
    store = resume.SessionStore(link, {})
    assert not store.save(session, {"account": "user", "workspace": "ws"})
    assert list(actual.iterdir()) == []


def test_corrupt_newer_session_does_not_hide_valid_session(tmp_path, session):
    identity = {"account": "user", "workspace": "ws"}
    first = resume.SessionStore(tmp_path / "sessions", {"path": "model"})
    second = resume.SessionStore(first.directory, first.binding)
    assert first.save(session, identity)
    assert second.save(session, identity)
    second.path.write_text("{broken")
    now = time.time()
    os.utime(first.path, (now - 10, now - 10))
    os.utime(second.path, (now, now))
    loaded = resume.SessionStore(first.directory, first.binding)
    assert loaded.load(identity) == session
    assert loaded.path == first.path


def test_unknown_identity_is_never_persisted(tmp_path, session):
    store = resume.SessionStore(tmp_path / "sessions", {})
    assert not store.save(session, None)
    assert not store.directory.exists()


def test_state_expiry_cannot_be_extended_by_a_future_server_expiry(tmp_path, session, monkeypatch):
    store = resume.SessionStore(tmp_path / "sessions", {})
    identity = {"account": "user", "workspace": "ws"}
    assert store.save({**session, "expires_at": "2099-01-01T00:00:00Z"}, identity)
    saved = json.loads(store.path.read_text())
    saved["created_at"] = time.time() - resume.LIFETIME - 1
    store.path.write_text(json.dumps(saved))
    assert store.load(identity) is None


@pytest.mark.parametrize("piece", [None, 0, True, 123, 128 * 1024**2])
def test_invalid_piece_size_is_refused_before_sending(session, file, clock, monkeypatch, piece):
    bucket = Bucket()
    monkeypatch.setattr(requests, "put", bucket)
    with pytest.raises(resume.TransferFailed):
        send({**session, "chunk_size_bytes": piece}, file, clock)
    assert bucket.calls == []


def test_range_cannot_go_backwards(session, file, clock, monkeypatch):
    responses = iter(["bytes=0-262143", "bytes=0-5"])
    monkeypatch.setattr(
        requests,
        "put",
        lambda *a, **k: SimpleNamespace(status_code=308, headers={"Range": next(responses)}, close=lambda: None),
    )
    with pytest.raises(resume.TransferFailed):
        send(session, file, clock)


@pytest.mark.skipif(os.name == "nt", reason="POSIX permission bits")
def test_world_readable_state_is_not_reused(tmp_path, session):
    store = resume.SessionStore(tmp_path / "sessions", {})
    identity = {"account": "user", "workspace": "ws"}
    assert store.save(session, identity)
    store.path.chmod(0o644)
    assert store.load(identity) is None


def test_failed_atomic_save_keeps_the_previous_session(tmp_path, session, monkeypatch):
    store = resume.SessionStore(tmp_path / "sessions", {})
    identity = {"account": "user", "workspace": "ws"}
    assert store.save(session, identity)

    def fail(*a):
        raise OSError("disk full")

    monkeypatch.setattr(resume.os, "replace", fail)
    assert not store.save({**session, "upload_id": "other"}, identity)
    assert store.load(identity) == session
    assert not list(store.directory.glob(".upload-*"))


@pytest.mark.parametrize("url", [1, True, {}, [], b"https://bucket.example/upload"])
def test_invalid_url_types_are_refused(session, url):
    assert not resume.valid_start({**session, "upload_url": url})


@pytest.mark.parametrize("url", [1, True, {}, [], "http://bucket.example/upload"])
def test_malformed_saved_url_is_ignored(tmp_path, session, url):
    store = resume.SessionStore(tmp_path / "sessions", {})
    identity = {"account": "user", "workspace": "ws"}
    assert store.save(session, identity)
    saved = json.loads(store.path.read_text())
    saved["start"]["upload_url"] = url
    store.path.write_text(json.dumps(saved))
    assert resume.SessionStore(store.directory, {}).load(identity) is None


def test_overlapping_uploads_keep_separate_sessions(tmp_path, session):
    identity = {"account": "user", "workspace": "ws"}
    first = resume.SessionStore(tmp_path / "sessions", {"path": "model", "hash": "hash"})
    second = resume.SessionStore(first.directory, first.binding)
    assert first.load(identity) is None
    assert second.load(identity) is None
    other_session = {**session, "upload_id": "upload-2", "upload_url": "https://bucket.example/other"}
    assert first.save(session, identity)
    assert second.save(other_session, identity)
    assert first.path != second.path
    first.clear()
    rerun = resume.SessionStore(first.directory, first.binding)
    assert rerun.load(identity) == other_session
    assert rerun.path == second.path
    rerun.clear()
    assert not list(first.directory.glob("*.json"))


@pytest.mark.parametrize(
    "url", ["http://bucket.example/upload", "https:///missing-host", "https://user:pass@bucket.example/upload"]
)
def test_resumable_addresses_still_require_https_without_userinfo(session, url):
    assert not resume.valid_start({**session, "upload_url": url})


@pytest.mark.parametrize("setting", ["REQUESTS_CA_BUNDLE", "CURL_CA_BUNDLE"])
@pytest.mark.parametrize(
    "contents", ["not a certificate\n", "-----BEGIN CERTIFICATE-----\ninvalid\n-----END CERTIFICATE-----\n"]
)
def test_existing_invalid_ca_bundle_is_fatal_with_real_requests(tmp_path, monkeypatch, setting, contents):
    """Exercise Requests, urllib3 and OpenSSL without contacting a remote service."""
    import socket

    bundle = tmp_path / "invalid-ca.pem"
    bundle.write_text(contents)
    monkeypatch.delenv("REQUESTS_CA_BUNDLE", raising=False)
    monkeypatch.setenv(setting, str(bundle))
    monkeypatch.setenv("NO_PROXY", "127.0.0.1")
    monkeypatch.setenv("no_proxy", "127.0.0.1")
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        listener.listen(1)
        start = {"upload_url": f"https://127.0.0.1:{listener.getsockname()[1]}/?secret=SESSION", "headers": {}}
        with pytest.raises(resume.TlsSetupFailed) as error:
            resume._request(start, b"", "bytes */16")
        assert "SESSION" not in str(error.value)


@pytest.mark.parametrize("certificate", [False, True], ids=["transient-eof", "certificate"])
def test_https_proxy_error_uses_real_requests_adapter(session, monkeypatch, certificate):
    """Keep Requests' proxy exception translation; replace only the network call."""
    import ssl

    from urllib3.connectionpool import HTTPSConnectionPool
    from urllib3.exceptions import MaxRetryError, ProxyError

    calls = []

    def fail(pool, method, url, **kwargs):
        calls.append((method, url))
        reason = ssl.SSLCertVerificationError("certificate verify failed") if certificate else ssl.SSLEOFError("EOF")
        raise MaxRetryError(pool, url, ProxyError("proxy failed", reason))

    monkeypatch.setattr(HTTPSConnectionPool, "urlopen", fail)
    if certificate:
        with pytest.raises(resume.TlsSetupFailed) as error:
            resume._request(session, b"", "bytes */16")
        assert "SESSION" not in str(error.value)
    else:
        assert resume._request(session, b"", "bytes */16") == (None, None)
    assert len(calls) == 1
