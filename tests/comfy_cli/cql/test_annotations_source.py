"""Tests for node-annotation resolution (cache → fetch → stale → bundled).

The invariants under test are the ones that bit us in review: a bad upstream
body must never reach the cache, a half-fetched pair must never be committed,
the implicit hot path must never stall or repeat a doomed fetch, and the
``--input`` path must never touch the network at all.
"""

from __future__ import annotations

import os
import time

import pytest

from comfy_cli.cql import annotations_source as src

VALID_SUP = b"node_packs:\n  - name: demo-pack\n    node_labels:\n      NodeA:\n        - NetworkAccess\n"
VALID_DIS = b"disable_nodes:\n  or:\n    - NetworkAccess: true\n"
GARBAGE = b"<!DOCTYPE html><html><body>429 Too Many Requests</body></html>"


@pytest.fixture(autouse=True)
def _isolate_cache(tmp_path, monkeypatch):
    """Point the cache at a temp dir and default to network-off for safety.

    Any test that wants the network path must opt in explicitly *and* stub the
    fetch — a real request escaping into the suite would make it flaky.
    """
    monkeypatch.setattr(src, "_cache_dir", lambda: tmp_path / "comfy-complete")
    monkeypatch.setenv("COMFY_CLI_NO_REMOTE_REFRESH", "1")
    yield


@pytest.fixture
def cache_dir(tmp_path):
    d = tmp_path / "comfy-complete"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _write_pair(cache_dir, sup=VALID_SUP, dis=VALID_DIS, *, age: float = 0.0):
    for name, data in ((src._SUPPORTED_NODES, sup), (src._CLOUD_DISABLE, dis)):
        path = cache_dir / name
        path.write_bytes(data)
        if age:
            stamp = time.time() - age
            os.utime(path, (stamp, stamp))


def _allow_network(monkeypatch):
    monkeypatch.setenv("COMFY_CLI_NO_REMOTE_REFRESH", "0")


# ---------------------------------------------------------------------------
# Environment flag
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("value", ["1", "true", "TRUE", "yes", "on", "anything"])
def test_network_disabled_truthy_values(monkeypatch, value):
    monkeypatch.setenv("COMFY_CLI_NO_REMOTE_REFRESH", value)
    assert src.network_disabled() is True


@pytest.mark.parametrize("value", ["", "0", "false", "False", "FALSE", "no", "off", "  Off  "])
def test_network_disabled_falsey_values(monkeypatch, value):
    """``=FALSE`` must read as "network on", not as its opposite."""
    monkeypatch.setenv("COMFY_CLI_NO_REMOTE_REFRESH", value)
    assert src.network_disabled() is False


def test_network_disabled_unset(monkeypatch):
    monkeypatch.delenv("COMFY_CLI_NO_REMOTE_REFRESH", raising=False)
    assert src.network_disabled() is False


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "body",
    [
        GARBAGE,
        b"",
        b"node_packs: []",  # syntactically fine, semantically empty
        b"- not: a mapping",
        b"labels:\n  - OnlyLabels\n",  # right file family, missing node_packs
    ],
)
def test_invalid_supported_nodes_rejected(body):
    assert src._valid_supported_nodes(body) is False


def test_valid_supported_nodes_accepted():
    assert src._valid_supported_nodes(VALID_SUP) is True
    assert src._valid_supported_nodes(src.bundled_bytes(src._SUPPORTED_NODES)) is True


@pytest.mark.parametrize("body", [GARBAGE, b"", b"disable_nodes: []", b"other_key: {}"])
def test_invalid_cloud_disable_rejected(body):
    assert src._valid_cloud_disable(body) is False


def test_valid_cloud_disable_accepted():
    assert src._valid_cloud_disable(VALID_DIS) is True
    assert src._valid_cloud_disable(src.bundled_bytes(src._CLOUD_DISABLE)) is True
    # An empty rule set is a legitimate upstream state, not a broken file.
    assert src._valid_cloud_disable(b"disable_nodes:\n  or: []\n") is True


# ---------------------------------------------------------------------------
# load_annotation_bytes — resolution order
# ---------------------------------------------------------------------------


def test_falls_back_to_bundled_when_offline():
    """Network disabled + empty cache → the package-bundled snapshot is used."""
    sup, dis = src.load_annotation_bytes()
    assert sup is not None and b"node_packs" in sup
    assert dis is not None and b"disable_nodes" in dis


def test_fresh_cache_wins_without_network(cache_dir, monkeypatch):
    _write_pair(cache_dir)
    _allow_network(monkeypatch)
    monkeypatch.setattr(src, "fetch_pair", lambda **kw: pytest.fail("network used on a fresh cache"))

    sup, dis = src.load_annotation_bytes()
    assert sup == VALID_SUP
    assert dis == VALID_DIS


def test_allow_network_false_never_fetches(cache_dir, monkeypatch):
    """The ``--input <dump>`` path is offline by contract, stale cache or not."""
    _write_pair(cache_dir, age=src._CACHE_TTL_SECONDS + 100)
    _allow_network(monkeypatch)
    monkeypatch.setattr(src, "fetch_pair", lambda **kw: pytest.fail("network used with allow_network=False"))

    sup, dis = src.load_annotation_bytes(allow_network=False)
    assert sup == VALID_SUP  # served straight from the stale cache


def test_half_present_cache_is_not_used(cache_dir, monkeypatch):
    """One file alone can't be paired with the other generation — go bundled."""
    (cache_dir / src._SUPPORTED_NODES).write_bytes(VALID_SUP)
    sup, _ = src.load_annotation_bytes()
    assert sup == src.bundled_bytes(src._SUPPORTED_NODES)
    assert sup != VALID_SUP


def test_poisoned_cache_is_ignored_in_favour_of_bundled(cache_dir):
    """A cache entry that doesn't validate is treated as absent, not served.

    Before this, a garbage-but-fresh entry was handed to
    ``engine.parse_supported_nodes``, which degrades to "no annotations"
    silently — blanking every node's labels for a whole TTL window.
    """
    _write_pair(cache_dir, sup=GARBAGE)
    sup, _ = src.load_annotation_bytes()
    assert sup == src.bundled_bytes(src._SUPPORTED_NODES)


def test_fetch_success_writes_both_files(cache_dir, monkeypatch):
    _allow_network(monkeypatch)
    monkeypatch.setattr(
        src, "fetch_pair", lambda **kw: {src._SUPPORTED_NODES: VALID_SUP, src._CLOUD_DISABLE: VALID_DIS}
    )

    sup, dis = src.load_annotation_bytes()
    assert (sup, dis) == (VALID_SUP, VALID_DIS)
    assert (cache_dir / src._SUPPORTED_NODES).read_bytes() == VALID_SUP
    assert (cache_dir / src._CLOUD_DISABLE).read_bytes() == VALID_DIS


def test_stale_cache_used_when_fetch_fails(cache_dir, monkeypatch):
    _write_pair(cache_dir, age=src._CACHE_TTL_SECONDS + 100)
    _allow_network(monkeypatch)

    def boom(**kw):
        raise src.AnnotationFetchError("network down")

    monkeypatch.setattr(src, "fetch_pair", boom)
    sup, dis = src.load_annotation_bytes()
    assert (sup, dis) == (VALID_SUP, VALID_DIS)  # stale beats bundled


def test_failed_fetch_is_negative_cached(cache_dir, monkeypatch):
    """A doomed fetch is attempted once, not once per invocation."""
    _write_pair(cache_dir, age=src._CACHE_TTL_SECONDS + 100)
    _allow_network(monkeypatch)
    calls = []

    def boom(**kw):
        calls.append(1)
        raise src.AnnotationFetchError("network down")

    monkeypatch.setattr(src, "fetch_pair", boom)
    for _ in range(3):
        src.load_annotation_bytes()
    assert len(calls) == 1
    assert src._failure_stamp().is_file()


def test_successful_fetch_clears_the_failure_stamp(cache_dir, monkeypatch):
    _allow_network(monkeypatch)
    src._mark_failure()
    assert src._in_failure_backoff() is True

    monkeypatch.setattr(
        src, "fetch_pair", lambda **kw: {src._SUPPORTED_NODES: VALID_SUP, src._CLOUD_DISABLE: VALID_DIS}
    )
    # The backoff is honoured, so this load serves bundled without fetching...
    src.load_annotation_bytes()
    assert not (cache_dir / src._SUPPORTED_NODES).exists()

    # ...until the stamp ages out.
    stamp = src._failure_stamp()
    old = time.time() - src._FAILURE_BACKOFF - 10
    os.utime(stamp, (old, old))
    src.load_annotation_bytes()
    assert (cache_dir / src._SUPPORTED_NODES).read_bytes() == VALID_SUP
    assert not stamp.exists()


def test_cache_write_failure_still_returns_fetched_data(cache_dir, monkeypatch):
    """A read-only cache dir must not discard data we already hold."""
    _allow_network(monkeypatch)
    monkeypatch.setattr(
        src, "fetch_pair", lambda **kw: {src._SUPPORTED_NODES: VALID_SUP, src._CLOUD_DISABLE: VALID_DIS}
    )
    monkeypatch.setattr(src, "_write_atomic", lambda *a, **k: (_ for _ in ()).throw(OSError("read-only fs")))

    sup, dis = src.load_annotation_bytes()
    assert (sup, dis) == (VALID_SUP, VALID_DIS)


# ---------------------------------------------------------------------------
# fetch_pair — atomicity, validation, bounded wait
# ---------------------------------------------------------------------------


def test_fetch_pair_rejects_garbage_body_without_caching(cache_dir, monkeypatch):
    """A 200 with an HTML body fails the pair; nothing is written."""
    _allow_network(monkeypatch)

    def fake_fetch_one(filename):
        # `_fetch_one` validates before returning, so garbage raises there.
        if not src._VALIDATORS[filename](GARBAGE):
            raise src.AnnotationFetchError(f"{filename}: invalid body")
        return GARBAGE

    monkeypatch.setattr(src, "_fetch_one", fake_fetch_one)
    with pytest.raises(src.AnnotationFetchError):
        src.fetch_pair()

    src.load_annotation_bytes()
    assert not (cache_dir / src._SUPPORTED_NODES).exists()
    assert not (cache_dir / src._CLOUD_DISABLE).exists()


def test_fetch_pair_is_all_or_nothing(cache_dir, monkeypatch):
    """One half failing must not commit the other — mixed generations
    mis-compute ``cloud_disabled``."""
    _allow_network(monkeypatch)

    def half_broken(filename):
        if filename == src._CLOUD_DISABLE:
            raise src.AnnotationFetchError("500")
        return VALID_SUP

    monkeypatch.setattr(src, "_fetch_one", half_broken)
    with pytest.raises(src.AnnotationFetchError, match="500"):
        src.fetch_pair()
    assert not (cache_dir / src._SUPPORTED_NODES).exists()


def test_fetch_pair_bounded_by_deadline(monkeypatch):
    """A hung fetch is abandoned at the deadline rather than stalling the CLI."""

    def hang(filename):
        time.sleep(30)
        return VALID_SUP

    monkeypatch.setattr(src, "_fetch_one", hang)
    started = time.monotonic()
    with pytest.raises(src.AnnotationFetchError, match="timed out"):
        src.fetch_pair(deadline=0.2)
    assert time.monotonic() - started < 5.0


def test_fetch_pair_runs_concurrently(monkeypatch):
    """Both files are in flight at once, so the wait is one timeout, not two."""

    def slow(filename):
        time.sleep(0.3)
        return VALID_SUP if filename == src._SUPPORTED_NODES else VALID_DIS

    monkeypatch.setattr(src, "_fetch_one", slow)
    started = time.monotonic()
    result = src.fetch_pair(deadline=5.0)
    elapsed = time.monotonic() - started
    assert set(result) == set(src._FILES)
    assert elapsed < 0.55, f"sequential fetch would take ~0.6s, took {elapsed:.2f}s"


# ---------------------------------------------------------------------------
# refresh_annotations
# ---------------------------------------------------------------------------


def test_refresh_annotations_reports_bundled_when_disabled():
    results = src.refresh_annotations()
    assert {r["name"] for r in results} == set(src._FILES)
    assert all(r["source"] == "bundled" for r in results)
    assert all("disabled" in r["error"] for r in results)


def test_refresh_annotations_reports_remote_on_success(cache_dir, monkeypatch):
    _allow_network(monkeypatch)
    monkeypatch.setattr(
        src, "fetch_pair", lambda **kw: {src._SUPPORTED_NODES: VALID_SUP, src._CLOUD_DISABLE: VALID_DIS}
    )

    results = src.refresh_annotations()
    assert all(r["source"] == "remote" for r in results)
    assert all(r["path"] for r in results)
    assert (cache_dir / src._SUPPORTED_NODES).read_bytes() == VALID_SUP


def test_refresh_annotations_ignores_the_backoff_stamp(monkeypatch):
    """An explicit refresh is the user overriding the backoff, not obeying it."""
    _allow_network(monkeypatch)
    src._mark_failure()
    monkeypatch.setattr(
        src, "fetch_pair", lambda **kw: {src._SUPPORTED_NODES: VALID_SUP, src._CLOUD_DISABLE: VALID_DIS}
    )

    results = src.refresh_annotations()
    assert all(r["source"] == "remote" for r in results)


def test_refresh_annotations_separates_cache_errors_from_fetch_errors(monkeypatch):
    """ "Downloaded fine, couldn't save it" must not be reported as a network failure."""
    _allow_network(monkeypatch)
    monkeypatch.setattr(
        src, "fetch_pair", lambda **kw: {src._SUPPORTED_NODES: VALID_SUP, src._CLOUD_DISABLE: VALID_DIS}
    )
    monkeypatch.setattr(src, "_write_atomic", lambda *a, **k: (_ for _ in ()).throw(OSError("disk full")))

    results = src.refresh_annotations()
    assert all(r["source"] == "remote" for r in results)
    assert all(r["path"] is None for r in results)
    assert all("disk full" in r["cache_error"] for r in results)
    assert all("error" not in r for r in results)


def test_refresh_annotations_marks_failure_for_the_hot_path(monkeypatch):
    _allow_network(monkeypatch)

    def boom(**kw):
        raise src.AnnotationFetchError("dns failure")

    monkeypatch.setattr(src, "fetch_pair", boom)
    results = src.refresh_annotations()
    assert all(r["source"] == "bundled" for r in results)
    assert all("dns failure" in r["error"] for r in results)
    assert src._in_failure_backoff() is True
