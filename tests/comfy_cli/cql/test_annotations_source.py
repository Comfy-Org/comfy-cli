"""Tests for node-annotation resolution (cache → fetch → stale → bundled).

The invariants under test are the ones that bit us in review: a bad upstream
body must never reach the cache, a half-fetched pair must never be committed,
the implicit hot path must never stall or repeat a doomed fetch, and the
``--input`` path must never touch the network at all.
"""

from __future__ import annotations

import json
import os
import threading
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
    """Seed the single-file cache the way `_persist_pair` would."""
    path = cache_dir / "annotations.json"
    path.write_text(
        json.dumps(
            {
                "schema": src._CACHE_SCHEMA,
                "files": {src._SUPPORTED_NODES: sup.decode(), src._CLOUD_DISABLE: dis.decode()},
            }
        )
    )
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


def test_cached_pair_is_served_without_a_yaml_parse(cache_dir, monkeypatch):
    """The pair cache is trusted as written; reading it must not re-parse YAML."""
    import yaml

    _write_pair(cache_dir)

    def boom(*a, **kw):
        raise AssertionError("yaml.safe_load must not run on a cache read")

    monkeypatch.setattr(yaml, "safe_load", boom)
    assert src._read_cached_pair(require_fresh=True) == {src._SUPPORTED_NODES: VALID_SUP, src._CLOUD_DISABLE: VALID_DIS}


def test_fetch_success_caches_the_pair(cache_dir, monkeypatch):
    _allow_network(monkeypatch)
    monkeypatch.setattr(
        src, "fetch_pair", lambda **kw: {src._SUPPORTED_NODES: VALID_SUP, src._CLOUD_DISABLE: VALID_DIS}
    )

    sup, dis = src.load_annotation_bytes()
    assert (sup, dis) == (VALID_SUP, VALID_DIS)
    cached = src._read_cached_pair(require_fresh=True)
    assert cached == {src._SUPPORTED_NODES: VALID_SUP, src._CLOUD_DISABLE: VALID_DIS}


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
    assert not src._cache_file().exists()

    # ...until the stamp ages out.
    stamp = src._failure_stamp()
    old = time.time() - src._FAILURE_BACKOFF - 10
    os.utime(stamp, (old, old))
    src.load_annotation_bytes()
    assert src._read_cached_pair(require_fresh=True) is not None
    assert not stamp.exists()


def test_failed_write_leaves_the_old_generation_intact(cache_dir, monkeypatch):
    """A disk-full mid-refresh must not leave a partly-updated cache.

    The pair rides in one file published by one `os.replace`, so there is no
    "first file committed, second didn't" state to land in — the old generation
    survives whole, and this run still uses the pair it fetched.
    """
    old_sup = b"node_packs:\n  - name: old-pack\n    node_labels: {}\n"
    old_dis = b"disable_nodes:\n  or:\n    - Stateful: true\n"
    _write_pair(cache_dir, sup=old_sup, dis=old_dis, age=src._CACHE_TTL_SECONDS + 100)
    _allow_network(monkeypatch)

    monkeypatch.setattr(src, "_stage_atomic", lambda *a, **k: (_ for _ in ()).throw(OSError("No space left")))
    monkeypatch.setattr(
        src, "fetch_pair", lambda **kw: {src._SUPPORTED_NODES: VALID_SUP, src._CLOUD_DISABLE: VALID_DIS}
    )

    sup, dis = src.load_annotation_bytes()
    assert (sup, dis) == (VALID_SUP, VALID_DIS)  # caching is best-effort
    assert src._read_cached_pair(require_fresh=False) == {
        src._SUPPORTED_NODES: old_sup,
        src._CLOUD_DISABLE: old_dis,
    }
    assert [p.name for p in cache_dir.iterdir() if p.name.endswith(".tmp")] == []


def test_interleaved_refreshes_cannot_mix_generations(cache_dir, monkeypatch):
    """Concurrent refreshes each publish a whole pair; last writer wins wholly.

    With one file per document this was a real race: process A renames its
    `supported_nodes.yaml`, B renames both of its own, then A renames its
    `cloud_disable_config.yaml` — leaving A's labels beside B's disable rules
    for a full TTL. One file, one rename, so a reader sees exactly one
    generation whichever order they land in.
    """
    gens = {
        "A": (b"node_packs:\n  - name: A\n    node_labels: {}\n", b"disable_nodes:\n  or:\n    - A: true\n"),
        "B": (b"node_packs:\n  - name: B\n    node_labels: {}\n", b"disable_nodes:\n  or:\n    - B: true\n"),
    }
    whole = [{src._SUPPORTED_NODES: sup, src._CLOUD_DISABLE: dis} for sup, dis in gens.values()]

    # Seed so readers always find something, then hammer it from both sides.
    src._persist_pair(whole[0])
    stop = threading.Event()
    observed: list[dict | None] = []
    errors: list[BaseException] = []

    def writer(tag):
        try:
            while not stop.is_set():
                src._persist_pair({src._SUPPORTED_NODES: gens[tag][0], src._CLOUD_DISABLE: gens[tag][1]})
        except BaseException as e:  # noqa: BLE001 — surfaced via `errors` below
            errors.append(e)

    def reader():
        try:
            while not stop.is_set():
                observed.append(src._read_cached_pair(require_fresh=False))
        except BaseException as e:  # noqa: BLE001
            errors.append(e)

    threads = [threading.Thread(target=writer, args=("A",)), threading.Thread(target=writer, args=("B",))]
    threads += [threading.Thread(target=reader) for _ in range(2)]
    for t in threads:
        t.start()
    time.sleep(0.4)
    stop.set()
    for t in threads:
        t.join(timeout=10.0)

    assert not errors, errors
    assert len(observed) > 50, f"reader barely ran ({len(observed)} samples) — test proves little"
    # Every sample is one whole generation. `None` is acceptable (a reader may
    # catch the file mid-replace on some platforms) — a *mixed* pair is not.
    mixed = [o for o in observed if o is not None and o not in whole]
    assert not mixed, f"observed {len(mixed)} mixed generation(s), e.g. {mixed[0]}"


def test_legacy_per_file_cache_is_ignored_and_cleaned_up(cache_dir, monkeypatch):
    """A cache written by an older build is re-fetched, not misread."""
    (cache_dir / src._SUPPORTED_NODES).write_bytes(VALID_SUP)
    (cache_dir / src._CLOUD_DISABLE).write_bytes(VALID_DIS)

    # Nothing reads the old layout, so offline resolution goes to bundled.
    sup, _ = src.load_annotation_bytes()
    assert sup == src.bundled_bytes(src._SUPPORTED_NODES)

    # And the first successful write sweeps the dead files away.
    src._persist_pair({src._SUPPORTED_NODES: VALID_SUP, src._CLOUD_DISABLE: VALID_DIS})
    assert not (cache_dir / src._SUPPORTED_NODES).exists()
    assert not (cache_dir / src._CLOUD_DISABLE).exists()
    assert src._read_cached_pair(require_fresh=True) is not None


@pytest.mark.parametrize(
    "blob",
    [
        b"not json at all",
        b'{"schema": 999, "files": {}}',
        b'{"schema": 1, "files": {"supported_nodes.yaml": "node_packs: [{name: x}]"}}',  # missing the pair
        b'{"schema": 1, "files": []}',
        b'{"schema": 1}',
        b"[]",
    ],
)
def test_malformed_cache_file_reads_as_a_miss(cache_dir, blob):
    """Any shape we don't recognise falls through to bundled, never to blank."""
    src._cache_file().write_bytes(blob)
    assert src._read_cached_pair(require_fresh=True) is None
    sup, _ = src.load_annotation_bytes()
    assert sup == src.bundled_bytes(src._SUPPORTED_NODES)


def test_cache_write_failure_still_returns_fetched_data(cache_dir, monkeypatch):
    """A read-only cache dir must not discard data we already hold."""
    _allow_network(monkeypatch)
    monkeypatch.setattr(
        src, "fetch_pair", lambda **kw: {src._SUPPORTED_NODES: VALID_SUP, src._CLOUD_DISABLE: VALID_DIS}
    )
    monkeypatch.setattr(src, "_stage_atomic", lambda *a, **k: (_ for _ in ()).throw(OSError("read-only fs")))

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
    assert not src._cache_file().exists()


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
    assert not src._cache_file().exists()


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
    """Both files are in flight at once, so the wait is one timeout, not two.

    Proved with a barrier rather than a stopwatch: each worker blocks until the
    other arrives, so the pair can only complete if both are running at the same
    time. A sequential implementation deadlocks and trips the barrier timeout,
    with no wall-clock threshold to flake under CI scheduling noise.
    """
    barrier = threading.Barrier(len(src._FILES), timeout=10.0)

    def rendezvous(filename):
        barrier.wait()
        return VALID_SUP if filename == src._SUPPORTED_NODES else VALID_DIS

    monkeypatch.setattr(src, "_fetch_one", rendezvous)
    result = src.fetch_pair(deadline=20.0)
    assert set(result) == set(src._FILES)


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
    assert src._read_cached_pair(require_fresh=True) is not None


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
    monkeypatch.setattr(src, "_stage_atomic", lambda *a, **k: (_ for _ in ()).throw(OSError("disk full")))

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


# ---------------------------------------------------------------------------
# Parsed-annotation cache
# ---------------------------------------------------------------------------


def _parsed_files(cache_dir):
    return sorted(cache_dir.glob("annotations-parsed-*.json"))


def _forbid_parse(monkeypatch):
    from comfy_cli.cql import engine

    def boom(data):
        raise AssertionError("parse_supported_nodes must not run on a cache hit")

    monkeypatch.setattr(engine, "parse_supported_nodes", boom)


def test_parsed_annotations_miss_writes_and_hit_skips_parse(cache_dir, monkeypatch):
    first = src.parsed_annotations(VALID_SUP, VALID_DIS)
    assert first == ({"NodeA": "demo-pack"}, {"NodeA": ["NetworkAccess"]}, {"NetworkAccess"})
    files = _parsed_files(cache_dir)
    assert len(files) == 1
    assert files[0].name.startswith(f"annotations-parsed-v{src._PARSED_SCHEMA}-")

    _forbid_parse(monkeypatch)
    assert src.parsed_annotations(VALID_SUP, VALID_DIS) == first


def test_parsed_annotations_corrupt_file_is_rebuilt(cache_dir):
    src.parsed_annotations(VALID_SUP, VALID_DIS)
    (path,) = _parsed_files(cache_dir)
    path.write_bytes(b"{not json")

    assert src.parsed_annotations(VALID_SUP, VALID_DIS) == (
        {"NodeA": "demo-pack"},
        {"NodeA": ["NetworkAccess"]},
        {"NetworkAccess"},
    )
    assert _parsed_files(cache_dir) == [path]
    assert json.loads(path.read_bytes())["node_pack"] == {"NodeA": "demo-pack"}


@pytest.mark.parametrize(
    "blob",
    [
        b'{"node_pack": [], "node_labels": {}, "disable_labels": []}',
        b'{"node_pack": {}, "node_labels": {}}',
        b'{"node_pack": {}, "node_labels": {}, "disable_labels": {}}',
        b'{"node_pack": {}, "node_labels": {}, "disable_labels": [["x"]]}',
        b'{"node_pack": {}, "node_labels": {"NodeA": 5}, "disable_labels": []}',
        b"[]",
        b"null",
    ],
)
def test_parsed_annotations_wrong_shape_is_a_miss(cache_dir, blob):
    src.parsed_annotations(VALID_SUP, VALID_DIS)
    (path,) = _parsed_files(cache_dir)
    path.write_bytes(blob)
    assert src.parsed_annotations(VALID_SUP, VALID_DIS)[0] == {"NodeA": "demo-pack"}


def test_parsed_annotations_new_bytes_evict_old_file(cache_dir):
    src.parsed_annotations(VALID_SUP, VALID_DIS)
    (old,) = _parsed_files(cache_dir)

    other_sup = VALID_SUP.replace(b"NodeA", b"NodeB")
    assert src.parsed_annotations(other_sup, VALID_DIS)[0] == {"NodeB": "demo-pack"}
    (new,) = _parsed_files(cache_dir)
    assert new != old
    assert not old.exists()


def test_parsed_annotations_no_cache_env_skips_disk(cache_dir, monkeypatch):
    monkeypatch.setenv("COMFY_NO_CACHE", "1")
    assert src.parsed_annotations(VALID_SUP, VALID_DIS)[0] == {"NodeA": "demo-pack"}
    assert _parsed_files(cache_dir) == []


def test_parsed_annotations_empty_input_touches_nothing(cache_dir, monkeypatch):
    _forbid_parse(monkeypatch)
    assert src.parsed_annotations(None, None) == ({}, {}, set())
    assert src.parsed_annotations(b"", b"") == ({}, {}, set())
    assert _parsed_files(cache_dir) == []


def test_parsed_annotations_survives_unwritable_dir(cache_dir, monkeypatch):
    def refuse(*a, **kw):
        raise OSError("read-only")

    monkeypatch.setattr(src, "atomic_write_text", refuse)
    assert src.parsed_annotations(VALID_SUP, VALID_DIS)[0] == {"NodeA": "demo-pack"}


def test_parsed_annotations_unserialisable_parse_result_is_returned_not_cached(cache_dir):
    """A body that passes the validators but yields a non-JSON or unsortable
    parse result must still annotate; only the disk write is skipped."""
    mixed_dis = b"disable_nodes:\n  or:\n    - NetworkAccess: true\n      1: true\n"
    assert src.parsed_annotations(VALID_SUP, mixed_dis)[2] == {"NetworkAccess", 1}
    date_sup = b"node_packs:\n  - name: 2020-01-02\n    node_labels: {NodeA: [NetworkAccess]}\n"
    assert src.parsed_annotations(date_sup, VALID_DIS)[1] == {"NodeA": ["NetworkAccess"]}
    assert _parsed_files(cache_dir) == []
