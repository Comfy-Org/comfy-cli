"""Phase 4: local secret store behavior."""

from __future__ import annotations

import json
import os
import stat
import sys

import pytest

from comfy_cli.auth import store as auth_store


@pytest.fixture
def isolated_secrets(tmp_path, monkeypatch):
    """Point the store at a tmp directory so we don't touch the real user config."""
    monkeypatch.setattr(auth_store, "secrets_path", lambda: tmp_path / "secrets.json")
    yield tmp_path / "secrets.json"


def test_list_is_empty_when_no_store(isolated_secrets):
    assert auth_store.list_records() == []


def test_set_creates_file_and_returns_record(isolated_secrets):
    rec = auth_store.set("comfy-cloud", "sk-test-1234567890")
    assert rec.provider == "comfy-cloud"
    assert rec.key == "sk-test-1234567890"
    assert isolated_secrets.exists()
    data = json.loads(isolated_secrets.read_text())
    assert data["providers"]["comfy-cloud"]["key"] == "sk-test-1234567890"


def test_set_redacts_in_to_dict(isolated_secrets):
    rec = auth_store.set("civitai", "abcdefgh12345678extra")
    redacted = rec.to_dict()
    assert redacted["key"] == "abcd…xtra"
    assert redacted["key_redacted"] is True


def test_set_redacts_short_keys(isolated_secrets):
    rec = auth_store.set("foo", "short")
    assert rec.to_dict()["key"] == "***"


def test_set_overwrites_existing(isolated_secrets):
    auth_store.set("civitai", "first")
    auth_store.set("civitai", "second")
    assert auth_store.get("civitai").key == "second"


def test_get_returns_none_when_absent(isolated_secrets):
    assert auth_store.get("nope") is None


def test_remove_returns_true_only_when_existed(isolated_secrets):
    auth_store.set("civitai", "k")
    assert auth_store.remove("civitai") is True
    assert auth_store.remove("civitai") is False


def test_list_is_sorted(isolated_secrets):
    auth_store.set("zeta", "longerthan8chars")
    auth_store.set("alpha", "alsolonger12345")
    names = [r.provider for r in auth_store.list_records()]
    assert names == ["alpha", "zeta"]


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX permissions only")
def test_secrets_file_is_owner_only(isolated_secrets):
    auth_store.set("comfy-cloud", "this-is-a-long-key")
    mode = stat.S_IMODE(os.stat(isolated_secrets).st_mode)
    assert mode == 0o600


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX permissions only")
def test_secrets_file_mode_under_permissive_umask(isolated_secrets):
    """Mode must be 0o600 from inception, even if the umask is 0o000.

    Regression for the TOCTOU window where a stat-after-write could observe a
    world-readable file before the chmod fired.
    """
    old_umask = os.umask(0o000)
    try:
        auth_store.set("civitai", "another-long-key-123")
    finally:
        os.umask(old_umask)
    mode = stat.S_IMODE(os.stat(isolated_secrets).st_mode)
    assert mode == 0o600


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX permissions only")
def test_secrets_parent_dir_is_owner_only(isolated_secrets):
    auth_store.set("civitai", "another-long-key-123")
    parent_mode = stat.S_IMODE(os.stat(isolated_secrets.parent).st_mode)
    assert parent_mode == 0o700


def test_concurrent_tmp_filenames_do_not_collide(isolated_secrets, monkeypatch):
    """Two writers landing in the same nanosecond must not corrupt each other.

    Simulated by stubbing os.replace to capture the tmp path each call uses.
    """
    seen_tmps = []
    real_replace = os.replace

    def spy_replace(src, dst):
        seen_tmps.append(src)
        return real_replace(src, dst)

    monkeypatch.setattr(auth_store.os, "replace", spy_replace)
    auth_store.set("alpha", "alpha-long-key-aa")
    auth_store.set("beta", "beta-long-key-bb")
    assert len(set(seen_tmps)) == 2  # distinct tmp paths per write


def test_corrupt_store_does_not_explode(isolated_secrets):
    isolated_secrets.parent.mkdir(parents=True, exist_ok=True)
    isolated_secrets.write_text("{ not json")
    # Treated as empty until something is written.
    assert auth_store.list_records() == []
    auth_store.set("civitai", "abcdefgh")
    assert auth_store.get("civitai").key == "abcdefgh"


def test_empty_key_is_rejected(isolated_secrets):
    with pytest.raises(ValueError):
        auth_store.set("civitai", "")
