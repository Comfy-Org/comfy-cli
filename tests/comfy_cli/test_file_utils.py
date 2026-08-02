import os
import stat
import sys
import zipfile

import pytest

from comfy_cli import file_utils
from comfy_cli.file_utils import atomic_write_bytes, atomic_write_text


def test_zip_files_respects_comfyignore(tmp_path, monkeypatch):
    project_dir = tmp_path
    (project_dir / "keep.txt").write_text("keep", encoding="utf-8")
    (project_dir / "ignore.log").write_text("ignore", encoding="utf-8")
    ignored_dir = project_dir / "ignored_dir"
    ignored_dir.mkdir()
    (ignored_dir / "nested.txt").write_text("nested", encoding="utf-8")

    (project_dir / ".comfyignore").write_text("*.log\nignored_dir/\n", encoding="utf-8")

    zip_path = project_dir / "node.zip"

    monkeypatch.chdir(project_dir)
    monkeypatch.setattr(
        file_utils,
        "list_git_tracked_files",
        lambda base_path=".": [
            "keep.txt",
            "ignore.log",
            "ignored_dir/nested.txt",
        ],
    )

    file_utils.zip_files(str(zip_path))

    with zipfile.ZipFile(zip_path, "r") as zf:
        names = set(zf.namelist())

    assert "keep.txt" in names
    assert "ignore.log" not in names
    assert not any(name.startswith("ignored_dir/") for name in names)


def test_zip_files_force_include_overrides_ignore(tmp_path, monkeypatch):
    project_dir = tmp_path
    include_dir = project_dir / "include_me"
    include_dir.mkdir()
    (include_dir / "data.json").write_text("{}", encoding="utf-8")

    (project_dir / "other.txt").write_text("ok", encoding="utf-8")
    (project_dir / ".comfyignore").write_text("include_me/\n", encoding="utf-8")

    zip_path = project_dir / "node.zip"

    monkeypatch.chdir(project_dir)
    monkeypatch.setattr(
        file_utils,
        "list_git_tracked_files",
        lambda base_path=".": [
            "other.txt",
            "include_me/data.json",
        ],
    )

    file_utils.zip_files(str(zip_path), includes=["include_me"])

    with zipfile.ZipFile(zip_path, "r") as zf:
        names = set(zf.namelist())

    assert "include_me/data.json" in names
    assert "other.txt" in names


def test_zip_files_without_git_falls_back_to_walk(tmp_path, monkeypatch):
    project_dir = tmp_path
    (project_dir / "file.txt").write_text("data", encoding="utf-8")
    zip_path = project_dir / "node.zip"

    monkeypatch.chdir(project_dir)
    monkeypatch.setattr(file_utils, "list_git_tracked_files", lambda base_path=".": [])

    file_utils.zip_files(str(zip_path))

    with zipfile.ZipFile(zip_path, "r") as zf:
        names = set(zf.namelist())

    assert "file.txt" in names
    assert "node.zip" not in names


def test_atomic_write_text_creates_file_and_parents(tmp_path):
    target = tmp_path / "sub" / "dir" / "out.json"
    atomic_write_text(target, '{"a": 1}')

    assert target.read_text(encoding="utf-8") == '{"a": 1}'
    # No stray tmp files left behind.
    assert list(target.parent.glob("*.tmp")) == []


def test_atomic_write_text_overwrites_existing(tmp_path):
    target = tmp_path / "out.txt"
    target.write_text("old", encoding="utf-8")

    atomic_write_text(target, "new")

    assert target.read_text(encoding="utf-8") == "new"


def test_atomic_write_text_fsync_true_still_writes(tmp_path, monkeypatch):
    target = tmp_path / "durable.txt"

    # Spy on os.fsync so we assert it's actually invoked, not silently no-op'd
    # (the class of bug flagged for the Windows O_RDONLY case). Delegate to the
    # real fsync so durability behavior is preserved.
    real_fsync = file_utils.os.fsync
    synced_fds = []

    def spy_fsync(fd):
        synced_fds.append(fd)
        return real_fsync(fd)

    monkeypatch.setattr(file_utils.os, "fsync", spy_fsync)

    atomic_write_text(target, "durable", fsync=True)

    assert target.read_text(encoding="utf-8") == "durable"
    assert list(target.parent.glob("*.tmp")) == []
    # The tmp file fd is always fsynced (O_RDWR, works cross-platform). On POSIX
    # the parent directory fd is fsynced too; Windows can't open a dir for fsync
    # and best-effort skips it, so only require the extra sync off Windows.
    if sys.platform == "win32":
        assert len(synced_fds) >= 1
    else:
        assert len(synced_fds) == 2


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX file-mode semantics")
def test_atomic_write_text_preserves_existing_mode(tmp_path):
    # A first atomic write must not clobber the destination's mode down to mkstemp's
    # hardcoded 0600. We only need a non-0600 mode to prove restoration happens, so
    # use owner-execute (0o700) — a bit mkstemp's 0600 lacks — rather than a
    # group/other-readable mode. That keeps this clear of the py/overly-permissive-file
    # scanner while still exercising the exact "restore bits beyond 0600" path.
    target = tmp_path / "shared.json"
    target.write_text("old", encoding="utf-8")
    os.chmod(target, 0o700)

    atomic_write_text(target, "new")

    assert stat.S_IMODE(os.stat(target).st_mode) == 0o700
    assert target.read_text(encoding="utf-8") == "new"


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX file-mode semantics")
def test_atomic_write_text_new_file_uses_umask_default(tmp_path):
    # A new destination gets the umask-derived default, not mkstemp's hardcoded 0600.
    old_umask = os.umask(0o022)
    try:
        target = tmp_path / "fresh.json"
        atomic_write_text(target, "data")
        assert stat.S_IMODE(os.stat(target).st_mode) == (0o666 & ~0o022)
    finally:
        os.umask(old_umask)


def test_atomic_write_text_cleans_up_tmp_on_failure(tmp_path, monkeypatch):
    target = tmp_path / "out.txt"
    target.write_text("original", encoding="utf-8")

    def boom(src, dst):
        raise OSError("replace failed")

    # Fail at the rename step, after the tmp file has been written.
    monkeypatch.setattr(file_utils.os, "replace", boom)

    with pytest.raises(OSError):
        atomic_write_text(target, "new content")

    # The tmp file is cleaned up and the destination is untouched.
    assert list(target.parent.glob("*.tmp")) == []
    assert target.read_text(encoding="utf-8") == "original"


def test_atomic_write_text_tmp_name_is_unique_per_write(tmp_path, monkeypatch):
    # Two writes from the "same" pid must not collide on a shared tmp path.
    target = tmp_path / "out.txt"
    seen = []
    real_mkstemp = file_utils.tempfile.mkstemp

    def spy(*args, **kwargs):
        fd, name = real_mkstemp(*args, **kwargs)
        seen.append(name)
        return fd, name

    monkeypatch.setattr(file_utils.tempfile, "mkstemp", spy)
    monkeypatch.setattr(file_utils.os, "getpid", lambda: 4242)

    atomic_write_text(target, "a")
    atomic_write_text(target, "b")

    assert len(seen) == 2
    assert seen[0] != seen[1]
    assert target.read_text(encoding="utf-8") == "b"


def test_atomic_write_text_does_not_follow_symlinked_tmp(tmp_path):
    # A pre-planted symlink at a predictable tmp path must not redirect the write.
    target = tmp_path / "out.txt"
    victim = tmp_path / "victim.txt"
    victim.write_text("do-not-touch", encoding="utf-8")
    # The old scheme used "<dest>.<pid>.tmp"; plant a symlink there.
    import os as _os

    decoy = tmp_path / f"out.txt.{_os.getpid()}.tmp"
    decoy.symlink_to(victim)

    atomic_write_text(target, "new")

    assert target.read_text(encoding="utf-8") == "new"
    assert victim.read_text(encoding="utf-8") == "do-not-touch"


# --- atomic_write_bytes -----------------------------------------------------
# The bytes twin shares one private implementation with atomic_write_text, so
# these mirror the text-variant cases over the seam that differs: the payload is
# handed through verbatim, with no encode and no newline translation.


def test_atomic_write_bytes_creates_file_and_parents(tmp_path):
    target = tmp_path / "sub" / "dir" / "index.json"
    atomic_write_bytes(target, b'{"a": 1}')

    assert target.read_bytes() == b'{"a": 1}'
    # No stray tmp files left behind.
    assert list(target.parent.glob("*.tmp")) == []


def test_atomic_write_bytes_overwrites_existing(tmp_path):
    target = tmp_path / "index.json"
    target.write_bytes(b"old")

    atomic_write_bytes(target, b"new")

    assert target.read_bytes() == b"new"


def test_atomic_write_bytes_writes_payload_verbatim(tmp_path):
    # No UTF-8 encode and no platform newline translation: non-UTF-8 bytes and
    # bare LFs survive byte-for-byte, which is the reason this variant exists.
    target = tmp_path / "raw.bin"
    payload = b"\xff\xfe\r\n\x00line\n"

    atomic_write_bytes(target, payload)

    assert target.read_bytes() == payload


def test_atomic_write_bytes_fsync_true_still_writes(tmp_path, monkeypatch):
    target = tmp_path / "durable.bin"

    real_fsync = file_utils.os.fsync
    synced_fds = []

    def spy_fsync(fd):
        synced_fds.append(fd)
        return real_fsync(fd)

    monkeypatch.setattr(file_utils.os, "fsync", spy_fsync)

    atomic_write_bytes(target, b"durable", fsync=True)

    assert target.read_bytes() == b"durable"
    assert list(target.parent.glob("*.tmp")) == []
    # Same split as the text variant: the tmp fd always syncs; the parent
    # directory fd only off Windows, which can't open a dir for fsync.
    if sys.platform == "win32":
        assert len(synced_fds) >= 1
    else:
        assert len(synced_fds) == 2


def test_atomic_write_bytes_cleans_up_tmp_on_failure(tmp_path, monkeypatch):
    target = tmp_path / "index.json"
    target.write_bytes(b"original")

    def boom(src, dst):
        raise OSError("replace failed")

    # Fail at the rename step, after the tmp file has been written.
    monkeypatch.setattr(file_utils.os, "replace", boom)

    with pytest.raises(OSError):
        atomic_write_bytes(target, b"new content")

    # The tmp file is cleaned up and the destination is untouched.
    assert list(target.parent.glob("*.tmp")) == []
    assert target.read_bytes() == b"original"


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX file-mode semantics")
def test_atomic_write_bytes_new_file_uses_umask_default(tmp_path):
    # The templates gallery cache relies on this: the old inline mkstemp left it
    # at 0600, and the switch to the umask-derived default is intended.
    old_umask = os.umask(0o022)
    try:
        target = tmp_path / "fresh.bin"
        atomic_write_bytes(target, b"data")
        assert stat.S_IMODE(os.stat(target).st_mode) == (0o666 & ~0o022)
    finally:
        os.umask(old_umask)
