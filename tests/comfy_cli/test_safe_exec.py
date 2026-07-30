"""Tests for :mod:`comfy_cli._safe_exec`.

The contract under test: ``resolve_binary`` hands back only a *trusted absolute
path*, and returns ``None`` — "skip this probe" — for every CWD-anchored match,
while never rejecting a legitimate system binary and never raising.
"""

from __future__ import annotations

import os
from unittest.mock import patch

from comfy_cli import _safe_exec


class TestResolveBinaryCwdGuard:
    """A binary that ``shutil.which`` resolves *directly inside* the current
    working directory is rejected — closing the CWD binary-planting hole. The
    guard fires on every platform (``$PATH`` can search the CWD on POSIX too via a
    ``.``/empty entry) and rejects only the immediate directory so a legitimate
    system binary in a subdirectory is never lost."""

    def test_rejects_binary_planted_in_cwd(self, tmp_path):
        planted = tmp_path / "nvidia-smi.exe"
        planted.write_text("")
        with (
            patch.object(_safe_exec.os, "getcwd", return_value=str(tmp_path)),
            patch.object(_safe_exec.shutil, "which", return_value=str(planted)),
        ):
            assert _safe_exec.resolve_binary("nvidia-smi") is None

    def test_allows_system_binary_outside_cwd(self, tmp_path):
        cwd = tmp_path / "attacker"
        system_dir = tmp_path / "System32"
        cwd.mkdir()
        system_dir.mkdir()
        legit = system_dir / "nvidia-smi.exe"
        legit.write_text("")
        with (
            patch.object(_safe_exec.os, "getcwd", return_value=str(cwd)),
            patch.object(_safe_exec.shutil, "which", return_value=str(legit)),
        ):
            assert _safe_exec.resolve_binary("nvidia-smi") == str(legit)

    def test_allows_system_binary_in_subdirectory_of_cwd(self, tmp_path):
        """Running from an ancestor of the binary (e.g. ``C:\\Windows`` with the
        real binary under ``System32``) must NOT reject it — only a binary
        directly in the CWD is a plant."""
        system_dir = tmp_path / "System32"
        system_dir.mkdir()
        legit = system_dir / "nvidia-smi.exe"
        legit.write_text("")
        with (
            # CWD is the ANCESTOR (tmp_path), binary lives one level deeper.
            patch.object(_safe_exec.os, "getcwd", return_value=str(tmp_path)),
            patch.object(_safe_exec.shutil, "which", return_value=str(legit)),
        ):
            assert _safe_exec.resolve_binary("nvidia-smi") == str(legit)

    def test_posix_style_plant_in_cwd_is_rejected(self, tmp_path):
        """A ``.``/empty entry in ``$PATH`` lets ``shutil.which`` return a CWD
        match on POSIX too, so the guard applies there as well."""
        planted = tmp_path / "nvidia-smi"
        planted.write_text("")
        with (
            patch.object(_safe_exec.os, "getcwd", return_value=str(tmp_path)),
            patch.object(_safe_exec.shutil, "which", return_value=str(planted)),
        ):
            assert _safe_exec.resolve_binary("nvidia-smi") is None

    def test_posix_allows_system_binary_outside_cwd(self, tmp_path):
        """A legitimate binary outside the CWD is returned as-is on POSIX."""
        cwd = tmp_path / "project"
        bin_dir = tmp_path / "usr_bin"
        cwd.mkdir()
        bin_dir.mkdir()
        resolved = bin_dir / "sysctl"
        resolved.write_text("")
        with (
            patch.object(_safe_exec.os, "getcwd", return_value=str(cwd)),
            patch.object(_safe_exec.shutil, "which", return_value=str(resolved)),
        ):
            assert _safe_exec.resolve_binary("sysctl") == str(resolved)


class TestResolveBinaryRejectsRelativeMatches:
    """``shutil.which`` returns ``os.path.join(entry, name)``, so a relative
    ``$PATH`` entry yields a relative match anchored in the CWD. Executing that
    string would let ``subprocess`` re-resolve it against the attacker-controlled
    CWD, so such a match is skipped rather than run."""

    def test_rejects_relative_subdirectory_match(self):
        """``PATH=subdir`` → ``subdir/nvidia-smi``: not *directly* in the CWD, so
        the planted-in-CWD guard lets it through — the absolute-path check is what
        stops it."""
        relative = os.path.join("subdir", "nvidia-smi")
        # Precondition: this is exactly the case the CWD guard does NOT catch.
        assert not _safe_exec.is_planted_in_cwd(relative)
        with patch.object(_safe_exec.shutil, "which", return_value=relative):
            assert _safe_exec.resolve_binary("nvidia-smi") is None

    def test_rejects_dot_relative_match(self):
        """Windows prepends ``os.curdir`` to the search path, so a CWD plant comes
        back as ``.\\nvidia-smi.exe``."""
        with patch.object(_safe_exec.shutil, "which", return_value=os.path.join(os.curdir, "nvidia-smi.exe")):
            assert _safe_exec.resolve_binary("nvidia-smi") is None

    def test_rejects_bare_name_match(self):
        """An empty ``$PATH`` entry joins to a bare name, which ``subprocess``
        would resolve by its own PATH/CWD search — the bare-name invocation this
        helper exists to remove."""
        with patch.object(_safe_exec.shutil, "which", return_value="nvidia-smi"):
            assert _safe_exec.resolve_binary("nvidia-smi") is None


class TestResolveBinaryNeverRaises:
    def test_missing_binary_returns_none(self):
        with patch.object(_safe_exec.shutil, "which", return_value=None):
            assert _safe_exec.resolve_binary("nvidia-smi") is None

    def test_broken_lookup_returns_none(self):
        with patch.object(_safe_exec.shutil, "which", side_effect=RuntimeError("boom")):
            assert _safe_exec.resolve_binary("nvidia-smi") is None

    def test_is_planted_in_cwd_swallows_path_errors(self):
        """An unresolvable path (e.g. a different Windows drive) is treated as
        *not* planted so a legitimate binary is never rejected."""
        with patch.object(_safe_exec.os, "getcwd", side_effect=OSError("gone")):
            assert _safe_exec.is_planted_in_cwd("/usr/bin/nvidia-smi") is False
