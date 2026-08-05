"""Tests for :mod:`comfy_cli._safe_exec`.

The contract under test: ``resolve_binary`` hands back only a *fully qualified,
unambiguous* path, and returns ``None`` — "skip this probe" — for every
CWD-anchored match and every resolution it cannot vet, without ever raising.
"""

from __future__ import annotations

import ntpath
import os
from unittest.mock import patch

import pytest

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

    def test_is_planted_in_cwd_fails_closed_on_path_errors(self):
        """A CWD that cannot be read (deleted out from under the process) leaves
        us unable to prove the binary sits *outside* it, so it is reported as
        planted and the caller skips the probe rather than spawning an unvetted
        path."""
        with patch.object(_safe_exec.os, "getcwd", side_effect=OSError("gone")):
            assert _safe_exec.is_planted_in_cwd("/usr/bin/nvidia-smi") is True

    def test_resolve_binary_skips_probe_when_cwd_unreadable(self):
        with (
            patch.object(_safe_exec.shutil, "which", return_value=os.path.join(os.sep, "usr", "bin", "nvidia-smi")),
            patch.object(_safe_exec.os, "getcwd", side_effect=OSError("gone")),
        ):
            assert _safe_exec.resolve_binary("nvidia-smi") is None


class TestResolveBinaryRejectsNonBareNames:
    """``shutil.which`` looks a name with a directory component up *directly*
    instead of searching ``$PATH``, handing the caller's own string back after a
    bare ``isfile``+``X_OK`` check — which would sail past both CWD guards. Such a
    name is refused before the lookup."""

    @pytest.mark.parametrize(
        "name",
        [
            "/tmp/attacker/evil",
            "attacker/evil",
            r"C:\attacker\evil.exe",
            r"..\evil.exe",
            "C:evil.exe",  # drive-relative: no separator, still not a bare name
            "",
        ],
    )
    def test_rejects_name_with_path_component(self, name):
        with patch.object(_safe_exec.shutil, "which") as which:
            assert _safe_exec.resolve_binary(name) is None
        which.assert_not_called()

    def test_accepts_bare_name(self, tmp_path):
        legit = tmp_path / "nvidia-smi"
        legit.write_text("")
        with (
            patch.object(_safe_exec.os, "getcwd", return_value=str(tmp_path / "elsewhere")),
            patch.object(_safe_exec.shutil, "which", return_value=str(legit)),
        ):
            assert _safe_exec.resolve_binary("nvidia-smi") == str(legit)


class TestResolveBinaryRequiresFullyQualifiedMatch:
    """``ntpath.isabs`` accepts a drive-less rooted path, which ``CreateProcess``
    re-resolves against the process's *current drive* — so "absolute" alone is not
    enough to call a Windows match trusted."""

    def test_drive_less_rooted_windows_path_is_not_fully_qualified(self):
        """``ntpath.isabs`` accepts this shape on the 3.10–3.12 interpreters this
        package still supports (3.13 tightened it), so the guard must not lean on
        ``isabs`` alone — it has to reject the path on every version."""
        with (
            patch.object(_safe_exec.os, "name", "nt"),
            patch.object(_safe_exec.os, "path", ntpath),
        ):
            assert _safe_exec._is_fully_qualified(r"\tools\nvidia-smi.exe") is False

    @pytest.mark.parametrize("path", [r"C:\Windows\System32\nvidia-smi.exe", r"\\host\share\nvidia-smi.exe"])
    def test_drive_and_unc_paths_are_fully_qualified(self, path):
        with (
            patch.object(_safe_exec.os, "name", "nt"),
            patch.object(_safe_exec.os, "path", ntpath),
        ):
            assert _safe_exec._is_fully_qualified(path) is True

    def test_posix_absolute_path_is_fully_qualified(self):
        assert _safe_exec._is_fully_qualified(os.path.join(os.sep, "usr", "bin", "nvidia-smi")) is True

    def test_resolve_binary_skips_drive_less_windows_match(self):
        with (
            patch.object(_safe_exec.os, "name", "nt"),
            patch.object(_safe_exec.os, "path", ntpath),
            patch.object(_safe_exec.shutil, "which", return_value=r"\tools\nvidia-smi.exe"),
        ):
            assert _safe_exec.resolve_binary("nvidia-smi") is None


class TestResolveRequiredBinary:
    """The required-binary companion: same gates, but a failure is loud.

    ``git``/``ffmpeg`` failing is fatal to the command that wanted them, so
    "return ``None`` and let the caller skip" — right for a probe — would just
    push a ``TypeError`` one frame down."""

    def test_returns_the_resolved_path(self, tmp_path):
        system_dir = tmp_path / "System32"
        system_dir.mkdir()
        legit = system_dir / "git"
        legit.write_text("")
        with (
            patch.object(_safe_exec.os, "getcwd", return_value=str(tmp_path)),
            patch.object(_safe_exec.shutil, "which", return_value=str(legit)),
        ):
            assert _safe_exec.resolve_required_binary("git") == str(legit)

    def test_raises_when_the_binary_is_absent(self):
        with patch.object(_safe_exec.shutil, "which", return_value=None):
            with pytest.raises(_safe_exec.BinaryNotFoundError, match="not found on PATH") as exc_info:
                _safe_exec.resolve_required_binary("git")
        assert exc_info.value.reason is _safe_exec.BinaryRefusal.ABSENT
        assert exc_info.value.is_absent is True
        assert exc_info.value.binary == "git"

    def test_raises_for_a_binary_planted_in_the_cwd(self, tmp_path):
        planted = tmp_path / "git"
        planted.write_text("")
        with (
            patch.object(_safe_exec.os, "getcwd", return_value=str(tmp_path)),
            patch.object(_safe_exec.shutil, "which", return_value=str(planted)),
        ):
            with pytest.raises(_safe_exec.BinaryNotFoundError, match="refusing to run") as exc_info:
                _safe_exec.resolve_required_binary("git")
        assert exc_info.value.reason is _safe_exec.BinaryRefusal.CWD_ANCHORED
        assert exc_info.value.candidate == str(planted)

    def test_a_refusal_is_not_reported_as_an_absent_binary(self, tmp_path):
        """The distinction the diagnostics hang on: a refused binary is present.

        Telling a user with a working ``git`` to go install ``git`` sends them
        the wrong way and hides the interesting part — something named ``git``
        was found sitting in the directory they ran from.
        """
        planted = tmp_path / "git"
        planted.write_text("")
        with (
            patch.object(_safe_exec.os, "getcwd", return_value=str(tmp_path)),
            patch.object(_safe_exec.shutil, "which", return_value=str(planted)),
        ):
            with pytest.raises(_safe_exec.BinaryNotFoundError) as exc_info:
                _safe_exec.resolve_required_binary("git")
        assert exc_info.value.is_absent is False
        assert "install" not in str(exc_info.value).lower().split("make sure")[0]
        assert str(planted) in str(exc_info.value)

    def test_an_unplaceable_match_is_reported_as_unverifiable(self, tmp_path):
        """A deleted/unreadable CWD is refused too, but it is not evidence of a
        plant, so it gets its own reason rather than borrowing the plant's."""
        legit = tmp_path / "System32" / "git"
        legit.parent.mkdir()
        legit.write_text("")
        with (
            patch.object(_safe_exec.os, "getcwd", side_effect=OSError("cwd deleted")),
            patch.object(_safe_exec.shutil, "which", return_value=str(legit)),
        ):
            with pytest.raises(_safe_exec.BinaryNotFoundError) as exc_info:
                _safe_exec.resolve_required_binary("git")
        assert exc_info.value.reason is _safe_exec.BinaryRefusal.UNVERIFIABLE
        assert exc_info.value.is_absent is False

    def test_raises_for_a_non_bare_name(self):
        """The gates are shared with ``resolve_binary`` — a name carrying a path
        component is refused here too, rather than looked up."""
        with pytest.raises(_safe_exec.BinaryNotFoundError):
            _safe_exec.resolve_required_binary("./git")

    @pytest.mark.parametrize("caught_as", [FileNotFoundError, OSError, RuntimeError])
    def test_is_catchable_by_the_existing_degradation_handlers(self, caught_as):
        """Call sites that already tolerated a missing binary catch
        ``FileNotFoundError``/``OSError``; the ``RuntimeError`` base is there for
        callers that want to name the failure without an OS-error class. Both
        must hold, or converting a tolerant site would silently turn a graceful
        degradation into a crash."""
        with patch.object(_safe_exec.shutil, "which", return_value=None):
            with pytest.raises(caught_as):
                _safe_exec.resolve_required_binary("git")
