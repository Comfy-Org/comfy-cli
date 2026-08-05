"""Shared helpers for spawning trusted system binaries.

Windows' ``CreateProcess`` searches the *current working directory* before
``$PATH``, so invoking a probe by bare name (``["nvidia-smi"]``) from an
attacker-prepared directory executes whatever ``nvidia-smi.exe`` was planted
there. :func:`resolve_binary` closes that vector by resolving the name to a
trusted absolute path up front, and returning ``None`` — "skip this probe" —
whenever the only match is anchored in the CWD.

The module is a leaf on purpose: it imports nothing from ``comfy_cli``, so both
:mod:`comfy_cli.hardware` and :mod:`comfy_cli.cuda_detect` can use it without
the import cycle that would come from ``cuda_detect`` importing ``hardware``
(``hardware`` already imports ``cuda_detect``).

Two entry points share one set of gates:

* :func:`resolve_binary` — for *optional* probes. **Never raises**; every failure
  (a missing binary, a broken ``$PATH`` lookup, an unresolvable path) degrades to
  ``None`` so callers keep their existing degrade-to-``None`` behaviour.
* :func:`resolve_required_binary` — for binaries a command cannot run without
  (``git``, ``ffmpeg``). Raises :class:`BinaryNotFoundError` with an actionable
  message instead of returning ``None``. It is a thin wrapper over
  :func:`resolve_binary_detail`, so the CWD/qualification gates cannot drift
  apart.

Both are built on :func:`resolve_binary_detail`, which reports *why* a name was
refused (:class:`BinaryRefusal`). "Absent from ``$PATH``" and "present but
refused because it resolved into the CWD" are very different events — one is a
missing install, the other is a possible planted binary — and collapsing them
into one message tells a user with a perfectly good ``git`` to go install ``git``.
Callers that surface the failure should branch on
:attr:`BinaryNotFoundError.reason`.
"""

from __future__ import annotations

import enum
import logging
import ntpath
import os
import shutil
from typing import NamedTuple

logger = logging.getLogger(__name__)

_PATH_SEPARATORS = ("/", "\\")


class BinaryRefusal(enum.Enum):
    """Why :func:`resolve_binary_detail` declined to hand back a path."""

    ABSENT = "absent"
    """``shutil.which`` found no match at all — the binary is not installed."""

    NOT_BARE_NAME = "not_bare_name"
    """The caller passed something carrying a path component; see
    :func:`_is_bare_name`. A programming error, not a user-fixable condition."""

    CWD_ANCHORED = "cwd_anchored"
    """A match was found, but it is anchored in the current working directory
    (a relative ``$PATH`` entry, or an absolute entry that *is* the CWD). The
    binary may well be legitimate — running from a directory that is itself a
    ``$PATH`` entry such as ``/usr/bin`` lands here — but it cannot be
    distinguished from a plant, so it is refused."""

    UNVERIFIABLE = "unverifiable"
    """A match was found but could not be placed relative to the CWD (a deleted
    or unreadable CWD, an unresolvable path). Refused fail-closed."""


class BinaryResolution(NamedTuple):
    """What :func:`resolve_binary_detail` concluded about one name."""

    path: str | None
    """The trusted absolute path, or ``None`` if the name was refused."""

    reason: BinaryRefusal | None
    """``None`` on success; otherwise why the name was refused."""

    candidate: str | None
    """The path ``shutil.which`` returned, even when it was then refused — so a
    diagnostic can name the file it declined to run. ``None`` when the lookup
    never got that far."""


class BinaryNotFoundError(RuntimeError, FileNotFoundError):
    """A binary a command cannot run without could not be trusted-resolved.

    Raised by :func:`resolve_required_binary` — either the binary is absent from
    ``$PATH``, or the only match was CWD-anchored and therefore refused.
    :attr:`reason` says which, so callers can render an accurate diagnostic
    instead of always claiming the binary is not installed; :attr:`candidate`
    carries the path that was refused, when there was one.

    It deliberately subclasses **both** ``RuntimeError`` and
    ``FileNotFoundError``. ``FileNotFoundError`` is the exception ``subprocess``
    already raises today when a bare-name spawn finds no such binary, so every
    call site that already degrades on a missing binary (``except
    (subprocess.SubprocessError, FileNotFoundError)`` in
    :mod:`comfy_cli.file_utils`, ``except OSError`` in
    :mod:`comfy_cli.command.outdated`, …) keeps degrading *identically* rather
    than starting to crash. ``RuntimeError`` is carried for callers that want to
    name the failure explicitly without reaching for an OS-error class.
    """

    def __init__(
        self,
        message: str,
        *,
        binary: str,
        reason: BinaryRefusal,
        candidate: str | None = None,
    ):
        super().__init__(message)
        self.binary = binary
        self.reason = reason
        self.candidate = candidate

    @property
    def is_absent(self) -> bool:
        """``True`` only when the binary is genuinely not installed.

        Everything else means a match *was* found and then refused — the
        distinction callers need before telling a user to install something they
        already have.
        """
        return self.reason is BinaryRefusal.ABSENT


def _is_bare_name(name: str) -> bool:
    """Return ``True`` if ``name`` is a plain binary name with no path part.

    :func:`shutil.which` short-circuits its ``$PATH`` search when the name holds
    a directory component — it looks that path up directly — so a caller-supplied
    path would sail through both CWD guards below. Both separators and a Windows
    drive prefix (``C:nvidia-smi`` is *drive-relative*) are checked on every
    platform, because the caller's string is not necessarily native to the host.
    """
    return bool(name) and not any(sep in name for sep in _PATH_SEPARATORS) and not ntpath.splitdrive(name)[0]


def _is_fully_qualified(path: str) -> bool:
    """Return ``True`` if ``path`` names one unambiguous location.

    :func:`os.path.isabs` is not the same thing as "fully qualified" on Windows:
    ``ntpath.isabs(r"\\tools\\nvidia-smi.exe")`` is ``True`` on Python ≤ 3.12, yet
    ``CreateProcess`` re-resolves such a drive-less rooted path against the
    process's *current drive*. Requiring a drive (or a UNC share) makes the
    "trusted absolute path" assumption actually hold. POSIX has no drives, so the
    extra requirement applies only where it means something.
    """
    if not os.path.isabs(path):
        return False
    if os.name == "nt":
        return bool(os.path.splitdrive(path)[0])
    return True


def is_planted_in_cwd(path: str) -> bool:
    """Return ``True`` only if ``path`` resolves to a file sitting *directly* in
    ``os.getcwd()`` — the signature of a planted probe binary.

    ``shutil.which`` searches the current directory first on Windows (and on any
    platform whose ``$PATH`` contains ``.`` or an empty entry), so an attacker who
    controls the directory the user runs ``comfy`` from can drop a malicious
    ``nvidia-smi.exe`` there. Those relative-``$PATH``-entry matches come back as
    *relative* paths and are rejected by :func:`resolve_binary` directly; this
    guard covers the remaining shape, an absolute ``$PATH`` entry that happens to
    be the CWD (``PATH="$(pwd):$PATH"`` build wrappers, and Windows' implicit
    current-directory search). Only the binary's immediate parent is compared, so
    a legitimate system binary in a *subdirectory* — e.g. ``System32`` even when
    the CWD is ``C:\\Windows`` — is left untouched. Paths are compared with
    :func:`os.path.normcase` so Windows' case-insensitivity can't fail the guard
    open.

    A resolution error (an unreadable/deleted CWD, an unresolvable path) means we
    cannot prove the binary is *outside* the CWD, so it is reported as planted:
    the caller then skips the probe, which is the same degradation as the binary
    being absent. Failing open here would be the module's only error path that
    hands an unvetted string to :mod:`subprocess`.

    Callers that need to *tell those two apart* — "definitely in the CWD" vs "we
    could not look" — should use :func:`_place_relative_to_cwd`, which keeps them
    separate; this wrapper collapses both to the refusing answer.
    """
    return _place_relative_to_cwd(path) is not False


def _place_relative_to_cwd(path: str) -> bool | None:
    """Tri-state companion to :func:`is_planted_in_cwd`.

    ``True`` — ``path``'s immediate parent *is* the CWD. ``False`` — it
    demonstrably is not. ``None`` — the comparison could not be made at all
    (deleted CWD, unresolvable path). Both ``True`` and ``None`` are refusals,
    but only ``True`` is evidence of a plant, and the two deserve different
    messages.
    """
    try:
        cwd = os.path.normcase(os.path.realpath(os.getcwd()))
        # ``os.path.dirname`` of a bare/relative ``which`` result is "", which
        # ``realpath`` correctly resolves against the CWD.
        parent = os.path.normcase(os.path.realpath(os.path.dirname(path)))
    except (OSError, ValueError):
        logger.debug("cannot place %r relative to the CWD; treating as planted", path, exc_info=True)
        return None
    return parent == cwd


def resolve_binary(name: str) -> str | None:
    """Resolve a system binary to a trusted absolute path, or ``None`` to skip it.

    :func:`shutil.which` performs a PATH lookup and returns ``None`` when the
    binary is absent (so the caller simply degrades to ``None``). Passing the
    resolved absolute path to :mod:`subprocess` — rather than the bare name —
    prevents Windows ``CreateProcess`` from searching the current working
    directory, so running ``comfy`` from an attacker-controlled directory cannot
    execute a planted ``nvidia-smi.exe``.

    ``name`` must be a bare binary name (see :func:`_is_bare_name`); anything
    carrying a path component is refused rather than looked up, because
    :func:`shutil.which` would hand such a string straight back.

    ``shutil.which`` may itself resolve against the current directory (always on
    Windows; on any platform when ``$PATH`` holds ``.`` or an empty entry), so as
    defense-in-depth two CWD-anchored results are additionally rejected on every
    platform:

    * a result that is not **fully qualified** (see :func:`_is_fully_qualified`).
      ``which`` returns ``os.path.join(entry, name)``, so a relative path means the
      matching ``$PATH`` entry was itself relative (``.``, an empty entry,
      ``subdir``, or Windows' implicitly prepended ``os.curdir``) and the binary
      therefore lives under the attacker-controlled CWD. Handing that string to
      :mod:`subprocess` would re-resolve it against the CWD — exactly the hijack
      this function exists to prevent — so the probe is skipped instead. A binary
      found through a normal absolute ``$PATH`` entry always comes back fully
      qualified and is unaffected.
    * an absolute result sitting directly **in** the CWD (see
      :func:`is_planted_in_cwd`), which covers the CWD appearing in ``$PATH`` as
      an absolute entry.

    A legitimate system binary (e.g. ``nvidia-smi.exe`` under ``System32``) is
    unaffected by either check. The one known false positive is running ``comfy``
    from a directory that is *itself* an absolute ``$PATH`` entry (``/usr/bin``,
    ``C:\\Windows\\System32``): the probe is then skipped and the caller degrades
    exactly as it would if the binary were not installed.
    """
    return resolve_binary_detail(name).path


def resolve_binary_detail(name: str) -> BinaryResolution:
    """The gate shared by :func:`resolve_binary` and :func:`resolve_required_binary`.

    Returns a :class:`BinaryResolution`: ``path`` set and ``reason`` ``None`` on
    success, or ``path`` ``None`` and ``reason`` naming the
    :class:`BinaryRefusal` that stopped it. See :func:`resolve_binary` for the
    full rationale behind each gate; this function exists so the *required*
    wrapper can say which gate fired rather than emitting one message that
    guesses.

    Like :func:`resolve_binary`, it never raises: an unexpected error degrades to
    :attr:`BinaryRefusal.UNVERIFIABLE`.
    """
    try:
        if not _is_bare_name(name):
            logger.debug("refusing to resolve %r: not a bare binary name", name)
            return BinaryResolution(None, BinaryRefusal.NOT_BARE_NAME, None)
        path = shutil.which(name)
        if path is None:
            return BinaryResolution(None, BinaryRefusal.ABSENT, None)
        if not _is_fully_qualified(path):
            # ``which`` returns ``os.path.join(entry, name)``, so a relative
            # result means the matching ``$PATH`` entry was relative — i.e. the
            # match lives under the CWD.
            logger.debug("skipping %r: match is not fully qualified (%s)", name, path)
            return BinaryResolution(None, BinaryRefusal.CWD_ANCHORED, path)
        in_cwd = _place_relative_to_cwd(path)
        if in_cwd is None:
            logger.debug("skipping %r: cannot place match relative to the CWD (%s)", name, path)
            return BinaryResolution(None, BinaryRefusal.UNVERIFIABLE, path)
        if in_cwd:
            logger.debug("skipping %r: resolved into CWD (%s)", name, path)
            return BinaryResolution(None, BinaryRefusal.CWD_ANCHORED, path)
        return BinaryResolution(path, None, path)
    except Exception:
        logger.debug("resolving binary %r failed", name, exc_info=True)
        return BinaryResolution(None, BinaryRefusal.UNVERIFIABLE, None)


def resolve_required_binary(name: str) -> str:
    """Resolve a *required* system binary to a trusted absolute path.

    The required-binary companion to :func:`resolve_binary`: identical gates
    (bare-name check, fully-qualified check, CWD-plant check), but a failure
    raises :class:`BinaryNotFoundError` instead of returning ``None``. Use it for
    binaries whose absence is fatal to the command — ``git``, ``ffmpeg``,
    ``ffprobe`` — where "skip this probe" is not a meaningful degradation.

    Resolving deliberately happens at *call* time, not import time, so the answer
    reflects the ``$PATH`` and CWD in force when the process is actually spawned.
    Callers that ``os.chdir`` into a user-supplied directory should resolve
    **before** the ``chdir``: the absolute path returned here already defeats
    ``CreateProcess``'s current-directory search at spawn time, and resolving
    first means a plant in the target directory cannot even shadow a legitimate
    binary into a hard failure.

    The raised error names the *specific* gate that fired (see
    :attr:`BinaryNotFoundError.reason`), because "not installed" and "installed
    but refused" need opposite remedies — telling someone with a working ``git``
    to install ``git`` sends them the wrong way and hides the fact that a binary
    was found sitting in their working directory.

    :raises BinaryNotFoundError: the binary is not on ``$PATH``, or the only match
        resolved into the current working directory and was refused.
    """
    resolution = resolve_binary_detail(name)
    if resolution.path is None:
        raise BinaryNotFoundError(
            _refusal_message(name, resolution),
            binary=name,
            reason=resolution.reason or BinaryRefusal.UNVERIFIABLE,
            candidate=resolution.candidate,
        )
    return resolution.path


def _refusal_message(name: str, resolution: BinaryResolution) -> str:
    """An actionable one-liner for the gate that refused ``name``."""
    where = f" ({resolution.candidate})" if resolution.candidate else ""
    if resolution.reason is BinaryRefusal.ABSENT:
        return f"{name!r} was not found on PATH. Install {name} and make sure it is on PATH, then try again."
    if resolution.reason is BinaryRefusal.CWD_ANCHORED:
        return (
            f"refusing to run {name!r}: the only match on PATH{where} is in the directory you are "
            f"running from, which is not trusted — a program planted there would run instead of the "
            f"real {name}. Run from a different directory, or make sure {name} is installed on PATH "
            f"outside your working directory."
        )
    if resolution.reason is BinaryRefusal.UNVERIFIABLE:
        return (
            f"refusing to run {name!r}: could not check whether the match on PATH{where} lives in the "
            f"directory you are running from (the working directory may have been deleted). "
            f"Run from a directory that still exists and try again."
        )
    return f"refusing to resolve {name!r}: not a bare binary name."
