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
  :func:`resolve_binary`, so the CWD/qualification gates cannot drift apart.
"""

from __future__ import annotations

import logging
import ntpath
import os
import shutil

logger = logging.getLogger(__name__)

_PATH_SEPARATORS = ("/", "\\")


class BinaryNotFoundError(RuntimeError, FileNotFoundError):
    """A binary a command cannot run without could not be trusted-resolved.

    Raised by :func:`resolve_required_binary` — either the binary is absent from
    ``$PATH``, or the only match was CWD-anchored and therefore refused.

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
    """
    try:
        cwd = os.path.normcase(os.path.realpath(os.getcwd()))
        # ``os.path.dirname`` of a bare/relative ``which`` result is "", which
        # ``realpath`` correctly resolves against the CWD.
        parent = os.path.normcase(os.path.realpath(os.path.dirname(path)))
        return parent == cwd
    except (OSError, ValueError):
        logger.debug("cannot place %r relative to the CWD; treating as planted", path, exc_info=True)
        return True


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
    try:
        if not _is_bare_name(name):
            logger.debug("refusing to resolve %r: not a bare binary name", name)
            return None
        path = shutil.which(name)
        if path is None:
            return None
        if not _is_fully_qualified(path):
            logger.debug("skipping %r: match is not fully qualified (%s)", name, path)
            return None
        if is_planted_in_cwd(path):
            logger.debug("skipping %r: resolved into CWD (%s)", name, path)
            return None
        return path
    except Exception:
        logger.debug("resolving binary %r failed", name, exc_info=True)
        return None


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

    :raises BinaryNotFoundError: the binary is not on ``$PATH``, or the only match
        resolved into the current working directory and was refused.
    """
    path = resolve_binary(name)
    if path is None:
        raise BinaryNotFoundError(
            f"{name!r} was not found on PATH, or the only match resolved into the current directory "
            f"(which is not trusted). Install {name} and make sure it is on PATH outside the directory "
            f"you are running from."
        )
    return path
