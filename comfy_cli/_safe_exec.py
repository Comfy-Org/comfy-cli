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

Contract: **never raises.** Every failure — a missing binary, a broken ``$PATH``
lookup, an unresolvable path — degrades to ``None`` so callers can keep their
existing degrade-to-``None`` behaviour without new error handling.
"""

from __future__ import annotations

import logging
import ntpath
import os
import shutil

logger = logging.getLogger(__name__)

_PATH_SEPARATORS = ("/", "\\")


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
