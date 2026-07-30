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
import os
import shutil

logger = logging.getLogger(__name__)


def is_planted_in_cwd(path: str) -> bool:
    """Return ``True`` only if ``path`` resolves to a file sitting *directly* in
    ``os.getcwd()`` — the signature of a planted probe binary.

    ``shutil.which`` searches the current directory first on Windows (and on any
    platform whose ``$PATH`` contains ``.`` or an empty entry), so an attacker who
    controls the directory the user runs ``comfy`` from can drop a malicious
    ``nvidia-smi.exe`` there. Such a plant always lands in the CWD *itself*, so we
    reject only a resolved binary whose parent directory **is** the CWD. A
    legitimate system binary in a *subdirectory* — e.g. ``System32`` even when the
    CWD is ``C:\\Windows``, or a drive root — is left untouched, honouring the
    "a legitimate system binary is never rejected" guarantee. Paths are compared
    with :func:`os.path.normcase` so Windows' case-insensitivity can't fail the
    guard open. Ambiguity — a path on a different drive, or an unresolvable one —
    is treated as *not* planted so a legitimate binary is never rejected.
    """
    try:
        cwd = os.path.normcase(os.path.realpath(os.getcwd()))
        # ``os.path.dirname`` of a bare/relative ``which`` result is "", which
        # ``realpath`` correctly resolves against the CWD.
        parent = os.path.normcase(os.path.realpath(os.path.dirname(path)))
        return parent == cwd
    except (OSError, ValueError):
        # Different drives (Windows) or an unresolvable path → not planted.
        return False


def resolve_binary(name: str) -> str | None:
    """Resolve a system binary to a trusted absolute path, or ``None`` to skip it.

    :func:`shutil.which` performs a PATH lookup and returns ``None`` when the
    binary is absent (so the caller simply degrades to ``None``). Passing the
    resolved absolute path to :mod:`subprocess` — rather than the bare name —
    prevents Windows ``CreateProcess`` from searching the current working
    directory, so running ``comfy`` from an attacker-controlled directory cannot
    execute a planted ``nvidia-smi.exe``.

    ``shutil.which`` may itself resolve against the current directory (always on
    Windows; on any platform when ``$PATH`` holds ``.`` or an empty entry), so as
    defense-in-depth two CWD-anchored results are additionally rejected on every
    platform:

    * a **relative** result. ``which`` returns ``os.path.join(entry, name)``, so a
      relative path means the matching ``$PATH`` entry was itself relative (``.``,
      an empty entry, ``subdir``, or Windows' implicitly prepended ``os.curdir``)
      and the binary therefore lives under the attacker-controlled CWD. Handing
      that string to :mod:`subprocess` would re-resolve it against the CWD —
      exactly the hijack this function exists to prevent — so the probe is skipped
      instead. A binary found through a normal absolute ``$PATH`` entry always
      comes back absolute and is unaffected.
    * an absolute result sitting directly **in** the CWD (see
      :func:`is_planted_in_cwd`), which covers the CWD appearing in ``$PATH`` as
      an absolute entry.

    A legitimate system binary (e.g. ``nvidia-smi.exe`` under ``System32``) is
    unaffected by either check.
    """
    try:
        path = shutil.which(name)
        if path is None:
            return None
        if not os.path.isabs(path):
            logger.debug("skipping %r: relative PATH match anchored in CWD (%s)", name, path)
            return None
        if is_planted_in_cwd(path):
            logger.debug("skipping %r: resolved into CWD (%s)", name, path)
            return None
        return path
    except Exception:
        logger.debug("resolving binary %r failed", name, exc_info=True)
        return None
