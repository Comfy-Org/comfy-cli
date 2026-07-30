"""Auto-detect CUDA driver version and resolve the best PyTorch wheel suffix."""

from __future__ import annotations

import ctypes
import logging
import os
import platform
import re
import subprocess

from comfy_cli import _safe_exec

logger = logging.getLogger(__name__)

PYTORCH_CUDA_WHEELS: list[str] = [
    "cu130",
    "cu129",
    "cu128",
    "cu126",
    "cu124",
    "cu121",
    "cu118",
]

DEFAULT_CUDA_TAG = "cu126"


def _load_libcuda() -> ctypes.CDLL:
    """Load the NVIDIA CUDA driver library.

    Raises OSError when the library cannot be found on any known path.
    """
    system = platform.system()

    if system == "Windows":
        candidates = ["nvcuda.dll"]
    else:
        candidates = [
            "libcuda.so.1",
            "/usr/lib/wsl/lib/libcuda.so.1",
            "/usr/lib64/nvidia/libcuda.so.1",
            "/usr/lib/x86_64-linux-gnu/libcuda.so.1",
        ]

    for path in candidates:
        try:
            return ctypes.CDLL(path)
        except OSError:
            continue

    raise OSError("Could not load CUDA driver library from any known path")


def _detect_via_ctypes() -> int | None:
    """Return the raw driver version int from cuDriverGetVersion, or None."""
    try:
        libcuda = _load_libcuda()
    except OSError:
        logger.debug("Failed to load libcuda")
        return None

    try:
        ret = libcuda.cuInit(0)
        if ret != 0:
            logger.debug("cuInit returned %d", ret)
            return None

        version = ctypes.c_int()
        ret = libcuda.cuDriverGetVersion(ctypes.byref(version))
        if ret != 0:
            logger.debug("cuDriverGetVersion returned %d", ret)
            return None

        return version.value
    except Exception:
        logger.debug("ctypes CUDA call failed", exc_info=True)
        return None


def _detect_via_nvidia_smi() -> tuple[int, int] | None:
    """Parse CUDA version from nvidia-smi output, or return None.

    ``nvidia-smi`` is resolved to a trusted absolute path first: invoking it by
    bare name would let Windows' ``CreateProcess`` pick up an ``nvidia-smi.exe``
    planted in the current working directory. A binary that is absent, or whose
    only match is anchored in the CWD, resolves to ``None`` and the probe is
    skipped — the same degrade-to-``None`` outcome as a failed run.

    Spawning a resolved absolute path surfaces ``OSError`` variants that a bare
    name never reached: ``PermissionError`` on a ``noexec``/SELinux-restricted
    mount, or ``OSError: [Errno 8] Exec format error`` for a file that is ``+x``
    but not a valid executable. Those are caught alongside
    :class:`subprocess.SubprocessError` so the promised degradation to ``None``
    holds instead of aborting ``comfy install`` with a traceback.
    """
    nvidia_smi = _safe_exec.resolve_binary("nvidia-smi")
    if nvidia_smi is None:
        return None

    try:
        output = subprocess.check_output(
            [nvidia_smi],
            text=True,
            timeout=10,
            stderr=subprocess.DEVNULL,
        )
    except (OSError, subprocess.SubprocessError):
        logger.debug("nvidia-smi probe failed", exc_info=True)
        return None

    match = re.search(r"CUDA Version:\s*(\d+)\.(\d+)", output)
    if not match:
        return None

    return int(match.group(1)), int(match.group(2))


def detect_cuda_driver_version() -> tuple[int, int] | None:
    """Detect the CUDA driver version.

    Tries ctypes (cuDriverGetVersion) first, then falls back to nvidia-smi.
    Returns (major, minor) or None if detection fails entirely.
    """
    saved = os.environ.get("CUDA_VISIBLE_DEVICES")
    try:
        if saved is not None:
            os.environ.pop("CUDA_VISIBLE_DEVICES", None)

        raw = _detect_via_ctypes()
        if raw is not None:
            major = raw // 1000
            minor = (raw % 1000) // 10
            return major, minor

        return _detect_via_nvidia_smi()
    finally:
        if saved is not None:
            os.environ["CUDA_VISIBLE_DEVICES"] = saved


def resolve_cuda_wheel(driver_version: tuple[int, int]) -> str | None:
    """Map a driver CUDA version to the best PyTorch wheel suffix.

    Picks the highest wheel tag whose CUDA version <= the driver version.
    Returns None if the driver is too old for any known wheel.
    """
    drv_major, drv_minor = driver_version

    for tag in PYTORCH_CUDA_WHEELS:
        digits = tag[2:]
        whl_major = int(digits[:2])
        whl_minor = int(digits[2:])

        if (whl_major, whl_minor) <= (drv_major, drv_minor):
            return tag

    return None
