"""Auto-detect CUDA driver version and resolve the best PyTorch wheel suffix."""

from __future__ import annotations

import ctypes
import logging
import os
import platform
import re
import subprocess

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
    """Parse CUDA version from nvidia-smi output, or return None."""
    try:
        output = subprocess.check_output(
            ["nvidia-smi"],
            text=True,
            timeout=10,
            stderr=subprocess.DEVNULL,
        )
    except (FileNotFoundError, subprocess.SubprocessError):
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
    finally:
        if saved is not None:
            os.environ["CUDA_VISIBLE_DEVICES"] = saved

    return _detect_via_nvidia_smi()


def resolve_cuda_wheel(driver_version: tuple[int, int]) -> str | None:
    """Map a driver CUDA version to the best PyTorch wheel suffix.

    Picks the highest wheel tag whose CUDA version <= the driver version.
    Returns None if the driver is too old for any known wheel.
    """
    drv_major, drv_minor = driver_version

    for tag in PYTORCH_CUDA_WHEELS:
        digits = tag[2:]
        if len(digits) == 3:
            whl_major = int(digits[0:2])
            whl_minor = int(digits[2])
        else:
            whl_major = int(digits[0:2])
            whl_minor = int(digits[2:])

        if (whl_major, whl_minor) <= (drv_major, drv_minor):
            return tag

    return None
