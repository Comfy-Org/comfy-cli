"""Real-hardware integration tests for CUDA auto-detection.

These tests call the detection functions without mocks, exercising the
actual ctypes/nvidia-smi code paths on machines with NVIDIA drivers.

Automatically skipped when nvidia-smi is not available (i.e. no NVIDIA GPU).
Runs on GPU CI runners (run-on-gpu.yml) and any dev machine with a GPU.
"""

import shutil
import subprocess

import pytest

from comfy_cli.cuda_detect import (
    PYTORCH_CUDA_WHEELS,
    _detect_via_nvidia_smi,
    detect_cuda_driver_version,
    resolve_cuda_wheel,
)

_has_nvidia_smi = shutil.which("nvidia-smi") is not None

pytestmark = pytest.mark.skipif(
    not _has_nvidia_smi,
    reason="nvidia-smi not found — no NVIDIA GPU available",
)


def _nvidia_smi_cuda_version() -> tuple[int, int] | None:
    """Parse CUDA version directly from nvidia-smi for cross-checking."""
    try:
        out = subprocess.check_output(["nvidia-smi"], text=True, timeout=10, stderr=subprocess.DEVNULL)
    except (FileNotFoundError, subprocess.SubprocessError):
        return None
    import re

    m = re.search(r"CUDA Version:\s*(\d+)\.(\d+)", out)
    return (int(m.group(1)), int(m.group(2))) if m else None


class TestRealDetection:
    def test_detect_returns_valid_tuple(self):
        result = detect_cuda_driver_version()
        assert result is not None, "detect_cuda_driver_version() returned None on a machine with nvidia-smi"
        major, minor = result
        assert isinstance(major, int)
        assert isinstance(minor, int)
        assert major >= 11, f"Unexpected CUDA major version: {major}"

    def test_detect_matches_nvidia_smi(self):
        smi_version = _nvidia_smi_cuda_version()
        assert smi_version is not None

        detected = detect_cuda_driver_version()
        assert detected is not None
        assert detected == smi_version, (
            f"detect_cuda_driver_version() returned {detected} but nvidia-smi reports {smi_version}"
        )

    def test_nvidia_smi_fallback_works(self):
        result = _detect_via_nvidia_smi()
        assert result is not None, "_detect_via_nvidia_smi() returned None despite nvidia-smi being available"
        major, minor = result
        assert major >= 11

    def test_resolve_wheel_for_detected_driver(self):
        detected = detect_cuda_driver_version()
        assert detected is not None

        tag = resolve_cuda_wheel(detected)
        assert tag is not None, f"resolve_cuda_wheel({detected}) returned None — driver too old for any wheel?"
        assert tag in PYTORCH_CUDA_WHEELS

    def test_resolved_wheel_version_not_greater_than_driver(self):
        detected = detect_cuda_driver_version()
        assert detected is not None
        drv_major, drv_minor = detected

        tag = resolve_cuda_wheel(detected)
        assert tag is not None

        digits = tag[2:]
        whl_major = int(digits[0:2])
        whl_minor = int(digits[2:])
        assert (whl_major, whl_minor) <= (drv_major, drv_minor), (
            f"Wheel {tag} requires CUDA {whl_major}.{whl_minor} but driver only supports {drv_major}.{drv_minor}"
        )
