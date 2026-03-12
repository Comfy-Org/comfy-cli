"""Integration tests for torch backend compilation.

These tests do real uv pip compile with a torch requirement and verify
the compiled output contains the correct torch variant for each backend.

Requires network access. Gated behind TEST_TORCH_BACKEND=true.

Platform constraints (PyTorch wheel availability):
  - NVIDIA (cu126): Linux, Windows
  - AMD (rocm6.1): Linux only
  - CPU: Linux, Windows, macOS
  - Default PyPI (mac path): all platforms
"""

import os
import shutil
import sys
from pathlib import Path

import pytest

from comfy_cli.constants import GPU_OPTION
from comfy_cli.uv import DependencyCompiler

pytestmark = pytest.mark.skipif(
    os.environ.get("TEST_TORCH_BACKEND") != "true",
    reason="Set TEST_TORCH_BACKEND=true to run torch backend integration tests",
)

_here = Path(__file__).parent.resolve()
_temp = _here.parent / "temp" / "test_torch_backend"


@pytest.fixture(autouse=True)
def _setup_temp():
    shutil.rmtree(_temp, ignore_errors=True)
    _temp.mkdir(exist_ok=True, parents=True)
    (_temp / "reqs.txt").write_text("torch\n")


def _compile_for(gpu):
    dc = DependencyCompiler(
        cwd=_temp,
        gpu=gpu,
        outDir=_temp,
        reqFilesCore=[_temp / "reqs.txt"],
        reqFilesExt=[],
    )
    dc.make_override()
    dc.compile_core_plus_ext()
    return dc.out.read_text()


@pytest.mark.skipif(sys.platform == "darwin", reason="No CUDA wheels for macOS")
def test_compile_nvidia():
    content = _compile_for(GPU_OPTION.NVIDIA)
    assert "+cu126" in content
    assert "download.pytorch.org/whl/cu126" in content
    assert "--extra-index-url" not in content


@pytest.mark.skipif(sys.platform != "linux", reason="ROCm wheels are Linux-only")
def test_compile_amd():
    content = _compile_for(GPU_OPTION.AMD)
    assert "+rocm" in content
    assert "download.pytorch.org/whl/rocm" in content
    assert "--extra-index-url" not in content


@pytest.mark.skipif(sys.platform == "darwin", reason="No +cpu variant wheels for macOS")
def test_compile_cpu():
    content = _compile_for(GPU_OPTION.CPU)
    assert "+cpu" in content
    assert "download.pytorch.org/whl/cpu" in content
    assert "--extra-index-url" not in content
    # must not pull in nvidia CUDA libraries
    assert "nvidia-" not in content


def test_compile_mac():
    content = _compile_for(GPU_OPTION.MAC_M_SERIES)
    assert "torch==" in content
    # default PyPI torch — no GPU-specific local version suffix
    assert "+cu" not in content
    assert "+rocm" not in content
    assert "+cpu" not in content
    assert "--extra-index-url" not in content
