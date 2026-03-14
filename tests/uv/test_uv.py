import shutil
from pathlib import Path
from unittest.mock import patch

import pytest

from comfy_cli import ui
from comfy_cli.constants import GPU_OPTION
from comfy_cli.uv import DependencyCompiler

hereDir = Path(__file__).parent.resolve()
mockComfyDir = hereDir / "mock_comfy"
mockReqsDir = hereDir / "mock_requirements"

# set up a temp dir to write files to
testsDir = hereDir.parent.resolve()
temp = testsDir / "temp" / "test_uv"
shutil.rmtree(temp, ignore_errors=True)
temp.mkdir(exist_ok=True, parents=True)


@pytest.fixture
def mock_prompt_select(monkeypatch):
    mockChoices = ["==1.13.0", "==2.0.0"]

    def _mock_prompt_select(*args, **kwargs):
        return mockChoices.pop(0)

    monkeypatch.setattr(ui, "prompt_select", _mock_prompt_select)


def test_find_req_files():
    mockNodesDir = mockComfyDir / "custom_nodes"

    knownReqFilesCore = [mockComfyDir / "pyproject.toml"]
    knownReqFilesExt = sorted(
        [
            mockNodesDir / "x" / "requirements.txt",
            mockNodesDir / "y" / "setup.cfg",
            mockNodesDir / "z" / "setup.py",
        ]
    )

    depComp = DependencyCompiler(cwd=mockComfyDir)

    testReqFilesCore = depComp.reqFilesCore
    testReqFilesExt = sorted(depComp.reqFilesExt)

    assert knownReqFilesCore == testReqFilesCore
    assert knownReqFilesExt == testReqFilesExt


def test_compile(mock_prompt_select):
    depComp = DependencyCompiler(
        cwd=temp,
        gpu=GPU_OPTION.AMD,
        outDir=temp,
        reqFilesCore=[mockReqsDir / "core_reqs.txt"],
        reqFilesExt=[mockReqsDir / "x_reqs.txt", mockReqsDir / "y_reqs.txt"],
    )

    depComp.make_override()
    depComp.compile_core_plus_ext()

    with open(mockReqsDir / "requirements.compiled") as known, open(temp / "requirements.compiled") as test:
        # compare all non-commented lines in generated file vs reference file
        knownLines, testLines = [
            [line for line in known.readlines() if not line.strip().startswith("#")],
            [line for line in test.readlines() if not line.strip().startswith("#")],
        ]

        optionalPrefixes = ("colorama==",)

        def _filter_optional(lines: list[str]) -> list[str]:
            # drop platform-specific extras (Windows pulls in colorama via tqdm)
            return [line for line in lines if not any(line.strip().startswith(prefix) for prefix in optionalPrefixes)]

        knownLines, testLines = [_filter_optional(lines) for lines in (knownLines, testLines)]

        assert knownLines == testLines


def test_torch_backend_nvidia():
    depComp = DependencyCompiler(cwd=temp, gpu=GPU_OPTION.NVIDIA, outDir=temp, reqFilesCore=[], reqFilesExt=[])
    assert depComp.torchBackend == "cu126"
    assert depComp.gpuUrl == DependencyCompiler.nvidiaPytorchUrl


def test_torch_backend_amd():
    depComp = DependencyCompiler(cwd=temp, gpu=GPU_OPTION.AMD, outDir=temp, reqFilesCore=[], reqFilesExt=[])
    assert depComp.torchBackend == "rocm6.3"
    assert depComp.gpuUrl == DependencyCompiler.rocmPytorchUrl


def test_torch_backend_cpu():
    depComp = DependencyCompiler(cwd=temp, gpu=GPU_OPTION.CPU, outDir=temp, reqFilesCore=[], reqFilesExt=[])
    assert depComp.torchBackend == "cpu"
    assert depComp.gpuUrl == DependencyCompiler.cpuPytorchUrl


def test_torch_backend_none():
    with patch.object(DependencyCompiler, "Resolve_Gpu", return_value=None):
        depComp = DependencyCompiler(cwd=temp, gpu=None, outDir=temp, reqFilesCore=[], reqFilesExt=[])
    assert depComp.torchBackend is None
    assert depComp.gpuUrl is None


def test_compile_passes_torch_backend():
    """Verify that Compile() includes --torch-backend in the command when provided."""
    with patch("comfy_cli.uv._run") as mock_run:
        mock_run.return_value = type("R", (), {"stdout": "", "stderr": "", "returncode": 0})()
        DependencyCompiler.Compile(
            cwd=temp,
            reqFiles=[mockReqsDir / "core_reqs.txt"],
            torch_backend="cu126",
        )
    cmd = mock_run.call_args[0][0]
    idx = cmd.index("--torch-backend")
    assert cmd[idx + 1] == "cu126"


def test_compile_omits_torch_backend_when_none():
    """Verify that Compile() does not include --torch-backend when torch_backend is None."""
    with patch("comfy_cli.uv._run") as mock_run:
        mock_run.return_value = type("R", (), {"stdout": "", "stderr": "", "returncode": 0})()
        DependencyCompiler.Compile(
            cwd=temp,
            reqFiles=[mockReqsDir / "core_reqs.txt"],
            torch_backend=None,
        )
    cmd = mock_run.call_args[0][0]
    assert "--torch-backend" not in cmd


def test_compiled_output_has_no_extra_index_url(mock_prompt_select):
    """The compiled output must not contain --extra-index-url (torch-backend handles routing)."""
    depComp = DependencyCompiler(
        cwd=temp,
        gpu=GPU_OPTION.AMD,
        outDir=temp,
        reqFilesCore=[mockReqsDir / "core_reqs.txt"],
        reqFilesExt=[mockReqsDir / "x_reqs.txt", mockReqsDir / "y_reqs.txt"],
    )
    depComp.make_override()
    depComp.compile_core_plus_ext()

    content = depComp.out.read_text()
    assert "--extra-index-url" not in content


def test_override_file_has_no_extra_index_url():
    depComp = DependencyCompiler(
        cwd=temp,
        gpu=GPU_OPTION.AMD,
        outDir=temp,
        reqFilesCore=[mockReqsDir / "core_reqs.txt"],
        reqFilesExt=[],
    )
    depComp.make_override()

    content = depComp.override.read_text()
    assert "--extra-index-url" not in content
    assert "torch" in content
