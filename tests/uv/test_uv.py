import shutil
import subprocess
from pathlib import Path
from unittest.mock import patch

import pytest

from comfy_cli import ui
from comfy_cli.constants import GPU_OPTION
from comfy_cli.uv import DependencyCompiler, _check_call, parse_req_file

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


def test_make_override_does_not_strip_cuda_toolkit_extras():
    """Regression test for #412: override must not pin cuda-toolkit without extras.

    torch >= 2.11 depends on cuda-toolkit[cublas,cudart,...]==13.0.2.
    make_override() appends the first compile's flat output to override.txt,
    which writes 'cuda-toolkit==13.0.2' (no extras). When compile_core_plus_ext()
    uses this override, uv replaces torch's extras-bearing requirement with the
    bare pin, silently dropping nvidia-cuda-runtime, nvidia-cuda-nvrtc, and
    8 other CUDA packages.
    """
    # Simulate torch >= 2.11 first-compile output (flat pins, no extras)
    mock_stdout = "\n".join(
        [
            "cuda-bindings==13.2.0",
            "cuda-pathfinder==1.5.1",
            "cuda-toolkit==13.0.2",
            "nvidia-cublas==13.1.0.3",
            "nvidia-cuda-cupti==13.0.85",
            "nvidia-cuda-nvrtc==13.0.88",
            "nvidia-cuda-runtime==13.0.96",
            "nvidia-cudnn-cu13==9.19.0.56",
            "nvidia-cufft==12.0.0.61",
            "nvidia-cufile==1.15.1.6",
            "nvidia-curand==10.4.0.35",
            "nvidia-cusolver==12.0.4.66",
            "nvidia-cusparse==12.6.3.3",
            "nvidia-cusparselt-cu13==0.8.0",
            "nvidia-nccl-cu13==2.28.9",
            "nvidia-nvjitlink==13.0.88",
            "nvidia-nvshmem-cu13==3.4.5",
            "nvidia-nvtx==13.0.85",
            "torch==2.11.0+cu130",
            "torchaudio==2.11.0+cu130",
            "torchsde==0.2.6",
            "torchvision==0.26.0+cu130",
            "",
        ]
    )
    mock_result = type("R", (), {"stdout": mock_stdout, "stderr": "", "returncode": 0})()

    depComp = DependencyCompiler(
        cwd=temp,
        gpu=GPU_OPTION.NVIDIA,
        outDir=temp,
        reqFilesCore=[mockReqsDir / "core_reqs.txt"],
        reqFilesExt=[],
        cuda_version="13.0",
    )

    with patch.object(DependencyCompiler, "Compile", return_value=mock_result):
        depComp.make_override()

    override_content = depComp.override.read_text()

    # The override must not contain a bare 'cuda-toolkit==X.Y.Z' pin.
    # If cuda-toolkit appears, it must include extras like [cublas,cudart,...].
    # A bare pin causes uv --override to replace torch's extras-bearing
    # requirement, dropping all extras-only transitive CUDA runtime packages.
    for line in override_content.splitlines():
        stripped = line.strip()
        if stripped.startswith("cuda-toolkit=="):
            pytest.fail(
                f"Override contains bare cuda-toolkit pin without extras: {stripped!r}\n"
                "This causes uv to strip extras from torch's cuda-toolkit dependency, "
                "dropping nvidia-cuda-runtime, nvidia-cuda-nvrtc, and other CUDA packages.\n"
                "See: https://github.com/Comfy-Org/comfy-cli/issues/412"
            )


def test_nvidia_custom_cuda_version():
    depComp = DependencyCompiler(
        cwd=temp, gpu=GPU_OPTION.NVIDIA, outDir=temp, reqFilesCore=[], reqFilesExt=[], cuda_version="11.8"
    )
    assert depComp.torchBackend == "cu118"
    assert depComp.gpuUrl == "https://download.pytorch.org/whl/cu118"


def test_nvidia_cuda_13():
    depComp = DependencyCompiler(
        cwd=temp, gpu=GPU_OPTION.NVIDIA, outDir=temp, reqFilesCore=[], reqFilesExt=[], cuda_version="13.0"
    )
    assert depComp.torchBackend == "cu130"
    assert depComp.gpuUrl == "https://download.pytorch.org/whl/cu130"


def test_amd_custom_rocm_version():
    depComp = DependencyCompiler(
        cwd=temp, gpu=GPU_OPTION.AMD, outDir=temp, reqFilesCore=[], reqFilesExt=[], rocm_version="7.1"
    )
    assert depComp.torchBackend == "rocm7.1"
    assert depComp.gpuUrl == "https://download.pytorch.org/whl/rocm7.1"


def test_nvidia_auto_detected_tag():
    depComp = DependencyCompiler(
        cwd=temp, gpu=GPU_OPTION.NVIDIA, outDir=temp, reqFilesCore=[], reqFilesExt=[], cuda_version="12.8"
    )
    assert depComp.torchBackend == "cu128"
    assert depComp.gpuUrl == "https://download.pytorch.org/whl/cu128"


def test_nvidia_no_cuda_version_uses_default():
    depComp = DependencyCompiler(
        cwd=temp, gpu=GPU_OPTION.NVIDIA, outDir=temp, reqFilesCore=[], reqFilesExt=[], cuda_version=None
    )
    assert depComp.torchBackend == DependencyCompiler.nvidiaTorchBackend
    assert depComp.gpuUrl == DependencyCompiler.nvidiaPytorchUrl


@pytest.mark.parametrize("gpu", [GPU_OPTION.NVIDIA, GPU_OPTION.AMD, GPU_OPTION.CPU])
def test_skip_torch_disables_gpu_url_and_backend(gpu):
    depComp = DependencyCompiler(cwd=temp, gpu=gpu, outDir=temp, reqFilesCore=[], reqFilesExt=[], skip_torch=True)
    assert depComp.torchBackend is None
    assert depComp.gpuUrl is None


def test_skip_torch_override_has_no_torch():
    depComp = DependencyCompiler(
        cwd=temp,
        gpu=GPU_OPTION.NVIDIA,
        outDir=temp,
        reqFilesCore=[mockReqsDir / "core_reqs.txt"],
        reqFilesExt=[],
        skip_torch=True,
    )
    depComp.make_override()
    content = depComp.override.read_text()
    assert "torch" not in content


def test_skip_torch_install_deps_no_extra_index_url():
    depComp = DependencyCompiler(
        cwd=temp, gpu=GPU_OPTION.NVIDIA, outDir=temp, reqFilesCore=[], reqFilesExt=[], skip_torch=True
    )
    depComp.out.write_text("requests==2.31.0\n")
    with patch("comfy_cli.uv._check_call") as mock_check_call:
        depComp.install_deps()
    cmd = mock_check_call.call_args[0][0]
    assert "--extra-index-url" not in cmd


def test_check_call_prints_nfs_hint_on_uv_install_failure(capsys):
    """When a uv pip install command fails, _check_call should print an NFS hint."""
    cmd = ["python", "-m", "uv", "pip", "install", "--requirement", "reqs.txt"]
    with patch("subprocess.check_call", side_effect=subprocess.CalledProcessError(2, cmd)):
        with pytest.raises(subprocess.CalledProcessError):
            _check_call(cmd)

    captured = capsys.readouterr().out
    assert "network filesystem" in captured
    assert "UV_LINK_MODE" in captured
    assert "UV_CACHE_DIR" in captured


def test_check_call_prints_nfs_hint_on_uv_sync_failure(capsys):
    """When a uv pip sync command fails, _check_call should print an NFS hint."""
    cmd = ["python", "-m", "uv", "pip", "sync", "reqs.txt"]
    with patch("subprocess.check_call", side_effect=subprocess.CalledProcessError(2, cmd)):
        with pytest.raises(subprocess.CalledProcessError):
            _check_call(cmd)

    captured = capsys.readouterr().out
    assert "network filesystem" in captured


def test_check_call_no_hint_for_non_uv_failure(capsys):
    """Non-uv commands should not trigger the NFS hint."""
    cmd = ["python", "-m", "pip", "install", "requests"]
    with patch("subprocess.check_call", side_effect=subprocess.CalledProcessError(1, cmd)):
        with pytest.raises(subprocess.CalledProcessError):
            _check_call(cmd)

    captured = capsys.readouterr().out
    assert "network filesystem" not in captured


def test_check_call_no_hint_on_uv_compile_failure(capsys):
    """uv pip compile failures should not trigger the NFS hint (only install/sync)."""
    cmd = ["python", "-m", "uv", "pip", "compile", "reqs.in"]
    with patch("subprocess.check_call", side_effect=subprocess.CalledProcessError(1, cmd)):
        with pytest.raises(subprocess.CalledProcessError):
            _check_call(cmd)

    captured = capsys.readouterr().out
    assert "network filesystem" not in captured


def test_check_call_no_hint_for_pip_install_uv(capsys):
    """'pip install uv' must not trigger the hint even though 'uv' and 'install' are both present."""
    cmd = ["python", "-m", "pip", "install", "--upgrade", "pip", "uv"]
    with patch("subprocess.check_call", side_effect=subprocess.CalledProcessError(1, cmd)):
        with pytest.raises(subprocess.CalledProcessError):
            _check_call(cmd)

    captured = capsys.readouterr().out
    assert "network filesystem" not in captured


# Issue #431: parse_req_file feeds its output into pip argv (pip download /
# pip wheel). Inline comments would be rejected by pip; VCS URL fragments must
# be preserved verbatim (e.g. `#subdirectory=pkg`, `#egg=foo`).


def test_parse_req_file_strips_inline_comments(tmp_path):
    rf = tmp_path / "requirements.txt"
    rf.write_text("foo>=1.0  # trailing comment\n")
    assert parse_req_file(rf) == ["foo>=1.0"]


def test_parse_req_file_strips_inline_comment_with_single_space(tmp_path):
    rf = tmp_path / "requirements.txt"
    rf.write_text("bar==2.3 # single space before hash\n")
    assert parse_req_file(rf) == ["bar==2.3"]


def test_parse_req_file_skips_full_line_comments(tmp_path):
    rf = tmp_path / "requirements.txt"
    rf.write_text("# heading\nfoo>=1.0\n   # indented heading\nbaz\n")
    assert parse_req_file(rf) == ["foo>=1.0", "baz"]


def test_parse_req_file_preserves_vcs_subdirectory_fragment(tmp_path):
    # Regression guard: any naive `split("#")[0]` would break this. `#` is only
    # a comment marker when preceded by whitespace (pip's rule).
    rf = tmp_path / "requirements.txt"
    rf.write_text("git+https://github.com/org/mono.git#subdirectory=pkg\n")
    assert parse_req_file(rf) == ["git+https://github.com/org/mono.git#subdirectory=pkg"]


def test_parse_req_file_preserves_vcs_egg_fragment(tmp_path):
    rf = tmp_path / "requirements.txt"
    rf.write_text("git+https://github.com/org/repo.git@main#egg=foo\n")
    assert parse_req_file(rf) == ["git+https://github.com/org/repo.git@main#egg=foo"]


def test_parse_req_file_preserves_direct_url_hash(tmp_path):
    rf = tmp_path / "requirements.txt"
    rf.write_text("foo @ https://host/f.whl#sha256=abc123\n")
    assert parse_req_file(rf) == ["foo @ https://host/f.whl#sha256=abc123"]


def test_parse_req_file_vcs_with_inline_comment_strips_only_comment(tmp_path):
    # The trickiest case: a VCS spec with a fragment AND a trailing comment.
    # Comment is preceded by whitespace so it must be stripped; fragment is
    # part of the URL and must survive.
    rf = tmp_path / "requirements.txt"
    rf.write_text("git+https://host/r.git#subdirectory=pkg  # note\n")
    assert parse_req_file(rf) == ["git+https://host/r.git#subdirectory=pkg"]


def test_parse_req_file_preserves_double_dash_options(tmp_path):
    rf = tmp_path / "requirements.txt"
    rf.write_text("--extra-index-url https://example.com/simple\nfoo\n")
    assert parse_req_file(rf) == ["--extra-index-url", "https://example.com/simple", "foo"]
