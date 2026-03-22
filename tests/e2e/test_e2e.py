import os
import subprocess
from datetime import datetime
from textwrap import dedent

import pytest

from comfy_cli.resolve_python import resolve_workspace_python


def e2e_test(func):
    return pytest.mark.skipif(
        os.getenv("TEST_E2E", "false") != "true",
        reason="Test e2e is not explicitly enabled",
    )(func)


def exec(cmd: str, **kwargs) -> subprocess.CompletedProcess[str]:
    cmd = dedent(cmd).strip()
    print(f"cmd: {cmd}")

    proc = subprocess.run(
        args=cmd,
        capture_output=True,
        text=True,
        shell=True,
        encoding="utf-8",
        check=False,
        **kwargs,
    )
    print(proc.stdout, proc.stderr)
    return proc


@pytest.fixture(scope="module")
def workspace():
    ws = os.path.join(os.getcwd(), f"comfy-{datetime.now().timestamp()}")
    install_flags = os.getenv("TEST_E2E_COMFY_INSTALL_FLAGS", "--cpu")
    comfy_url = os.getenv("TEST_E2E_COMFY_URL", "")
    url_flag = f"--url {comfy_url}" if comfy_url else ""
    proc = exec(
        f"""
            comfy --skip-prompt --workspace {ws} install {url_flag} {install_flags}
            comfy --skip-prompt set-default {ws}
            comfy --skip-prompt --no-enable-telemetry env
        """
    )
    assert proc.returncode == 0

    # Populate Manager cache before any node operations (blocking fetch).
    proc = exec(f"comfy --workspace {ws} node update-cache")
    assert proc.returncode == 0, f"update-cache failed:\n{proc.stderr}"

    proc = exec(
        f"""
        comfy --workspace {ws} launch --background -- {os.getenv("TEST_E2E_COMFY_LAUNCH_FLAGS_EXTRA", "--cpu")}
        """
    )
    assert proc.returncode == 0

    yield ws

    proc = exec(
        f"""
        comfy --workspace {ws} stop
        """
    )
    assert proc.returncode == 0


@pytest.fixture()
def comfy_cli(workspace):
    exec("comfy --skip-prompt --no-enable-telemetry env")
    return f"comfy --workspace {workspace}"


@e2e_test
def test_model(comfy_cli):
    url = "https://huggingface.co/guoyww/animatediff/resolve/cd71ae134a27ec6008b968d6419952b0c0494cf2/mm_sd_v14.ckpt?download=true"
    path = os.path.join("models", "animatediff_models")
    proc = exec(
        f"""
            {comfy_cli} model download --url {url} --relative-path {path} --filename animatediff_models
        """
    )
    assert proc.returncode == 0

    proc = exec(
        f"""
            {comfy_cli} model list --relative-path {path}
        """
    )
    assert proc.returncode == 0
    assert "animatediff_models" in proc.stdout

    proc = exec(
        f"""
            {comfy_cli} model remove --relative-path {path} --model-names animatediff_models --confirm
        """
    )
    assert proc.returncode == 0


@e2e_test
def test_node(comfy_cli, workspace):
    node = "comfyui-animatediff-evolved"

    # Use --exit-on-fail so the CLI returns non-zero on git clone failure
    # instead of silently succeeding. Retry to handle transient network
    # errors (GitHub rate-limiting git clones on Actions runners).
    for attempt in range(3):
        proc = exec(
            f"""
                {comfy_cli} node install --exit-on-fail {node}
            """
        )
        if proc.returncode == 0:
            break
    assert proc.returncode == 0, f"node install failed after 3 attempts:\n{proc.stderr}"

    for attempt in range(3):
        proc = exec(
            f"""
                {comfy_cli} node reinstall {node}
            """
        )
        if proc.returncode == 0:
            break
    assert proc.returncode == 0, f"node reinstall failed after 3 attempts:\n{proc.stderr}"

    proc = exec(
        f"""
            {comfy_cli} node show all
        """
    )
    assert proc.returncode == 0
    # cm-cli may display the repo name (ComfyUI-AnimateDiff-Evolved) rather
    # than the registry id (comfyui-animatediff-evolved), so compare lowercase.
    assert node.lower() in proc.stdout.lower()

    proc = exec(
        f"""
            {comfy_cli} node update {node}
        """
    )
    assert proc.returncode == 0

    proc = exec(
        f"""
            {comfy_cli} node disable {node}
        """
    )
    assert proc.returncode == 0

    proc = exec(
        f"""
            {comfy_cli} node enable {node}
        """
    )
    assert proc.returncode == 0

    pubID = "comfytest123"
    pubToken = "6075cf7b-47e7-4c58-a3de-38f59a9bcc22"
    proc = exec(
        f"""
            sed 's/PublisherId = ".*"/PublisherId = "{pubID}"/g' pyproject.toml
            {comfy_cli} node publish --token {pubToken}
        """,
        env={"ENVIRONMENT": "stage"},
        cwd=os.path.join(workspace, "custom_nodes", node),
    )


@e2e_test
def test_manager_installed(comfy_cli, workspace):
    """Verify ComfyUI-Manager was installed via manager_requirements.txt."""
    proc = exec(
        f"""
            {comfy_cli} node show all
        """
    )
    assert proc.returncode == 0, f"node show all failed: {proc.stderr}"

    # Check cm_cli is importable (Manager v4 installed as pip package)
    ws_python = resolve_workspace_python(workspace)
    proc = exec(
        f"""
            {ws_python} -c "import cm_cli; print('cm_cli OK')"
        """
    )
    assert proc.returncode == 0, f"cm_cli import failed: {proc.stderr}"
    assert "cm_cli OK" in proc.stdout


@e2e_test
def test_node_uv_compile(comfy_cli):
    """Test --uv-compile flag for node install (requires Manager v4.1+)."""
    node = "comfyui-impact-pack"
    proc = exec(
        f"""
            {comfy_cli} node install --uv-compile {node}
        """
    )
    assert proc.returncode == 0

    # Standalone uv-sync command
    proc = exec(
        f"""
            {comfy_cli} node uv-sync
        """
    )
    assert proc.returncode == 0


@e2e_test
def test_uv_compile_default_config(comfy_cli):
    """Test comfy manager uv-compile-default config command."""
    proc = exec(
        f"""
            {comfy_cli} manager uv-compile-default true
        """
    )
    assert proc.returncode == 0
    assert "enabled" in proc.stdout.lower()

    # Verify it shows in env
    proc = exec(
        """
            comfy --skip-prompt --no-enable-telemetry env
        """
    )
    assert proc.returncode == 0
    assert "UV Compile Default" in proc.stdout
    assert "Enabled" in proc.stdout

    # Disable it back
    proc = exec(
        f"""
            {comfy_cli} manager uv-compile-default false
        """
    )
    assert proc.returncode == 0
    assert "disabled" in proc.stdout.lower()


@e2e_test
def test_run(comfy_cli):
    url = "https://huggingface.co/Comfy-Org/stable-diffusion-v1-5-archive/resolve/main/v1-5-pruned-emaonly-fp16.safetensors?download=true"
    path = os.path.join("models", "checkpoints")
    name = "v1-5-pruned-emaonly.safetensors"
    proc = exec(
        f"""
            {comfy_cli} model download --url {url} --relative-path {path} --filename {name}
        """
    )
    assert proc.returncode == 0

    workflow = os.path.join(os.path.dirname(os.path.realpath(__file__)), "workflow.json")
    proc = exec(
        f"""
        {comfy_cli} run --workflow {workflow} --wait --timeout 180
        """
    )
    assert proc.returncode == 0
