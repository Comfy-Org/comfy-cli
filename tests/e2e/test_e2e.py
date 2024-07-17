import os
import subprocess
from datetime import datetime
from textwrap import dedent

import pytest


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
    proc = exec(
        f"""
            comfy --skip-prompt --workspace {ws} install {os.getenv("TEST_E2E_COMFY_INSTALL_FLAGS", "--cpu")}
            comfy --skip-prompt set-default {ws}
            comfy --skip-prompt --no-enable-telemetry env
        """
    )
    assert 0 == proc.returncode

    proc = exec(
        f"""
        comfy --workspace {ws} launch --background -- {os.getenv("TEST_E2E_COMFY_LAUNCH_FLAGS_EXTRA", "--cpu")}
        """
    )
    assert 0 == proc.returncode

    yield ws

    proc = exec(
        f"""
        comfy --workspace {ws} stop
        """
    )
    assert 0 == proc.returncode


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
    assert 0 == proc.returncode

    proc = exec(
        f"""
            {comfy_cli} model list --relative-path {path}
        """
    )
    assert 0 == proc.returncode
    assert "animatediff_models" in proc.stdout

    proc = exec(
        f"""
            {comfy_cli} model remove --relative-path {path} --model-names animatediff_models --confirm
        """
    )
    assert 0 == proc.returncode


@e2e_test
def test_node(comfy_cli, workspace):
    node = "ComfyUI-AnimateDiff-Evolved"
    proc = exec(
        f"""
            {comfy_cli} node install {node}
        """
    )
    assert 0 == proc.returncode

    proc = exec(
        f"""
            {comfy_cli} node reinstall {node}
        """
    )
    assert 0 == proc.returncode

    proc = exec(
        f"""
            {comfy_cli} node show all
        """
    )
    assert 0 == proc.returncode
    assert node in proc.stdout

    proc = exec(
        f"""
            {comfy_cli} node update {node}
        """
    )
    assert 0 == proc.returncode

    proc = exec(
        f"""
            {comfy_cli} node disable {node}
        """
    )
    assert 0 == proc.returncode

    proc = exec(
        f"""
            {comfy_cli} node enable {node}
        """
    )
    assert 0 == proc.returncode

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
def test_run(comfy_cli):
    url = "https://huggingface.co/runwayml/stable-diffusion-v1-5/resolve/main/v1-5-pruned-emaonly.ckpt"
    path = os.path.join("models", "checkpoints")
    name = "v1-5-pruned-emaonly.ckpt"
    proc = exec(
        f"""
            {comfy_cli} model download --url {url} --relative-path {path} --filename {name}
        """
    )
    assert 0 == proc.returncode

    workflow = os.path.join(
        os.path.dirname(os.path.realpath(__file__)), "workflow.json"
    )
    proc = exec(
        f"""
        {comfy_cli} run --workflow {workflow} --wait --timeout 600
        """
    )
    assert 0 == proc.returncode
