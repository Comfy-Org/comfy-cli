"""E2E tests for comfy-cli uv-compile support (requires Manager v4.1+).

Tests the full stack: comfy node → execute_cm_cli → cm_cli subprocess.
Uses ltdrdata's dedicated test packs (nodepack-test1-do-not-install,
nodepack-test2-do-not-install) which intentionally conflict on ansible
versions and contain no executable code.

Supply-chain safety policy:
    Only node packs from verified, controllable authors (ltdrdata,
    comfyanonymous) are used. Adding packs from unverified sources
    is prohibited.

Usage:
    TEST_E2E=true \\
    TEST_E2E_COMFY_URL="https://github.com/ltdrdata/ComfyUI.git@dr-bump-manager" \\
    pytest tests/e2e/test_e2e_uv_compile.py -v
"""

import os
import shutil
import subprocess
import sys
from datetime import datetime
from textwrap import dedent

import pytest

# Real node packs for normal installation testing
PACK_IMPACT = "comfyui-impact-pack"
PACK_INSPIRE = "comfyui-inspire-pack"

# Test node packs from ltdrdata — intentionally conflict on ansible versions
REPO_TEST1 = "https://github.com/ltdrdata/nodepack-test1-do-not-install"
REPO_TEST2 = "https://github.com/ltdrdata/nodepack-test2-do-not-install"
PACK_TEST1 = "nodepack-test1-do-not-install"
PACK_TEST2 = "nodepack-test2-do-not-install"


def _e2e_enabled():
    return os.getenv("TEST_E2E", "false") == "true"


pytestmark = [
    pytest.mark.skipif(not _e2e_enabled(), reason="TEST_E2E not enabled"),
]


def exec(cmd: str, timeout: int = 600, **kwargs) -> subprocess.CompletedProcess[str]:
    cmd = dedent(cmd).strip()
    print(f"cmd: {cmd}")
    try:
        proc = subprocess.run(
            args=cmd,
            capture_output=True,
            text=True,
            shell=True,
            encoding="utf-8",
            check=False,
            timeout=timeout,
            **kwargs,
        )
    except subprocess.TimeoutExpired as e:
        print(f"[exec] TIMEOUT after {timeout}s: {cmd}", flush=True)
        # Return a synthetic failed result so tests get a clear failure message
        return subprocess.CompletedProcess(
            args=cmd,
            returncode=124,
            stdout=e.stdout or "",
            stderr=e.stderr or f"Timed out after {timeout}s",
        )
    print(proc.stdout, proc.stderr)
    return proc


def _rmtree_retry(path, retries=5, delay=2.0):
    """Remove directory with retries for Windows file lock delays.

    On Windows, .git/objects/pack/* files may be briefly locked after
    git clone exits. Retries with read-only file handling.
    """
    import stat
    import time

    def _on_rm_error(func, fpath, _exc_info):
        """Handle read-only files on Windows (e.g. .git/objects/pack/*.idx)."""
        try:
            os.chmod(fpath, stat.S_IWRITE)
            func(fpath)
        except OSError:
            pass

    for attempt in range(retries):
        try:
            shutil.rmtree(path, onerror=_on_rm_error)
            return
        except (PermissionError, OSError):
            if attempt < retries - 1:
                time.sleep(delay)
    shutil.rmtree(path, ignore_errors=True)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def workspace():
    """Install ComfyUI (with Manager v4) and launch in background."""
    ws = os.path.join(os.getcwd(), f"comfy-uv-{datetime.now().timestamp()}")
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

    # Override Manager if MANAGER_OVERRIDE is set (skip PyPI/Core PR cycle).
    # Accepts:  "4.1b7" (PyPI)  or  "Comfy-Org/ComfyUI-Manager@branch" (git clone + uv)
    venv_dir = os.path.join(ws, ".venv")
    if sys.platform == "win32":
        venv_python = os.path.join(venv_dir, "Scripts", "python.exe")
    else:
        venv_python = os.path.join(venv_dir, "bin", "python")

    manager_override = os.getenv("MANAGER_OVERRIDE", "")
    if manager_override:
        if "@" in manager_override and "/" in manager_override:
            # Branch install: "Comfy-Org/ComfyUI-Manager@fix-branch"
            # Uses uv (not pip) because Manager repo has flat-layout incompatible with setuptools.
            repo_spec, branch = manager_override.rsplit("@", 1)
            clone_dir = os.path.join(ws, "_manager_override")
            if os.path.isdir(clone_dir):
                shutil.rmtree(clone_dir)
            proc = exec(f"git clone --branch {branch} --depth 1 https://github.com/{repo_spec}.git {clone_dir}")
            assert proc.returncode == 0, f"Manager clone failed:\n{proc.stderr}"
            proc = exec(f"uv pip install {clone_dir} --reinstall-package comfyui-manager --python {venv_dir}")
            assert proc.returncode == 0, f"Manager override install failed:\n{proc.stderr}"
        else:
            # PyPI version: "4.1b7"
            proc = exec(
                f"{venv_python} -m pip install comfyui-manager=={manager_override} --pre --force-reinstall --no-deps"
            )
            assert proc.returncode == 0, f"Manager override failed:\n{proc.stderr}"

    # Populate Manager cache before any node operations (blocking fetch).
    proc = exec(f"comfy --workspace {ws} node update-cache")
    assert proc.returncode == 0, f"update-cache failed:\n{proc.stderr}"

    # NOTE: No 'comfy launch --background' here. These tests only exercise
    # cm_cli commands (node install/reinstall/update/fix/uv-sync) and don't
    # need a running ComfyUI server. Launching ComfyUI in background causes
    # Windows file lock issues: ComfyUI scans custom_nodes/, holds handles
    # on .git/objects/pack/*.idx, and prevents cleanup between tests.

    yield ws


@pytest.fixture()
def comfy_cli(workspace):
    return f"comfy --workspace {workspace}"


@pytest.fixture(autouse=True)
def _clean_test_packs(workspace):
    """Remove test node packs before and after each test."""
    custom_nodes = os.path.join(workspace, "custom_nodes")

    def _remove(name):
        path = os.path.join(custom_nodes, name)
        if os.path.islink(path):
            os.unlink(path)
        elif os.path.isdir(path):
            _rmtree_retry(path)

    _remove(PACK_TEST1)
    _remove(PACK_TEST2)
    yield
    _remove(PACK_TEST1)
    _remove(PACK_TEST2)


# ---------------------------------------------------------------------------
# Normal installation with real packs
# ---------------------------------------------------------------------------


def test_real_packs_sequential_no_conflict(comfy_cli):
    """Sequential install of two real packs with --uv-compile — no conflicts."""
    proc = exec(f"{comfy_cli} node install --uv-compile {PACK_IMPACT}")
    combined = proc.stdout + proc.stderr

    assert proc.returncode == 0
    assert "Resolving dependencies for" in combined

    proc = exec(f"{comfy_cli} node install --uv-compile {PACK_INSPIRE}")
    combined = proc.stdout + proc.stderr

    assert proc.returncode == 0
    assert "Resolving dependencies for" in combined
    assert "Conflicting packages" not in combined


def test_real_packs_simultaneous_no_conflict(comfy_cli):
    """Simultaneous install of two real packs with --uv-compile — no conflicts."""
    proc = exec(f"{comfy_cli} node install --uv-compile {PACK_IMPACT} {PACK_INSPIRE}")
    combined = proc.stdout + proc.stderr

    assert proc.returncode == 0
    assert "Resolving dependencies for" in combined
    assert "Conflicting packages" not in combined


# ---------------------------------------------------------------------------
# Progressive conflict (real packs + conflict packs)
# ---------------------------------------------------------------------------


def test_progressive_conflict(comfy_cli):
    """Real packs installed → +conflict-pack-1 OK → +conflict-pack-2 CONFLICT."""
    # Step 1: Install real packs — no conflict
    proc = exec(f"{comfy_cli} node install --uv-compile {PACK_IMPACT} {PACK_INSPIRE}")
    combined = proc.stdout + proc.stderr
    assert proc.returncode == 0
    assert "Conflicting packages" not in combined

    # Step 2: Add first conflict test pack — still no conflict
    proc = exec(f"{comfy_cli} node install --uv-compile {REPO_TEST1}")
    combined = proc.stdout + proc.stderr
    assert proc.returncode == 0
    assert "Conflicting packages" not in combined

    # Step 3: Add second conflict test pack — conflict between test packs
    proc = exec(f"{comfy_cli} node install --uv-compile {REPO_TEST2}")
    combined = proc.stdout + proc.stderr
    assert "Conflicting packages (by node pack):" in combined
    assert PACK_TEST1 in combined
    assert PACK_TEST2 in combined


# ---------------------------------------------------------------------------
# Reinstall / Update / Fix with --uv-compile
# ---------------------------------------------------------------------------


def test_node_reinstall_uv_compile(comfy_cli):
    """Reinstall with --uv-compile → resolution runs."""
    setup = exec(f"{comfy_cli} node install {REPO_TEST1}")
    assert setup.returncode == 0, f"Setup install failed: {setup.stderr}"

    proc = exec(f"{comfy_cli} node reinstall --uv-compile {REPO_TEST1}")
    combined = proc.stdout + proc.stderr

    assert proc.returncode == 0, f"reinstall failed: {proc.stderr}"
    assert "Resolving dependencies for" in combined


def test_node_update_uv_compile(comfy_cli):
    """Update with --uv-compile → resolution runs."""
    setup = exec(f"{comfy_cli} node install {REPO_TEST1}")
    assert setup.returncode == 0, f"Setup install failed: {setup.stderr}"

    proc = exec(f"{comfy_cli} node update --uv-compile {REPO_TEST1}")
    combined = proc.stdout + proc.stderr

    assert proc.returncode == 0, f"update failed: {proc.stderr}"
    assert "Resolving dependencies for" in combined


def test_node_fix_uv_compile(comfy_cli):
    """Fix with --uv-compile → resolution runs."""
    setup = exec(f"{comfy_cli} node install {REPO_TEST1}")
    assert setup.returncode == 0, f"Setup install failed: {setup.stderr}"

    proc = exec(f"{comfy_cli} node fix --uv-compile {REPO_TEST1}")
    combined = proc.stdout + proc.stderr

    assert proc.returncode == 0, f"fix failed: {proc.stderr}"
    assert "Resolving dependencies for" in combined


def test_node_restore_deps_uv_compile(comfy_cli):
    """restore-dependencies --uv-compile → resolution runs."""
    setup = exec(f"{comfy_cli} node install {REPO_TEST1}")
    assert setup.returncode == 0, f"Setup install failed: {setup.stderr}"

    proc = exec(f"{comfy_cli} node restore-dependencies --uv-compile")
    combined = proc.stdout + proc.stderr

    assert proc.returncode == 0, f"restore-dependencies failed: {proc.stderr}"
    assert "Resolving dependencies for" in combined


# ---------------------------------------------------------------------------
# Standalone uv-sync
# ---------------------------------------------------------------------------


def test_node_uv_sync_standalone(comfy_cli):
    """Standalone comfy node uv-sync with installed pack."""
    setup = exec(f"{comfy_cli} node install {REPO_TEST1}")
    assert setup.returncode == 0, f"Setup install failed: {setup.stderr}"

    proc = exec(f"{comfy_cli} node uv-sync")
    combined = proc.stdout + proc.stderr

    assert proc.returncode == 0
    assert "Resolving dependencies for" in combined


def test_node_uv_sync_standalone_conflict(comfy_cli):
    """Standalone uv-sync with conflicting packs → conflict attribution."""
    setup1 = exec(f"{comfy_cli} node install {REPO_TEST1}")
    assert setup1.returncode == 0, f"Setup install test1 failed: {setup1.stderr}"
    setup2 = exec(f"{comfy_cli} node install {REPO_TEST2}")
    assert setup2.returncode == 0, f"Setup install test2 failed: {setup2.stderr}"

    proc = exec(f"{comfy_cli} node uv-sync")
    combined = proc.stdout + proc.stderr

    assert "Conflicting packages (by node pack):" in combined
    assert PACK_TEST1 in combined
    assert PACK_TEST2 in combined


# ---------------------------------------------------------------------------
# Config default
# ---------------------------------------------------------------------------


def test_uv_compile_config_default(comfy_cli):
    """Config default true → install without flag triggers resolution."""
    proc = exec(f"{comfy_cli} manager uv-compile-default true")
    assert proc.returncode == 0

    try:
        proc = exec(f"{comfy_cli} node install {REPO_TEST1}")
        combined = proc.stdout + proc.stderr

        assert proc.returncode == 0, f"install failed: {proc.stderr}"
        assert "Resolving dependencies for" in combined
    finally:
        exec(f"{comfy_cli} manager uv-compile-default false")


def test_no_uv_compile_overrides_config(comfy_cli):
    """--no-uv-compile overrides config default."""
    proc = exec(f"{comfy_cli} manager uv-compile-default true")
    assert proc.returncode == 0

    try:
        proc = exec(f"{comfy_cli} node install --no-uv-compile {REPO_TEST1}")
        combined = proc.stdout + proc.stderr

        assert proc.returncode == 0, f"install failed: {proc.stderr}"
        assert "Resolving dependencies for" not in combined
    finally:
        exec(f"{comfy_cli} manager uv-compile-default false")


# ---------------------------------------------------------------------------
# Mutual exclusivity
# ---------------------------------------------------------------------------


def test_uv_compile_mutual_exclusivity(comfy_cli):
    """--uv-compile cannot be used with --fast-deps or --no-deps."""
    # --uv-compile + --fast-deps
    proc = exec(f"{comfy_cli} node install --uv-compile --fast-deps {REPO_TEST1}")
    assert proc.returncode != 0
    assert "Cannot use" in (proc.stdout + proc.stderr)

    # --uv-compile + --no-deps
    proc = exec(f"{comfy_cli} node install --uv-compile --no-deps {REPO_TEST1}")
    assert proc.returncode != 0
    assert "Cannot use" in (proc.stdout + proc.stderr)
