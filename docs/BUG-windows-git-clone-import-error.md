# Bug Report: Windows E2E — git-clone install fails on Windows

**Affected**: ComfyUI-Manager v4.1b5–b6 (pip-installed) on Windows
**Severity**: Medium — URL-based node installation broken on Windows; name-based installation unaffected
**CI Reference**: [PR #363 — windows-latest re-run](https://github.com/Comfy-Org/comfy-cli/actions/runs/23202343378/job/67572724097)

## Summary

When installing custom nodes by **URL** on Windows, Manager's `git_helper.py` subprocess fails. Installing by **package name** works fine. Linux and macOS are unaffected.

Two distinct failures were observed across Manager versions:

| Manager | Error | Failed Tests |
|---------|-------|-------------|
| 4.1b5 | `ModuleNotFoundError: No module named 'comfy'` | 3/18 |
| 4.1b6 | `git clone` exit code 128 (stderr swallowed) | 2/18 |

## Reproduction

```bash
# Windows only — fails
comfy node install https://github.com/ltdrdata/nodepack-test1-do-not-install

# Works on all platforms
comfy node install comfyui-impact-pack
```

## Timeline

### Phase 1: Manager 4.1b5 — `ModuleNotFoundError` (3 tests failing)

**Failing tests**: `test_progressive_conflict`, `test_node_reinstall_uv_compile`, `test_node_uv_sync_standalone_conflict`

**Execution chain**:
```
comfy node install {URL}
  └─ execute_cm_cli() → python -m cm_cli install {URL}
       └─ manager_core.py:1336 repo_install()
            └─ run_script([python, git_helper.py, "--clone", ...])
                 └─ git_helper.py:12
                      from comfyui_manager.common.timestamp_utils import ...
                      └─ comfyui_manager/__init__.py:6
                           from comfy.cli_args import args
                           └─ ModuleNotFoundError ← CRASH
```

**Root cause**: `git_helper.py` was executed as a standalone subprocess. Importing `timestamp_utils` triggered `comfyui_manager/__init__.py`, which had a top-level `from comfy.cli_args import args`. The `comfy` module (ComfyUI) was not on the subprocess's Python path on Windows.

**Why name-based install works**: Package-name installs use the registry download path — they do NOT spawn `git_helper.py`. Only URL-based installs trigger `manager_core.repo_install()` → `git_helper.py --clone`.

### Phase 2: Manager 4.1b6 — `git clone` exit 128 (2 tests failing)

4.1b6 fixed the `ModuleNotFoundError` by isolating `git_helper.py` from the `comfyui_manager` package (Option B from the original analysis was applied — `get_backup_branch_name()` inlined, `comfyui_manager` imports removed, with comment: "runs as a subprocess on Windows and must not import from comfyui_manager").

**Result**: `test_progressive_conflict` now passes. But 2 tests still fail with a **new error**.

**Remaining failures**:

| Test | Assertion | Actual |
|------|-----------|--------|
| `test_node_reinstall_uv_compile` | `"Resolving dependencies for" in combined` | Empty output — reinstall produced no dependency resolution |
| `test_node_uv_sync_standalone_conflict` | `"Conflicting packages (by node pack):" in combined` | No conflict — test packs not installed |

**New error pattern**:
```
Download: git clone 'https://github.com/ltdrdata/nodepack-test1-do-not-install'
Cmd('git') failed due to: exit code(128)
  cmdline: git clone -v --recursive --progress -- https://github.com/ltdrdata/nodepack-test1-do-not-install D:\a\...\custom_nodes\nodepack-test1-do-not-install
```

`git_helper.py` catches the exception and exits with `-1` (unsigned: `4294967295`):
```python
except Exception as e:
    print(e)       # ← only prints GitPython's summary, not git's stderr
    sys.exit(-1)
```

**Critical issue**: git's actual stderr (which would explain *why* exit 128 occurred) is **swallowed** by GitPython's exception handling. The `print(e)` only shows `Cmd('git') failed due to: exit code(128)` without the underlying error message (e.g., "destination path already exists", "could not resolve host", etc.).

**Why it's intermittent**: `cm_cli` returns exit code 0 even when `git_helper.py` subprocess fails (error is logged but not propagated). Some tests pass their `assert setup.returncode == 0` despite the node not actually being installed, then fail on the assertion that checks the expected output string.

## Root Cause Analysis: `git clone` exit 128

Probable causes (Windows-specific, exit 128):

1. **Target directory exists**: The `_clean_test_packs` fixture uses `shutil.rmtree(path, ignore_errors=True)`. On Windows, file locks (antivirus, `git.exe` handles, Python's `__pycache__`) can prevent deletion silently. The leftover directory causes `git clone` to fail.

2. **GitPython process handle leak**: `git.Repo.clone_from()` with `progress=GitProgress()` creates git subprocess handles. On Windows, these may not be released before the next test's clone attempt.

## Recommended Fixes (Manager-side, for 4.1b7)

### Fix 1: Pre-clone directory cleanup

`gitclone()` does not check if the target directory already exists. On Windows, leftover directories from failed clones cause `git clone` to fail with exit 128.

```python
# git_helper.py — gitclone()
def gitclone(custom_nodes_path, url, target_hash=None, repo_path=None):
    repo_name = os.path.splitext(os.path.basename(url))[0]
    if repo_path is None:
        repo_path = os.path.join(custom_nodes_path, repo_name)

    # Windows: clean up leftover directory from failed previous clone
    if os.path.exists(repo_path):
        import shutil
        shutil.rmtree(repo_path)

    repo = git.Repo.clone_from(url, repo_path, recursive=True, progress=GitProgress())
    ...
```

### Fix 2: Surface git stderr

The current error handler swallows git's actual error message, making debugging impossible:

```python
# Current (4.1b6) — only shows "Cmd('git') failed due to: exit code(128)"
except Exception as e:
    print(e)
    sys.exit(-1)

# Proposed — shows git's actual stderr (e.g., "fatal: destination path already exists")
except Exception as e:
    print(e, file=sys.stderr)
    if hasattr(e, 'stderr') and e.stderr:
        print(e.stderr, file=sys.stderr)
    sys.exit(1)  # use 1 instead of -1 (Windows shows -1 as 4294967295)
```

### Fix 3: Propagate install errors

`cm_cli` returns exit code 0 even when `git_helper.py` subprocess fails. The error is logged but not propagated to the caller. `comfy-cli` cannot detect whether node installation actually succeeded.

## Environment Details

- **OS**: windows-latest (Windows Server 2022)
- **Python**: 3.10.11
- **Manager 4.1b5**: `comfyui_manager/__init__.py` top-level `from comfy.cli_args import args`
- **Manager 4.1b6**: `git_helper.py` isolated, `__init__.py` unchanged, `git clone` exit 128
- **Workspace**: `D:\a\comfy-cli\comfy-cli\comfy-uv-{timestamp}`

## Related

- PR #363: feat: add ComfyUI-Manager v4 support and uv-compile unified dependency resolution
- ComfyUI PR #12957: `update-cache` feature (merged — unrelated to this issue)
- Manager 4.1b6: `git_helper.py` subprocess isolation fix (partially resolves this issue)
