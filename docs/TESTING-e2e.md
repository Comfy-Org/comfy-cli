# E2E Testing Guide

E2E tests perform real `comfy install`, `comfy launch`, and `comfy node` operations.
They are **disabled by default** and must be explicitly enabled.

## Environment variables

| Variable | Default | Description |
|----------|---------|-------------|
| `TEST_E2E` | `false` | Set to `true` to enable E2E tests |
| `TEST_E2E_COMFY_URL` | *(empty — uses default)* | Custom ComfyUI repo URL. Supports `@branch` syntax |
| `TEST_E2E_COMFY_INSTALL_FLAGS` | `--cpu` | Extra flags passed to `comfy install` |
| `TEST_E2E_COMFY_LAUNCH_FLAGS_EXTRA` | `--cpu` | Extra flags passed to `comfy launch` |

## Basic usage

```bash
TEST_E2E=true pytest tests/e2e/
```

Installs ComfyUI from the default upstream (`Comfy-Org/ComfyUI`), launches it
in the background, runs the test suite, then stops the server.

## Pre-release testing

To test features that depend on unreleased ComfyUI changes (e.g.
`manager_requirements.txt` not yet merged upstream), point the E2E suite at a
fork/branch:

```bash
TEST_E2E=true \
TEST_E2E_COMFY_URL="https://github.com/ltdrdata/ComfyUI.git@dr-bump-manager" \
pytest tests/e2e/ -v
```

This clones `ltdrdata/ComfyUI` at the `dr-bump-manager` branch, which contains
`manager_requirements.txt` for pip-based Manager v4 installation.

### Full pre-release run (GPU)

```bash
TEST_E2E=true \
TEST_E2E_COMFY_URL="https://github.com/ltdrdata/ComfyUI.git@dr-bump-manager" \
TEST_E2E_COMFY_INSTALL_FLAGS="" \
TEST_E2E_COMFY_LAUNCH_FLAGS_EXTRA="" \
pytest tests/e2e/ -v
```

## Test suites

### `test_e2e.py` — General functionality

Covers model download, custom node lifecycle, workflow execution, and basic
Manager v4 smoke tests.

| Test | Description |
|------|-------------|
| `test_model` | Download, list, and remove a model |
| `test_node` | Install, reinstall, show, update, disable, enable, publish a custom node |
| `test_manager_installed` | Verifies `cm_cli` is importable after install |
| `test_node_uv_compile` | Installs a node with `--uv-compile` and runs `comfy node uv-sync` |
| `test_uv_compile_default_config` | Sets `uv-compile-default`, verifies `comfy env` display |
| `test_run` | Downloads a checkpoint and executes a workflow end-to-end |

### `test_e2e_uv_compile.py` — Unified dependency resolution

Comprehensive `--uv-compile` E2E suite. **Requires Manager v4.1+** — automatically
skipped when `cm_cli` is not importable.

#### Test packs

Two categories of node packs are used:

- **Real packs** (`comfyui-impact-pack`, `comfyui-inspire-pack`) — production
  node packs for verifying normal installation succeeds without conflicts.
- **Conflict fixture packs** (`nodepack-test1-do-not-install`,
  `nodepack-test2-do-not-install`) — ltdrdata's dedicated test packs that
  intentionally conflict on ansible versions (`ansible==9.13.0` vs
  `ansible-core==2.14.0`). Contain no executable code.

Supply-chain safety: only node packs from verified authors (ltdrdata,
comfyanonymous, Comfy-Org) are used.

#### Test scenarios

**Normal installation (real packs)**

| Test | Scenario | Packs |
|------|----------|-------|
| `test_real_packs_sequential_no_conflict` | Install two real packs one-by-one with `--uv-compile` — each resolves successfully, no conflicts | impact, inspire |
| `test_real_packs_simultaneous_no_conflict` | Install two real packs in a single command with `--uv-compile` — resolves successfully, no conflicts | impact, inspire |

**Progressive conflict**

| Test | Scenario | Packs |
|------|----------|-------|
| `test_progressive_conflict` | Install real packs (OK) → add conflict-pack-1 (still OK) → add conflict-pack-2 (conflict detected with attribution) | impact, inspire, test1, test2 |

**Command coverage (--uv-compile flag on each command)**

| Test | Scenario | Packs |
|------|----------|-------|
| `test_node_reinstall_uv_compile` | Reinstall an installed pack with `--uv-compile` — resolution runs | test1 |
| `test_node_update_uv_compile` | Update an installed pack with `--uv-compile` — resolution runs | test1 |
| `test_node_fix_uv_compile` | Fix an installed pack with `--uv-compile` — resolution runs | test1 |
| `test_node_restore_deps_uv_compile` | `restore-dependencies --uv-compile` — resolution runs | test1 |

**Standalone uv-sync**

| Test | Scenario | Packs |
|------|----------|-------|
| `test_node_uv_sync_standalone` | `comfy node uv-sync` resolves installed pack dependencies | test1 |
| `test_node_uv_sync_standalone_conflict` | `comfy node uv-sync` with conflicting packs — shows conflict attribution | test1, test2 |

**Config default and overrides**

| Test | Scenario | Packs |
|------|----------|-------|
| `test_uv_compile_config_default` | `uv-compile-default true` → install without flag triggers resolution | test1 |
| `test_no_uv_compile_overrides_config` | Config default enabled, `--no-uv-compile` overrides — resolution does not run | test1 |

**Mutual exclusivity**

| Test | Scenario | Packs |
|------|----------|-------|
| `test_uv_compile_mutual_exclusivity` | `--uv-compile` with `--fast-deps` or `--no-deps` — rejected with error | test1 |

#### Fixtures and isolation

- `workspace` (module-scoped): installs ComfyUI, launches server in background,
  yields workspace path, stops server on teardown.
- `comfy_cli`: returns `comfy --workspace {ws}` command prefix.
- `_clean_test_packs` (autouse, function-scoped): removes conflict fixture packs
  before and after each test. Real packs are **not** removed between tests
  (they persist in the workspace).
- Config default tests use `try/finally` to restore the setting after each test.

## Notes

- E2E tests create a temporary workspace directory (`comfy-<timestamp>`) in the
  current working directory. It is **not** automatically cleaned up.
- Each test file has its own `workspace` fixture (`module`-scoped) — all tests
  within a file share a single ComfyUI installation.
- Tests that require Manager v4 are automatically skipped when `cm_cli` is not
  importable.
