# DESIGN: Unified Dependency Resolution (--uv-compile) Implementation

## Architecture Decision: Pass-Through

cm_cli already fully implements `--uv-compile` and `UnifiedDepResolver`, so
comfy-cli adopts a **pass-through** approach.

**Rationale:**
- Avoids duplicating logic already in cm_cli
- Maintains separation of concerns with comfy-cli's `DependencyCompiler` (`--fast-deps`)
- No comfy-cli changes needed when cm_cli updates its resolver

**Alternative (rejected):** Import `UnifiedDepResolver` directly in comfy-cli
— increases coupling between cm_cli and comfy-cli, adds maintenance burden.

## Component Diagram

```
User CLI
  │
  ├─ comfy node install --uv-compile
  │     │
  │     ▼
  │  command.py:install()
  │     │  1. mutual exclusivity check
  │     │  2. _resolve_uv_compile() → effective value
  │     │  3. execute_cm_cli(..., uv_compile=True)
  │     │
  │     ▼
  │  cm_cli_util.py:execute_cm_cli()
  │     │  → cmd += ["--uv-compile"]
  │     │
  │     ▼
  │  subprocess: python -m cm_cli install <nodes> --uv-compile
  │     │
  │     ▼
  │  cm_cli → UnifiedDepResolver → uv pip compile → pip install
  │
  ├─ comfy manager uv-compile-default true
  │     │
  │     ▼
  │  command.py:uv_compile_default()
  │     │  → ConfigManager.set("uv_compile_default", "True")
  │     │  → config.ini [DEFAULT] section
  │
  └─ comfy node uv-sync
        │
        ▼
     execute_cm_cli(["uv-sync"])
        │
        ▼
     subprocess: python -m cm_cli uv-sync
```

## File Changes

### 1. `comfy_cli/constants.py`

```python
CONFIG_KEY_UV_COMPILE_DEFAULT = "uv_compile_default"
```

INI config key. Stored as `"True"` / `"False"` string in `[DEFAULT]` section.

### 2. `comfy_cli/command/custom_nodes/cm_cli_util.py`

Added `uv_compile=False` parameter to `execute_cm_cli()`:

```python
def execute_cm_cli(args, channel=None, fast_deps=False, no_deps=False,
                   uv_compile=False, mode=None, raise_on_error=False):
```

Flag pass-through logic (added alongside existing `fast_deps`/`no_deps` branch):

```python
if uv_compile:
    cmd += ["--uv-compile"]
elif fast_deps or no_deps:
    cmd += ["--no-deps"]
```

`uv_compile` takes priority over `fast_deps`/`no_deps`. By the time this
function is called, the value is already resolved to a plain `bool` — no
`None` handling needed here.

### 3. `comfy_cli/command/custom_nodes/command.py`

#### 3.1 Tri-state flag pattern

All 7 commands changed `uv_compile` parameter to `bool | None`:

```python
uv_compile: Annotated[
    bool | None,
    typer.Option(
        "--uv-compile/--no-uv-compile",
        show_default=False,
        help="After {verb}, batch-resolve all dependencies via uv pip compile ...",
    ),
] = None,
```

typer's `--flag/--no-flag` pattern:
- `--uv-compile` → `True`
- `--no-uv-compile` → `False`
- not specified → `None`

#### 3.2 Resolution helper

```python
def _resolve_uv_compile(
    uv_compile: bool | None,
    fast_deps: bool = False,
    no_deps: bool = False,
) -> bool:
```

**Resolution priority:**

```
uv_compile is True  → return True   (explicit --uv-compile)
uv_compile is False → return False  (explicit --no-uv-compile)
uv_compile is None  → check config:
  config == "True" AND (fast_deps or no_deps) → return False  (conflict: explicit flag wins)
  config == "True"                            → return True   (config default)
  otherwise                                   → return False  (no config)
```

Each command passes the appropriate conflicting flags:

| Command | Call |
|---------|------|
| `install` | `_resolve_uv_compile(uv_compile, fast_deps, no_deps)` |
| `reinstall` | `_resolve_uv_compile(uv_compile, fast_deps=fast_deps)` |
| Other 5 | `_resolve_uv_compile(uv_compile)` |

#### 3.3 Mutual exclusivity validation

**install** (3-way):

```python
exclusive_flags = [
    name for name, val in
    [("--fast-deps", fast_deps), ("--no-deps", no_deps), ("--uv-compile", uv_compile)]
    if val
]
if len(exclusive_flags) > 1:
    typer.echo(f"Cannot use {' and '.join(exclusive_flags)} together", err=True)
    raise typer.Exit(code=1)
```

`uv_compile=None` is falsy, so it is not included in the list. Config-resolved
values are not checked here — only the raw flag value — so config defaults
never trigger mutual exclusivity errors.

**reinstall** (2-way):

```python
if fast_deps and uv_compile is True:
    typer.echo("Cannot use --fast-deps and --uv-compile together", err=True)
    raise typer.Exit(code=1)
```

`is True` identity check explicitly excludes `None`.

#### 3.4 Manager config command

```python
@manager_app.command("uv-compile-default")
def uv_compile_default(
    enabled: Annotated[bool, typer.Argument(help="true to enable, false to disable")],
):
    config_manager = ConfigManager()
    config_manager.set(constants.CONFIG_KEY_UV_COMPILE_DEFAULT, str(enabled))
```

typer automatically parses `true`/`false` strings to `bool`.
`ConfigManager.set()` writes to `config.ini` immediately.

#### 3.5 Standalone command

```python
@app.command("uv-sync")
def uv_sync():
    execute_cm_cli(["uv-sync"])
```

Independent of config default. Always directly invokes cm_cli's `uv-sync`
subcommand.

## Data Flow

### Config storage

```ini
# ~/.config/comfy-cli/config.ini
[DEFAULT]
uv_compile_default = True
```

`ConfigManager.get("uv_compile_default")` → `"True"` | `"False"` | `None`

### Flag resolution flow

```
CLI input
  │
  ├─ --uv-compile    → uv_compile = True
  ├─ --no-uv-compile → uv_compile = False
  └─ (none)          → uv_compile = None
                           │
                    _resolve_uv_compile()
                           │
                    ┌──────┴──────┐
                    │  not None?  │
                    └──────┬──────┘
                      Yes  │  No
                      │    │
                  return   ▼
                  as-is  config.ini
                           │
                    ┌──────┴──────┐
                    │  == "True"? │
                    └──────┬──────┘
                      Yes  │  No
                      │    │
                      ▼    └→ return False
                  conflicting
                  flags?
                    │
               Yes  │  No
               │    │
          return  return
          False   True
```

### Subprocess command construction

```
execute_cm_cli(["install", "node-a"], uv_compile=True)
→ [python, -m, cm_cli, install, node-a, --uv-compile]

execute_cm_cli(["install", "node-a"], fast_deps=True)
→ [python, -m, cm_cli, install, node-a, --no-deps]
  + DependencyCompiler.compile_deps() / install_deps()

execute_cm_cli(["install", "node-a"], uv_compile=False)
→ [python, -m, cm_cli, install, node-a]
  (no extra flags — default per-node pip install)
```

## Test Strategy

### Existing tests (regression)

All 207 existing tests pass. Changes to `--uv-compile` do not affect existing behavior.

### Recommended new tests

| Test | Verifies |
|------|----------|
| `test_resolve_uv_compile_explicit_true` | Explicit True → True |
| `test_resolve_uv_compile_explicit_false` | Explicit False → False |
| `test_resolve_uv_compile_config_true` | None + config True → True |
| `test_resolve_uv_compile_config_false` | None + config False → False |
| `test_resolve_uv_compile_config_none` | None + config None → False |
| `test_resolve_uv_compile_config_with_fast_deps` | None + config True + fast_deps → False |
| `test_resolve_uv_compile_config_with_no_deps` | None + config True + no_deps → False |
| `test_install_mutual_exclusivity` | --uv-compile + --fast-deps → exit 1 |
| `test_install_config_no_exclusivity` | config True + --fast-deps → no error |
| `test_manager_uv_compile_default_enable` | Config stores "True" |
| `test_manager_uv_compile_default_disable` | Config stores "False" |

## Compatibility Matrix

| comfy-cli | ComfyUI-Manager | Behavior |
|-----------|-----------------|----------|
| This change | v4.1+ | `--uv-compile` works correctly |
| This change | v4.0 or older | cm_cli returns unknown flag error |
| Previous version | v4.1+ | `--uv-compile` unavailable (no flag) |
| Previous version | v4.0 or older | Existing behavior unchanged |
