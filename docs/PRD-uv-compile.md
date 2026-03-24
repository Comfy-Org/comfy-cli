# PRD: Unified Dependency Resolution (--uv-compile) Support

## Overview

Add `--uv-compile` flag to comfy-cli to integrate with ComfyUI-Manager v4.1+'s
Unified Dependency Resolver. Users can batch-resolve all custom node dependencies
via `uv pip compile` after install/update operations.

## Background

### Problem

Each ComfyUI custom node ships its own `requirements.txt`. The default approach
(`pip install` per node) frequently causes dependency conflicts between nodes.
ComfyUI-Manager v4.1+ introduced `UnifiedDepResolver` to solve this, but
comfy-cli had no way to invoke it.

### Existing Approaches

| Approach | Flag | Implementation | Behavior |
|----------|------|----------------|----------|
| Default | (none) | cm_cli | Per-node `pip install` |
| Fast deps | `--fast-deps` | comfy-cli `DependencyCompiler` | comfy-cli side `uv pip compile` |
| No deps | `--no-deps` | cm_cli | Skip dependency installation |
| **Unified** | **`--uv-compile`** | **cm_cli `UnifiedDepResolver`** | **cm_cli side batch resolution** |

### Target Users

- ComfyUI-Manager v4.1+ users
- Users managing many custom nodes
- Users experiencing dependency conflicts

## Requirements

### FR-1: Add --uv-compile flag to 7 commands

**Target commands:**

| # | Command | Existing dep flags |
|---|---------|-------------------|
| 1 | `comfy node install` | `--fast-deps`, `--no-deps` |
| 2 | `comfy node reinstall` | `--fast-deps` |
| 3 | `comfy node update` | (none) |
| 4 | `comfy node fix` | (none) |
| 5 | `comfy node restore-snapshot` | (none) |
| 6 | `comfy node restore-dependencies` | (none) |
| 7 | `comfy node install-deps` | (none) |

**Behavior:** When the flag is passed, append `--uv-compile` to the cm_cli
subprocess command.

### FR-2: Standalone uv-sync command

```
comfy node uv-sync
```

Directly invokes cm_cli's `uv-sync` subcommand. Batch-resolves all installed
custom node dependencies without requiring a prior install/update operation.

### FR-3: --no-uv-compile flag

Add `--no-uv-compile` to all 7 commands so users can explicitly disable the
config default on a per-command basis.

### FR-4: Config default setting

```
comfy manager uv-compile-default true   # Enable by default
comfy manager uv-compile-default false  # Disable by default
```

Once enabled, `--uv-compile` is automatically applied to all custom node
operations.

### FR-5: Mutual exclusivity

`--uv-compile`, `--fast-deps`, and `--no-deps` are mutually exclusive.

| Combination | Result |
|-------------|--------|
| `--uv-compile --fast-deps` | Error |
| `--uv-compile --no-deps` | Error |
| `--fast-deps --no-deps` | Error |
| config default + `--fast-deps` | `--fast-deps` wins (no error) |
| config default + `--no-uv-compile` | Disabled |

### NFR-1: Backward compatibility

- No impact on existing `--fast-deps` / `--no-deps` behavior
- Without flag and without config, behavior is identical to before (per-node pip install)

### NFR-2: Minimum version

Requires ComfyUI-Manager v4.1+. On older versions, cm_cli returns its own
error for the unknown flag. comfy-cli does not perform version checking.

## Out of Scope

- `comfy install` (core ComfyUI installation) — separate dependency system
- Modifications to cm_cli's internal UnifiedDepResolver logic
- Automatic ComfyUI-Manager version detection
