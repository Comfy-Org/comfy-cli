---
name: comfy-custom-nodes-frontend
description: Packaged fallback for shipping browser-side assets from a custom-node pack.
---

# Frontend extension fallback

Two declarations expose a pack's browser-side directory, and both write into one registry keyed by name: module-level `WEB_DIRECTORY` in the pack's `__init__.py`, keyed by the pack's directory name, and `[tool.comfy].web` in `pyproject.toml`, keyed by `project.name`. Either is honored only when the directory exists on disk; a wrong relative path is skipped with no error, so "my script never loads" is usually a path. `[tool.comfy].web` also requires that the runtime can read `pyproject.toml` and is skipped with only a log warning otherwise, so `WEB_DIRECTORY` has fewer preconditions.

Declare one. Declaring both resolves by name rather than merging: equal names — the common case — mean `WEB_DIRECTORY` applies second and silently drops the `pyproject.toml` entry; differing names serve both prefixes and the second registration of one name raises.

Files are served at `/extensions/<name>/…`, and the listing the frontend loads enumerates `**/*.js` only, so build TypeScript or JSX to `.js`. An extension's `name` must be unique; registering it twice raises.

Treat frontend state as presentation only. Every non-secret value the Python node reads must live in the persisted widget value, because the browser layer is absent for an API-submitted prompt. Never put credentials in widget values — read secrets from the environment or the runtime's secret mechanism.
