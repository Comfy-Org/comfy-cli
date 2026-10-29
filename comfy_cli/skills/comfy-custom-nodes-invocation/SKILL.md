---
name: comfy-custom-nodes-invocation
description: Packaged fallback for deciding whether and how to enter custom-node authoring.
---

# Invocation fallback

Use this path for writing, fixing, reviewing, validating, packaging, or troubleshooting ComfyUI custom-node code. Prefer V3 for a new self-contained pack. V1 is legacy but supported and may be correct when extending a V1 pack or adapting an ecosystem reference. Use one registration form per pack.

Inspect the installed catalog first. Load only the next class needed by the current stage. Never claim Cloud compatibility from local validity.

For a publishable pack keep one folder under `custom_nodes/` with `__init__.py`, node modules,
optional `web/`, and `pyproject.toml`. `comfy node init` creates packaging metadata, not node
source. `comfy node publish --token <token>` publishes the version in `pyproject.toml`; never put
the token in source, widget values, or a workflow. Confirm every tracked file is intended for the
package before publishing.
