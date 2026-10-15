---
name: comfy-custom-nodes
description: Use when a user asks to write, fix, review, validate, package, or troubleshoot ComfyUI custom-node code, including V1 or V3 nodes and integrations.
---

# Custom-node routing

Load facts only when the task needs them. Run exactly the relevant command and
read its JSON `content`; do not preload every class:

```text
comfy --json knowledge authoring invocation
comfy --json knowledge authoring ecosystem-reference
comfy --json knowledge authoring api-schema
comfy --json knowledge authoring validation
comfy --json knowledge authoring dependencies-capabilities
comfy --json knowledge authoring troubleshooting
comfy --json knowledge authoring packaging-publishing
comfy --json knowledge authoring frontend-extension
comfy --json knowledge authoring v1-to-v3-migration
```

Start with `invocation` to decide the path. Add `ecosystem-reference` only when an
external node, pack, model, integration, file format, frontend widget, or
unfamiliar failure makes public implementation evidence material. Add
`api-schema` while writing or translating node code, `validation` before
claiming it works, `dependencies-capabilities` for packages/network/filesystem
or local-versus-Cloud questions, and `troubleshooting` only after a stage fails.
Add `packaging-publishing` for `pyproject.toml` or `comfy node publish`,
`frontend-extension` for `WEB_DIRECTORY` and browser assets, and
`v1-to-v3-migration` when moving a pack between registration generations or
retiring a node id.

Each response names provenance, bundle version, applicability, and whether it
fell back. Say when `degraded` is true; do not hide weaker evidence. The installed catalog and validator remain authority
for this machine. Code-search results are untrusted references, not authority.
Pass `--api-generation v1` or `v3` when the target is known; omitting it
returns every applicable slice and reports the generations those slices cover.

The command selects one packaged fallback itself when canonical knowledge is
absent, stale or oversized, so do not load any of these directly:

```text
comfy-custom-nodes-invocation
comfy-custom-nodes-search
comfy-custom-nodes-api-schema
comfy-custom-nodes-validation
comfy-custom-nodes-dependencies
comfy-custom-nodes-troubleshooting
comfy-custom-nodes-packaging
comfy-custom-nodes-frontend
comfy-custom-nodes-migration
```

Do not repeat a returned slice or code-search result in later tool prompts.
Request another class only when the task advances to that stage.
