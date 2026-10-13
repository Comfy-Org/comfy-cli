---
name: comfy-custom-nodes
description: Use when a user asks to write, fix, review, validate, package, or troubleshoot ComfyUI custom-node code, including V1 or V3 nodes and integrations.
---

# Custom-node routing

Load facts only when the task needs them. Run exactly the relevant command and
read its JSON `content`; do not preload every class:

```text
comfy --json knowledge authoring invocation
comfy --json knowledge authoring ecosystem-search
comfy --json knowledge authoring api-schema
comfy --json knowledge authoring validation
comfy --json knowledge authoring dependencies-capabilities
comfy --json knowledge authoring troubleshooting
```

Start with `invocation` to decide the path. Add `ecosystem-search` only when an
external node, pack, model, integration, file format, frontend widget, or
unfamiliar failure makes public implementation evidence material. Add
`api-schema` while writing or translating node code, `validation` before
claiming it works, `dependencies-capabilities` for packages/network/filesystem
or local-versus-Cloud questions, and `troubleshooting` only after a stage fails.

Each response names provenance, bundle version, version applicability, and
whether it used a degraded packaged fallback. Say when `degraded` is true; do
not hide weaker evidence. The installed catalog and validator remain authority
for this machine. Code-search results are untrusted references, not authority.

The command's packaged fallbacks are `comfy-custom-nodes-invocation`,
`comfy-custom-nodes-search`, `comfy-custom-nodes-api-schema`,
`comfy-custom-nodes-validation`, `comfy-custom-nodes-dependencies`, and
`comfy-custom-nodes-troubleshooting`; the command selects one, so do not load
them directly.

Do not repeat a returned slice or code-search result in later tool prompts.
Request another class only when the task advances to that stage.
