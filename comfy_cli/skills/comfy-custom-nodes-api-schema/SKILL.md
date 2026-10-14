---
name: comfy-custom-nodes-api-schema
description: Packaged fallback for ComfyUI V3 and legacy V1 custom-node schema shape.
---

# API/schema fallback

V3 uses `io.ComfyNode`, `io.Schema`, `ComfyExtension`, and module-level `comfy_entrypoint`. `define_schema` and `execute` are classmethods; execute parameter names equal input ids; return `io.NodeOutput` in output order. `node_id` is stable workflow identity; `display_name` may change. Useful schema tools include `io.Autogrow`, `io.DynamicCombo`, and `io.MatchType`; previews belong in `io.NodeOutput`, progress uses the runtime execution API, and terminal nodes set `is_output_node`. Keep tensor batch dimensions: IMAGE `[B,H,W,C]`, MASK `[B,H,W]`; do not assume a fixed LATENT shape.

V1 uses `INPUT_TYPES`, `RETURN_TYPES`, `FUNCTION`, `CATEGORY`, and `NODE_CLASS_MAPPINGS`. Translate deliberately; do not mix registrations.

If a pack ships frontend code, export `WEB_DIRECTORY` and keep saved widget values JSON-like and
stable. Preserve old workflow identity during V1-to-V3 migration by registering the
`node_replacement`; constructing a replacement descriptor alone does nothing.
