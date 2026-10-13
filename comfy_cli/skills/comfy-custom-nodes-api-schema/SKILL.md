---
name: comfy-custom-nodes-api-schema
description: Packaged fallback for ComfyUI V3 and legacy V1 custom-node schema shape.
---

# API/schema fallback

V3 uses `io.ComfyNode`, `io.Schema`, `ComfyExtension`, and module-level `comfy_entrypoint`. `define_schema` and `execute` are classmethods; execute parameter names equal input ids; return `io.NodeOutput` in output order. `node_id` is stable workflow identity; `display_name` may change. Keep tensor batch dimensions: IMAGE `[B,H,W,C]`, MASK `[B,H,W]`, LATENT samples `[B,C,H/8,W/8]`.

V1 uses `INPUT_TYPES`, `RETURN_TYPES`, `FUNCTION`, `CATEGORY`, and `NODE_CLASS_MAPPINGS`. Translate deliberately; do not mix registrations.
