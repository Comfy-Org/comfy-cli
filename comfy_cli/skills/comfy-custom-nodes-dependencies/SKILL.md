---
name: comfy-custom-nodes-dependencies
description: Packaged fallback for dependency, capability, and local-versus-Cloud custom-node constraints.
---

# Dependencies/capabilities fallback

Reuse ComfyUI's installed torch, numpy, Pillow, and related packages; declare only additional dependencies. Inspect existing pins before extending a pack, and report conflicts rather than replacing a shared dependency. Call out filesystem, network, custom endpoints, persistent state, credentials, and very large outputs. They may work locally and remain Cloud-incompatible: report the two verdicts separately. Never persist credentials in generated source.
