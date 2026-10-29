---
name: comfy-custom-nodes-troubleshooting
description: Packaged fallback for stage-specific custom-node troubleshooting.
---

# Troubleshooting fallback

- Parse/contract failure: repair source or identifiers.
- Import failure: use the startup traceback and ComfyUI's exact Python.
- Clean import but absent `/object_info`: repair registration or install path.
- Registered but execution fails: inspect traceback, shapes/types, and exact dependency versions.
- Search unavailable: continue only if local catalog and documented API settle the contract; otherwise stop instead of inventing an integration.

Restart and rediscover after every repair.
