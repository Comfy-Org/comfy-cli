---
name: comfy-custom-nodes-validation
description: Packaged fallback for staged validation of generated custom-node code.
---

# Validation fallback

Validate in stages: static contract and Python parse; import inside ComfyUI's own Python environment; restart; discovery in `/object_info`; then one small workflow execution. filesystem presence is never proof that a node loaded. After a repair, restart and rediscover. Do not promise hot reload.

Check one registration form, unique non-empty node/input/output ids, declared callable existence, and report dependency/filesystem/network behavior. A static scan cannot prove dynamic behavior is absent.
